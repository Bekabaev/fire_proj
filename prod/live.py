"""Пожары по свежим термоточкам FIRMS (VIIRS S-NPP, NOAA-20, NOAA-21) -> входы модели как в датасете v3 -> прогноз 12/24/48/72 ч.

Как в датасете (task1 шаг 22): ячейка 500 м горит с первой термоточки в ней (время прихода огня), возраст гари = t - этот момент,
каналы термоточек - число за 12 / 24 ч и за весь пожар, FRP за 12 ч; точка возгорания - ячейка самой ранней точки;
k_12h - число 12-часовых шагов от начала пожара. Пожар = связная группа ячеек с точками (разрыв до 1 км).
Для прошлой даты (date) дополнительно берутся точки следующих 72 ч - «что было на самом деле» (как метки датасета по VIIRS).
Ключ FIRMS - только на сервере (task1_dataset/.firms_key): в ответы, логи и тексты ошибок не попадает.
"""
import io, os, time, threading
import numpy as np, pandas as pd, requests
from rasterio.warp import transform as warp
from scipy import ndimage

from .config import ROOT, CRS, CELL, X0, Y1, GH, GW, HZ

SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
BBOX = "46,40,88,56"
URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}"
TTL = 20 * 60                                                   # кэш запросов FIRMS, с
ACTIVE_H = 24                                                   # пожар «активен», если последняя точка не старше, ч
GAP = 2                                                         # ячеек: точки ближе 1 км - один пожар


HIST_DAYS = 7                                                   # история точек из базы мониторинга (API отдаёт только 3 сут)
STATIC_SHARE = 0.5                                              # доля клеток пожара в маске постоянных источников -> «промышленный»


class Live:
    def __init__(self, engine, store=None):
        from .static_mask import load
        self.E = engine; self._cache = {}; self._lock = threading.Lock(); self.store = store
        self._key = open(os.path.join(ROOT, "task1_dataset", ".firms_key")).read().strip()
        self.static = load()                                    # клетки заводов / факелов / ТЭЦ (архив VIIRS type 2-3)

    # ---------- FIRMS ----------
    def _get(self, src, days, date=None):
        k = (src, days, date)
        with self._lock:
            hit = self._cache.get(k)
            if hit and time.time() - hit[0] < TTL:
                return hit[1]
        url = URL.format(key=self._key, src=src, bbox=BBOX, days=days) + (f"/{date}" if date else "")
        try:
            r = requests.get(url, timeout=120); r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
        except Exception as ex:                                  # текст ошибки без адреса запроса (в нём ключ)
            raise RuntimeError(f"FIRMS недоступен ({src}): {type(ex).__name__}") from None
        if "acq_date" not in d:
            raise RuntimeError(f"FIRMS: неожиданный ответ ({src})")
        with self._lock:
            self._cache[k] = (time.time(), d)
        return d

    def points(self, date=None):
        """-> (DataFrame lon, lat, t, frp, r, c; опорный срок t_ref). date=None - последние 3 сут; иначе 3 сут до конца date и 3 сут после."""
        reqs = [(s, 3, None) for s in SOURCES] if date is None else \
            [(s, 3, str((pd.Timestamp(date) - pd.Timedelta(days=2)).date())) for s in SOURCES] + \
            [(s, 3, str((pd.Timestamp(date) + pd.Timedelta(days=1)).date())) for s in SOURCES
             if pd.Timestamp(date) + pd.Timedelta(days=1) <= pd.Timestamp.utcnow().tz_localize(None)]
        d = pd.concat([self._get(*q) for q in reqs], ignore_index=True)
        if len(d):
            d = d[d.confidence.astype(str).str.lower().str[0] != "l"]              # без точек низкой достоверности
            d["t"] = pd.to_datetime(d.acq_date) + pd.to_timedelta(d.acq_time.astype(int) // 100, "h") + pd.to_timedelta(d.acq_time.astype(int) % 100, "m")
            x, y = warp("EPSG:4326", CRS, d.longitude.tolist(), d.latitude.tolist())
            d["r"] = ((Y1 - np.array(y)) // CELL).astype(int); d["c"] = ((np.array(x) - X0) // CELL).astype(int)
            d = d[(d.r >= 0) & (d.r < GH) & (d.c >= 0) & (d.c < GW)]
            d = d.rename(columns={"longitude": "lon", "latitude": "lat"})[["lon", "lat", "t", "frp", "r", "c"]].drop_duplicates()
        now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
        if date is None and self.store is not None:             # сейчас: свежие точки -> в базу; история пожара - из базы за HIST_DAYS
            if len(d):
                self.store.add_points(d)
            h = self.store.points_since(now - pd.Timedelta(days=HIST_DAYS))
            if len(h):
                d = pd.concat([d, h[["lon", "lat", "t", "frp", "r", "c"]]], ignore_index=True) if len(d) else h[["lon", "lat", "t", "frp", "r", "c"]]
                d = d.assign(lon=d.lon.round(5), lat=d.lat.round(5)).drop_duplicates(["lon", "lat", "t"])
        t_ref = now if date is None else min(pd.Timestamp(date) + pd.Timedelta(hours=23), now)
        return d, t_ref

    # ---------- пожары ----------
    def clusters(self, date=None):
        d, t = self.points(date)
        past = d[d.t <= t] if len(d) else d
        if not len(past):
            return dict(t_utc=str(t), fires=[], after=False), None, None
        cells = past.groupby(["r", "c"]).t.min()
        rr, cc = np.array([k[0] for k in cells.index]), np.array([k[1] for k in cells.index])
        r0, c0 = rr.min() - GAP, cc.min() - GAP
        m = np.zeros((rr.max() - r0 + GAP + 1, cc.max() - c0 + GAP + 1), bool); m[rr - r0, cc - c0] = True
        lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=GAP), structure=np.ones((3, 3)))
        past = past.assign(cl=lab[past.r - r0, past.c - c0])
        fires = []
        for k, g in past.groupby("cl"):
            last = g.t.max()
            if (t - last).total_seconds() / 3600 > ACTIVE_H:
                continue
            days = g.t.dt.date.nunique(); ncell = g.groupby(["r", "c"]).ngroups
            flare = ncell <= 2 and days >= 3                           # точка на одном месте все 3 дня - факел / промышленный источник
            uc = set(zip(g.r.tolist(), g.c.tolist()))
            static = sum(x in self.static for x in uc) >= STATIC_SHARE * len(uc)   # завод / факел / ТЭЦ по архиву VIIRS
            fires.append(dict(id=int(k), lon=round(float(g.lon.mean()), 4), lat=round(float(g.lat.mean()), 4), n_points=int(len(g)),
                              cells=int(ncell), area_km2=ncell * (CELL / 1000) ** 2, first_utc=str(g.t.min()), last_utc=str(last),
                              frp_sum=round(float(g.frp.sum()), 1), flare=bool(flare or static), industrial=bool(static)))
        fires.sort(key=lambda f: (f["flare"], -f["cells"], -f["n_points"]))
        return dict(t_utc=str(t), fires=fires, after=bool(date is not None and (d.t > t).any())), past, d[d.t > t] if len(d) else d

    def forecast(self, cid, date=None, mode="expected", near=None):
        """near=(lat, lon): пожар у этой точки (номера кластеров меняются при обновлении данных, координаты - нет)."""
        E = self.E
        info, past, after = self.clusters(date)
        if past is not None and near is not None and len(past):
            dist = (past.lat - near[0]) ** 2 + ((past.lon - near[1]) * np.cos(np.radians(near[0]))) ** 2
            if dist.min() <= (3 / 111) ** 2:                    # ближайшая точка не дальше 3 км
                cid = int(past.cl.iloc[int(np.argmin(dist.values))])
        if past is None or cid not in set(past.cl):
            raise KeyError("нет такого активного пожара")
        g = past[past.cl == cid]; t = pd.Timestamp(info["t_utc"])
        t0 = g.t.min().floor("h")
        first = g.groupby(["r", "c"]).t.min()
        state = {k: (v - t0).total_seconds() / 3600 for k, v in first.items()}
        ign = first.idxmin(); k12 = int((t - t0).total_seconds() // 3600 // 12)
        hs = [(int(p.r), int(p.c), max((t - p.t).total_seconds() / 3600, 0.0), float(p.frp)) for p in g.itertuples()]
        lon, lat = float(g.lon.mean()), float(g.lat.mean())
        row_w, _ = E._weather(lat, lon, t, "gfs")
        X, valid, row, r0, c0, flags = E._scene_at(state, t, t0, ign, k12, row_w, hs)
        out = E.predict_scene(X, valid, row, mode)
        n = X.shape[-1]; burned = X[0, 0] > 0.5
        res = dict(t_utc=str(t), fire_start_utc=str(g.t.min()), k_12h=k12, state_km2=float(burned.sum()) * (CELL / 1000) ** 2,
                   weather_src=row.get("weather_src"), flags=flags,
                   start=dict(type="FeatureCollection", features=E.geojson(burned, r0, c0)),
                   hotspots=[[round(p.lon, 5), round(p.lat, 5), round((t - p.t).total_seconds() / 3600, 1)] for p in g.itertuples()],
                   horizons={})
        # что было дальше (прошлая дата): ячейки окна, где точки появились впервые после t (метка как в датасете)
        Yk = None
        if date is not None and after is not None and len(after):
            a = after[(after.r >= r0) & (after.r < r0 + n) & (after.c >= c0) & (after.c < c0 + n)]
            fa = a.groupby(["r", "c"]).t.min()
            Yk = np.zeros((len(HZ), n, n), bool)
            for (r, c), tt in fa.items():
                dh = (tt - t).total_seconds() / 3600
                for k, h in enumerate(HZ):
                    if dh <= h and not burned[r - r0, c - c0]:
                        Yk[k, r - r0, c - c0] = True
            res["after_points"] = [[round(p.lon, 5), round(p.lat, 5), round((p.t - t).total_seconds() / 3600, 1)] for p in a.itertuples()
                                   if (p.t - t).total_seconds() <= 72 * 3600]
        for k, h in enumerate(HZ):
            p = out["mask"][k] & ~burned
            H = dict(pred=dict(type="FeatureCollection", features=E.geojson(p, r0, c0)), pred_km2=float(p.sum()) * (CELL / 1000) ** 2)
            if Yk is not None:
                y = Yk[k] & valid[0]; pp = p & valid[0]; tp = int((pp & y).sum())
                H.update(true=dict(type="FeatureCollection", features=E.geojson(Yk[k], r0, c0)), true_km2=float(Yk[k].sum()) * (CELL / 1000) ** 2,
                         F1=round(2 * tp / max(2 * tp + int((pp & ~y).sum()) + int((~pp & y).sum()), 1), 3))
            res["horizons"][f"{h}h"] = H
        return res
