"""Прод-движок прогноза и симуляции распространения степного пожара (Казахстан, сетка 500 м).

Модель - гибрид (task2 шаг 20): U-Net FrontV3 с погодой (wx) + физика FBP O-1b с эллипсом по ветру, стекинг по ячейкам
на каждом горизонте 12/24/48/72 ч, изотоническая калибровка и порог по val 2022 на прогнозе GFS.
- predict(): прогноз от текущего состояния пожара (маска гари / термоточки) на 12/24/48/72 ч;
- simulate(): сценарий «что если» от точки возгорания с прогнозом или заданной погодой; дольше 72 ч - прокрутка шагами 24 ч
  (состояние после шага -> новая сцена, окно сдвигается за пожаром).
Выход: вероятности (калиброванные, доля выгорающих ячеек), маски по порогу, площади, время прихода огня по физике, контуры GeoJSON.
"""
import json, math, os, sys
import numpy as np, pandas as pd, torch
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from .config import MODELS, SCRIPTS_T2, HZ, CELL, N, X0, Y1, CRS, GW, GH
from .landscape import Landscape
from . import scene as sc, weather as wx

sys.path.insert(0, SCRIPTS_T2)
from common_v3 import FrontV3, nn_maps, base_scalars, weather_vec, CH, UV, rot_wind, phys_maps   # noqa: E402
import sim_physics as sp                                                # noqa: E402

TMAX = 1000.0
SIM_N = 192                                                    # окно симулятора 96 км (физика); прогноз - 96 ячеек, как в обучении


class FireEngine:
    def __init__(self, models=MODELS, device=None, landscape=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(models, "prod_hybrid.json")))
        # U-Net: одна модель или ансамбль (итерации 05.11, ITERATIONS.md) - блок deploy.unet {models, tta}; по умолчанию wx без TTA
        uc = cfg["deploy"].get("unet", {"models": ["wx"], "tta": False}); self.tta = bool(uc.get("tta"))
        self.nets = []
        for run in uc["models"]:
            ck = torch.load(os.path.join(models, f"{run}_best.pt"), map_location=self.dev, weights_only=False)
            net = FrontV3(ck["cin"], ck["ns"]).to(self.dev); net.load_state_dict(ck["state"]); net.eval()
            self.nets.append((net, bool(ck.get("phys"))))
        self.unet_models = list(uc["models"])
        self.stats = json.load(open(os.path.join(models, "weather_stats.json")))
        self.k_ros = cfg["k_ros"]; self.variant = cfg["deploy"]["variant"]
        # expected - гибрид U-Net + физика (шаг 21); active - физика, откалиброванная по растущим пожарам (шаг 22, блок "sim")
        self.heads = {"expected": self._head(cfg["deploy"]), "active": self._head(cfg.get("sim", cfg["deploy"]))}
        f = cfg.get("sim", {}).get("speed_factor")              # симулятор после 72 ч: прирост «эффективного времени» за сутки
        self.tau_day = float(f["72h"] * 72 - f["48h"] * 48) if f else 24.0
        self.land = landscape or Landscape()

    @staticmethod
    def _head(d):
        return dict(W=[np.array(d["weights"][f"{h}h"]["coef"], np.float32) for h in HZ], b=[float(d["weights"][f"{h}h"]["intercept"]) for h in HZ],
                    lut=[np.array(d["lut"][f"{h}h"], np.float32) for h in HZ], level=[int(d["level"][f"{h}h"]) for h in HZ])

    # ---------- одна сцена ----------
    def _unet(self, X, valid, row, T=None):
        """Среднее вероятностей моделей ансамбля; tta - ещё и по 8 поворотам/отражениям (ветер поворачивается с картой).
        Модели с физическим каналом получают log(T/h) от физики (T - время прихода, ч)."""
        M = nn_maps(X, valid).astype(np.float32)
        si = pd.DataFrame([row])
        Z = torch.from_numpy(np.c_[base_scalars(si), weather_vec(si, self.stats, "gfsa")].astype(np.float32)).to(self.dev)
        uv = [(6 + iu, 6 + iv) for iu, iv in UV]
        tf = [(k, fl) for k in range(4) for fl in (False, True)] if self.tta else [(0, False)]
        acc = 0
        with torch.no_grad():
            for net, phys in self.nets:
                x0 = np.concatenate([M, phys_maps(T[None])], 1) if phys else M
                x0 = torch.from_numpy(x0).to(self.dev)
                for k, fl in tf:
                    x = torch.rot90(x0, k, (2, 3)); x = x.flip(3) if fl else x
                    p = torch.sigmoid(net(x, rot_wind(Z, k, fl, uv)))
                    p = p.flip(3) if fl else p
                    acc = acc + torch.rot90(p, -k, (2, 3))[0].cpu().numpy()
        p = acc / (len(self.nets) * len(tf))
        return np.round(p * 255).astype(np.uint8)

    def _physics(self, X, row):
        g = lambda k: X[0, CH.index(k)]
        fuel = sp.fuel_factor(*(np.nan_to_num(g(k)) for k in ("grass_frac", "crop_frac", "shrub_tree_frac", "wetland_frac", "nonburnable_frac")))
        elev = g("elev"); elev = np.where(np.isfinite(elev), elev, np.nanmean(elev) if np.isfinite(elev).any() else 0)
        winds = [(-float(row.get(f"fc_gfsa_wind_u_{h}h", 0) or 0), -float(row.get(f"fc_gfsa_wind_v_{h}h", 0) or 0)) for h in HZ]  # «откуда» -> «куда»
        ffmc = row.get("gfwed_FFMC", np.nan); ffmc = 88.0 if ffmc is None or not np.isfinite(ffmc) else float(ffmc)
        ndvi = np.nan_to_num(g("ndvi"), nan=0.0)                       # как в шаге 14 (кэш: пропуск NDVI -> 0); на этом подобраны веса гибрида
        return sp.arrival(g("burned_frac") > 0, fuel, ndvi, elev, winds, ffmc, k_ros=self.k_ros, lat=row.get("lat"))   # ячейка на местности по широте

    def predict_scene(self, X, valid, row, mode="expected"):
        """-> dict: prob [4,96,96] калиброванная, mask [4,96,96], unet [4,96,96], arrival_h [96,96] (физика), area_km2 [4].
        mode="expected" - гибрид U-Net + физика, ожидаемый исход (мелкие пожары часто гаснут) - для прогноза реальных пожаров;
        "active" - физика, откалиброванная на срезах, где пожар рос (≥ 1 км²): «куда пойдёт огонь, если не погаснет» - для симулятора."""
        H = self.heads[mode]
        T = self._physics(X, row)
        P = self._unet(X, valid, row, T) if any(w[0] != 0 for w in H["W"]) else np.zeros((len(HZ),) + T.shape, np.uint8)   # sim: без U-Net
        t = np.asarray(T, np.float32); far = t >= TMAX
        lt = np.log(np.clip(np.where(far, TMAX, t), 0.5, TMAX)[None] / np.array(HZ, np.float32)[:, None, None])
        p = np.clip(P.astype(np.float32) / 255, 0.5 / 255, 254.5 / 255); lp = np.log(p / (1 - p))
        prob = np.zeros(P.shape, np.float32); mask = np.zeros(P.shape, bool)
        for k, h in enumerate(HZ):
            ws = float(np.nan_to_num(row.get(f"fc_gfsa_ws_mean_{h}h", 0.0)))
            W = H["W"][k]; z = W[0] * lp[k] + W[1] * lt[k] + W[2] * far + W[3] * lt[k] * ws + H["b"][k]
            q = np.round(255 / (1 + np.exp(-z))).astype(np.uint8)
            prob[k] = H["lut"][k][q] * valid[0]; mask[k] = (q >= H["level"][k]) & valid[0]
        for k in range(1, len(HZ)):                                 # вложенность горизонтов: 12 ⊂ 24 ⊂ 48 ⊂ 72 ч
            mask[k] |= mask[k - 1]; prob[k] = np.maximum(prob[k], prob[k - 1])
        return dict(prob=prob, mask=mask, unet=P.astype(np.float32) / 255, arrival_h=np.minimum(t, TMAX),
                    area_km2=(mask.reshape(len(HZ), -1).sum(1) * (CELL / 1000) ** 2).tolist())

    # ---------- геометрия ----------
    @staticmethod
    def window_transform(r0, c0):
        return from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL)

    def geojson(self, mask, r0, c0, props=None):
        tr = self.window_transform(r0, c0); feats = []
        for geom, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr):
            feats.append(dict(type="Feature", geometry=transform_geom(CRS, "EPSG:4326", geom), properties=props or {}))
        return feats

    def bounds_lonlat(self, r0, c0, n=N):
        lon, lat = self.land.lonlat(np.array([r0 - 0.5, r0 - 0.5, r0 + n - 0.5, r0 + n - 0.5]), np.array([c0 - 0.5, c0 + n - 0.5, c0 - 0.5, c0 + n - 0.5]))
        return dict(corners_lonlat=np.c_[lon, lat].round(5).tolist())     # СЗ, СВ, ЮЗ, ЮВ (сетка EPSG:6933, не прямоугольник в lon/lat)

    # ---------- сценарии ----------
    def _weather(self, lat, lon, t, weather, hseries=None):
        if weather is None or weather == "gfs":
            row, h = wx.forecast_row(lat, lon, t, hseries)
            return row, h
        base = None
        if weather.get("use_real_history", True):
            try:
                hseries = hseries if hseries is not None else wx.fetch(lat, lon, t)
                base = wx.history_row(hseries, t, lon) or None
            except Exception:                                   # нет сети - сухое лето по умолчанию
                base = None
        return wx.scenario_row(weather["winds"], weather.get("T", 30.0), weather.get("RH", 20.0), weather.get("P", 0.0), base), hseries

    def _scene_at(self, state, t, t0, ign_g, k_12h, weather_row, hotspots_g=None, n=N):
        """state - {(r, c) глобальные: время выгорания, ч от t0}. Окно n × n по центру пожара."""
        rs = np.array([k[0] for k in state]); cs = np.array([k[1] for k in state]); ah = np.array(list(state.values()), np.float32)
        r0, c0 = self.land.window_origin(int(rs.mean()), int(cs.mean()), n)
        inside = (rs >= r0) & (rs < r0 + n) & (cs >= c0) & (cs < c0 + n)
        burned = np.zeros((n, n), bool); age = np.zeros((n, n), np.float32)
        burned[rs[inside] - r0, cs[inside] - c0] = True
        age[rs[inside] - r0, cs[inside] - c0] = (t - t0).total_seconds() / 3600 - ah[inside]
        static, cov, flags = self.land.static(r0, c0, t, n)
        hs = None if hotspots_g is None else [(r - r0, c - c0, dt, frp) for r, c, dt, frp in hotspots_g]
        X, valid, row = sc.build(static, cov, burned, age, t, (ign_g[0] - r0, ign_g[1] - c0), k_12h, hs, weather_row)
        row["lon"], row["lat"] = (float(v) for v in self.land.lonlat(r0 + n // 2, c0 + n // 2))
        return X, valid, row, r0, c0, flags

    def predict(self, burned_lonlat=None, lon=None, lat=None, t_utc=None, radius_km=1.0, hotspots=None, weather="gfs", age_h=6.0, mode="expected"):
        """Прогноз от состояния пожара: точки гари (lon, lat) или круг radius_km вокруг (lon, lat); hotspots - [(lon, lat, часов до t, frp)]."""
        t = pd.Timestamp(t_utc or pd.Timestamp.utcnow().tz_localize(None)).floor("h")
        if burned_lonlat:
            rc = [self.land.rc(a, b) for a, b in burned_lonlat]
            lon, lat = np.mean([p[0] for p in burned_lonlat]), np.mean([p[1] for p in burned_lonlat])
        else:
            r, c = self.land.rc(lon, lat); rad = max(radius_km / (CELL / 1000), 0.5)
            rr, cc = np.mgrid[-int(rad) - 1:int(rad) + 2, -int(rad) - 1:int(rad) + 2]; m = rr ** 2 + cc ** 2 <= rad ** 2
            rc = list(zip((r + rr[m]).tolist(), (c + cc[m]).tolist()))
        state = {k: -age_h for k in rc}
        hs = None if hotspots is None else [(*self.land.rc(a, b), dt, frp) for a, b, dt, frp in hotspots]
        row_w, _ = self._weather(lat, lon, t, weather)
        ign = (int(np.mean([k[0] for k in rc])), int(np.mean([k[1] for k in rc])))
        X, valid, row, r0, c0, flags = self._scene_at(state, t, t, ign, 0, row_w, hs)
        out = self.predict_scene(X, valid, row, mode)
        return self._pack(out, r0, c0, t, row, flags, valid)

    def simulate(self, lon, lat, t_utc=None, hours=72, radius_km=0.5, weather="gfs", mode="active"):
        """Сценарий от точки возгорания. ≤ 72 ч - один прогон (горизонты 12/24/48/72); больше - прокрутка шагами 24 ч.
        Кадры каждые 12 ч: маска гари, вероятность, площадь, контур."""
        t0 = pd.Timestamp(t_utc or pd.Timestamp.utcnow().tz_localize(None)).floor("h")
        r, c = self.land.rc(lon, lat); rad = max(radius_km / (CELL / 1000), 0.5)
        rr, cc = np.mgrid[-int(rad) - 1:int(rad) + 2, -int(rad) - 1:int(rad) + 2]; m = rr ** 2 + cc ** 2 <= rad ** 2
        state = {(r + a, c + b): 0.0 for a, b in zip(rr[m].tolist(), cc[m].tolist())}
        frames = [dict(hour=0, area_km2=len(state) * (CELL / 1000) ** 2, geojson=self._state_geojson(state))]
        hseries = None; t = t0; k12 = 0; first = None
        while (t - t0).total_seconds() / 3600 < hours:
            row_w, hseries = self._weather(lat, lon, t, weather, hseries)
            n = SIM_N if mode == "active" else N
            X, valid, row, r0, c0, flags = self._scene_at(state, t, t0, (r, c), k12, row_w, n=n)
            out = self.predict_scene(X, valid, row, mode)
            dt0 = (t - t0).total_seconds() / 3600
            if first is None:
                first = self._pack(out, r0, c0, t, row, flags, valid, n)
            if mode == "active" and dt0 > 0:                      # симулятор после первых 72 ч: сутки = tau_day ч распространения по физике
                T = out["arrival_h"]; burned0 = X[0, CH.index("burned_frac")].astype(bool)
                for h, frac in ((12, 0.5), (24, 1.0)):
                    if dt0 + h > hours:
                        break
                    new = (T > 0) & (T <= self.tau_day * frac) & valid[0] & ~burned0
                    for i, j in zip(*np.nonzero(new)):
                        state.setdefault((r0 + i, c0 + j), dt0 + h * float(T[i, j]) / (self.tau_day * frac))
                    edge = bool(new[0].any() or new[-1].any() or new[:, 0].any() or new[:, -1].any())
                    frames.append(dict(hour=dt0 + h, area_km2=len(state) * (CELL / 1000) ** 2, geojson=self._state_geojson(state), edge_clipped=edge))
                t = t + pd.Timedelta(hours=24); k12 += 2
                continue
            # первый прогон симулятора - сразу все горизонты ≤ 72 ч; гибрид (expected): остаток ≤ 72 ч - прямой прогноз, иначе шаг 24 ч
            last = hours - dt0 <= 72
            steps = [(k, h) for k, h in enumerate(HZ) if dt0 + h <= hours] if (last or mode == "active") else [(0, 12), (1, 24)]
            for k, h in steps:
                new = out["mask"][k] & ~X[0, CH.index("burned_frac")].astype(bool)
                for i, j in zip(*np.nonzero(new)):
                    state.setdefault((r0 + i, c0 + j), dt0 + float(np.clip(out["arrival_h"][i, j], 0, h)))
                mk = out["mask"][k]; edge = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
                frames.append(dict(hour=dt0 + h, area_km2=len(state) * (CELL / 1000) ** 2, geojson=self._state_geojson(state),
                                   prob_max=float(out["prob"][k].max()), edge_clipped=edge))
            if last:
                break
            dt = 72 if mode == "active" else 24
            t = t + pd.Timedelta(hours=dt); k12 += dt // 12
        return dict(start=dict(lon=lon, lat=lat, t_utc=str(t0)), first_pass=first, frames=frames,
                    arrival_h={f"{a},{b}": round(v, 1) for (a, b), v in state.items()})

    def _state_geojson(self, state):
        rs = np.array([k[0] for k in state]); cs = np.array([k[1] for k in state])
        r0, c0 = rs.min(), cs.min(); m = np.zeros((rs.max() - r0 + 1, cs.max() - c0 + 1), bool); m[rs - r0, cs - c0] = True
        return self.geojson(m, r0, c0)

    def _pack(self, out, r0, c0, t, row, flags, valid, n=N):
        contours = {f"{h}h": self.geojson(out["mask"][k], r0, c0, dict(horizon_h=h)) for k, h in enumerate(HZ)}
        return dict(t_utc=str(t), window=dict(r0=r0, c0=c0, cell_m=CELL, crs=CRS, size=n, **self.bounds_lonlat(r0, c0, n)), flags=flags,
                    area_km2=dict(zip([f"{h}h" for h in HZ], out["area_km2"])), contours=contours,
                    prob=out["prob"], mask=out["mask"], arrival_h=out["arrival_h"], valid=valid[0],
                    weather={k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v)
                             for k, v in row.items() if k.startswith(("fc_gfsa_", "gfwed_", "era5_", "weather_src"))})
