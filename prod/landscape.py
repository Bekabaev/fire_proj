"""Национальные слои 500 м (EPSG:6933) -> статические каналы сцены 96 × 96 в любой точке Казахстана.

Правила - как в task1 шаг 22 (сборка обучающих срезов):
- NDVI MOD13A1 - последний 16-дневный композит, закончившийся до t (doy композита + 16 ≤ doy t), аномалия - к климатологии того же doy;
  если композит старше 24 сут (пробел данных) - норма на дату t + аномалия последнего композита (флаг ndvi_stale_days);
  если NDVI за год t нет (оперативный режим после 2024) - климатология того doy, аномалия 0 (флаг ndvi_from_clim);
- прошлые гари: 1 - горело в год t−1 или в этом году до t, 0,5 - в год t−2 (GFA до 2024, MCD64A1 2025–26 - task1 шаг 36);
  «лет с прошлой гари» - по last_burn_<год>; для лет без данных - последние доступные (флаги burn_history_year, burns_known_until);
- рельеф GLO-90, доли WorldCover, расстояния OSM - статичны.
"""
import os
import numpy as np
from rasterio.warp import transform as warp_transform

from .config import KZ, CELL, N, X0, Y1, GW, GH, CRS


def _pad(a, val=-32768):
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, 2), (0, 2)], constant_values=val)


def _up2(a, r0, c0, n):
    """Поле сетки 1 км (int16 × 1e-4) -> окно n × n сетки 500 м."""
    s = a[r0 // 2:(r0 + n) // 2 + 1, c0 // 2:(c0 + n) // 2 + 1].astype(np.float32)
    s = np.repeat(np.repeat(s, 2, 0), 2, 1)[(r0 % 2):(r0 % 2) + n, (c0 % 2):(c0 % 2) + n]
    return np.where(s < -20000, np.nan, s * 1e-4)


class Landscape:
    def __init__(self, kz=KZ):
        self.kz = kz
        lc = np.load(os.path.join(kz, "lc.npz")); self.FR = lc["frac"]; self.FB = lc["fbfm40"]; self.COV = lc["covered"]
        dem = np.load(os.path.join(kz, "dem.npz")); self.ELEV = dem["elev"]; self.SLOPE = dem["slope"]
        osm = np.load(os.path.join(kz, "osm.npz")); self.DSET = osm["dist_settlement_km"]; self.DROAD = osm["dist_road_km"]; self.RDEN = osm["road_density"]
        cl = np.load(os.path.join(kz, "ndvi_clim.npz")); self.CLIM = _pad(cl["ndvi"]); self.CLIM_DOY = np.asarray(cl["doy"])
        self.years = sorted(int(f[4:8]) for f in os.listdir(kz) if f.startswith("doy_") and f.endswith(".npy"))
        self._cache = {}

    # --- координаты ---
    @staticmethod
    def rc(lon, lat):
        x, y = warp_transform("EPSG:4326", CRS, [float(lon)], [float(lat)])
        return int((Y1 - y[0]) // CELL), int((x[0] - X0) // CELL)

    @staticmethod
    def lonlat(r, c):
        """Центры ячеек (r, c) -> (lon, lat), массивы."""
        r = np.asarray(r, float); c = np.asarray(c, float)
        lon, lat = warp_transform(CRS, "EPSG:4326", (X0 + (c.ravel() + 0.5) * CELL).tolist(), (Y1 - (r.ravel() + 0.5) * CELL).tolist())
        return np.array(lon).reshape(r.shape), np.array(lat).reshape(r.shape)

    @staticmethod
    def window_origin(r, c, n=N):
        return int(np.clip(r - n // 2, 0, GH - n)), int(np.clip(c - n // 2, 0, GW - n))

    # --- погодичные слои (лениво, mmap) ---
    def _year(self, kind, y):
        k = (kind, y)
        if k not in self._cache:
            f = os.path.join(self.kz, {"doy": f"doy_{y}.npy", "last": f"last_burn_{y}.npy", "peak": f"ndvi_peak_{y}.npy",
                                       "ndvi": f"ndvi_{y}.npz"}[kind])
            if not os.path.exists(f):
                self._cache[k] = None
            elif kind == "ndvi":
                d = np.load(f); self._cache[k] = (_pad(d["ndvi"]), np.asarray(d["doy"]))
            elif kind == "peak":
                self._cache[k] = _pad(np.load(f))
            else:
                self._cache[k] = np.load(f, mmap_mode="r")
        return self._cache[k]

    def static(self, r0, c0, t_utc, n=N):
        """Статические и сезонные каналы окна. t_utc - pandas.Timestamp. -> (dict каналов, covered, флаги)."""
        y, doy = t_utc.year, t_utc.dayofyear
        w = (slice(r0, r0 + n), slice(c0, c0 + n)); flags = {}
        nd = self._year("ndvi", y)
        if nd is not None and (kn := np.nonzero(nd[1] + 16 <= doy)[0]).size:
            for j in kn[::-1]:                                  # последний композит, заполненный в окне (свежие композиты бывают частичными)
                nv = _up2(nd[0][j], r0, c0, n)
                if np.isfinite(nv).mean() >= 0.5:
                    break
            ci = int(np.argmin(np.abs(self.CLIM_DOY - nd[1][j]))); an = nv - _up2(self.CLIM[ci], r0, c0, n)
            flags["ndvi_from_clim"] = False
            stale = int(doy - (nd[1][j] + 16))
            if stale > 24:                                      # пробел в данных (MODIS 07–10.2025 нет в каталоге): норма на дату t + аномалия последнего композита
                ok = np.nonzero(self.CLIM_DOY + 16 <= doy)[0]; ct = int(ok[-1]) if ok.size else 0
                nv = _up2(self.CLIM[ct], r0, c0, n) + np.nan_to_num(an); flags["ndvi_stale_days"] = stale
        else:
            ok = np.nonzero(self.CLIM_DOY + 16 <= doy)[0]; ci = int(ok[-1]) if ok.size else 0
            nv = _up2(self.CLIM[ci], r0, c0, n); an = np.zeros_like(nv); flags["ndvi_from_clim"] = True
        yb = min(y, self.years[-1]); flags["burn_history_year"] = yb
        pk = self._year("peak", yb - 1)
        pk = _up2(pk, r0, c0, n) if pk is not None else np.full((n, n), np.nan, np.float32)
        p1 = self._year("doy", yb - 1); p2 = self._year("doy", yb - 2)
        prior = np.zeros((n, n), np.float32)
        if p2 is not None:
            prior[p2[w] > 0] = 0.5
        if p1 is not None:
            prior[p1[w] > 0] = 1.0
        p0 = self._year("doy", y) if y == yb else None          # гари этого года до t (в обучении - другие пожары года, выгоревшие ≤ t)
        if p0 is not None:
            d0 = np.asarray(p0[w]); prior[(d0 > 0) & (d0 < doy)] = 1.0
            fm = os.path.join(self.kz, f"mcd64_months_{y}.json")
            if os.path.exists(fm):
                import json
                meta = json.load(open(fm)); flags["burns_known_until"] = meta["months"][-1]
                if meta.get("filled_from_firms"):
                    flags["burns_filled_from_firms"] = meta["filled_from_firms"]     # месяцы без MCD64A1 (task1 шаг 37, IoU с GFA 0,17)
        lb = self._year("last", yb)
        ysb = np.where(lb[w] > 0, y - lb[w], 13).astype(np.float32) if lb is not None else np.full((n, n), 13.0, np.float32)
        S = dict(prior_burn=prior.astype(np.float32), years_since_burn=np.minimum(ysb, 13), fbfm40=self.FB[w].astype(np.float32),
                 grass_frac=self.FR[0][w] / 100, crop_frac=self.FR[1][w] / 100, shrub_tree_frac=self.FR[2][w] / 100,
                 wetland_frac=self.FR[3][w] / 100, nonburnable_frac=self.FR[4][w] / 100,
                 elev=self.ELEV[w].astype(np.float32), slope_deg=self.SLOPE[w].astype(np.float32), ndvi=nv, ndvi_anom=an, ndvi_peak_prev=pk,
                 dist_settlement_km=self.DSET[w].astype(np.float32), dist_road_km=self.DROAD[w].astype(np.float32),
                 road_density=self.RDEN[w].astype(np.float32))
        return S, np.asarray(self.COV[w]), flags
