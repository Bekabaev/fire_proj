"""Шаг 12 (расширение выборки). Статические и сезонные слои на национальной сетке 500 м EPSG:6933.

- ESA WorldCover 2021 (обзорный уровень 80 м) -> доли классов в ячейке 500 м: трава, пашня, кустарник/лес, водно-болотные, негорючее;
  доминирующий класс -> FBFM40 (то же сопоставление, что в шаге 8)            -> data/ext_kz/lc.npz
- Copernicus DEM GLO-90 (Planetary Computer, прорежённое чтение)        -> высота и уклон            -> data/ext_kz/dem.npz
- MODIS MOD13A1 NDVI 16 сут (Planetary Computer, обзор 1 км), март–ноябрь каждого года  -> data/ext_kz/ndvi_<год>.npz
  (ndvi [n, GH/2, GW/2] int16 ×1e-4, doy - день начала композита; для пожара берётся последний композит до начала пожара)
"""
import os, sys, time
import numpy as np, requests, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import Affine, from_origin
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
LON0, LAT0, LON1, LAT1 = transform_bounds(CRS, "EPSG:4326", X0, Y1 - GH * CELL, X0 + GW * CELL, Y1, densify_pts=50)
print("grid lon/lat bounds:", round(LON0, 2), round(LAT0, 2), round(LON1, 2), round(LAT1, 2), flush=True)
FBFM40 = {10: 161, 20: 121, 30: 102, 40: 101, 50: 91, 60: 99, 70: 92, 80: 98, 90: 104, 95: 98, 100: 99}
GROUPS = {"grass": [30], "crop": [40], "shrub_tree": [10, 20], "wetland": [90], "nonburnable": [50, 60, 70, 80, 95, 100]}


def sub_grid(bounds_ll):
    """Окно национальной сетки, покрывающее bbox (lon/lat): (r0, r1, c0, c1, transform)."""
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", CRS, *bounds_ll, densify_pts=21)
    c0 = max(int((x0 - X0) // CELL) - 1, 0); c1 = min(int((x1 - X0) // CELL) + 2, GW)
    r0 = max(int((Y1 - y1) // CELL) - 1, 0); r1 = min(int((Y1 - y0) // CELL) + 2, GH)
    return r0, r1, c0, c1, from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL)


def pc_token(acc, cont):
    return requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{acc}/{cont}", timeout=60).json()["token"]


def retry(fn, *a, n=4):
    for k in range(n):
        try:
            return fn(*a)
        except Exception as e:
            if k == n - 1:
                print("  failed:", str(e)[:120], flush=True); return None
            time.sleep(5 * (k + 1))


# ---------- WorldCover
fl = os.path.join(KZ, "lc.npz")
if not os.path.exists(fl):
    acc = {k: np.zeros((GH, GW), np.float32) for k in GROUPS}; cov = np.zeros((GH, GW), np.float32)
    tiles = [(la, lo) for la in range(int(np.floor(LAT0 / 3) * 3), int(LAT1) + 1, 3) for lo in range(int(np.floor(LON0 / 3) * 3), int(LON1) + 1, 3)]

    def wc_read(t):
        la, lo = t
        url = f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N{la:02d}E{lo:03d}_Map.tif"
        try:
            with rasterio.open(url) as s:
                a = s.read(1, out_shape=(s.height // 8, s.width // 8), resampling=Resampling.nearest)
                tr = s.transform * Affine.scale(s.width / a.shape[1], s.height / a.shape[0])
                return t, a, tr
        except rasterio.errors.RasterioIOError:
            return t, None, None
    with ThreadPoolExecutor(6) as ex:
        for (la, lo), a, tr in ex.map(wc_read, tiles):
            if a is None:
                continue
            r0, r1, c0, c1, str_ = sub_grid((lo, la, lo + 3, la + 3))
            for k, cls in {**GROUPS, "_valid": None}.items():
                src = (a > 0).astype(np.float32) if cls is None else np.isin(a, cls).astype(np.float32)
                src[a == 0] = np.nan
                dst = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                reproject(src, dst, src_transform=tr, src_crs="EPSG:4326", dst_transform=str_, dst_crs=CRS,
                          resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
                ok = np.isfinite(dst)
                if cls is None:
                    cov[r0:r1, c0:c1][ok] = 1
                else:
                    acc[k][r0:r1, c0:c1][ok] = dst[ok]
            print("  worldcover", la, lo, flush=True)
    fr = np.stack([acc[k] for k in GROUPS])                         # [5, GH, GW]
    dom_group = np.argmax(fr, 0)
    fb = np.array([102, 101, 121, 104, 99], np.int16)[dom_group]   # трава GR2, пашня GR1, кустарник GS1, тростник GR4, негорючее NB9
    fb[cov == 0] = 0
    np.savez_compressed(fl, frac=(fr * 100).round().astype(np.uint8), groups=list(GROUPS), fbfm40=fb, covered=cov.astype(bool))
    print("worldcover done, covered cells:", int(cov.sum()), flush=True)

# ---------- DEM
fd = os.path.join(KZ, "dem.npz")
if not os.path.exists(fd):
    tok = pc_token("elevationeuwest", "copernicus-dem")
    elev = np.full((GH, GW), np.nan, np.float32)
    items = []
    for la0 in range(int(np.floor(LAT0)), int(np.ceil(LAT1)), 5):
        j = requests.post(STAC, json={"collections": ["cop-dem-glo-90"], "bbox": [LON0, la0, LON1, min(la0 + 5, LAT1)], "limit": 1000}, timeout=120).json()
        items += [f["assets"]["data"]["href"] for f in j["features"]]
    items = sorted(set(items)); print("DEM tiles:", len(items), flush=True)

    def dem_read(h):
        def f():
            with rasterio.open("/vsicurl/" + h + "?" + tok) as s:
                a = s.read(1, out_shape=(max(s.height // 4, 1), max(s.width // 4, 1)), resampling=Resampling.average).astype(np.float32)
                return a, s.transform * Affine.scale(s.width / a.shape[1], s.height / a.shape[0]), s.bounds
        return retry(f)
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(dem_read, items)):
            if res is None:
                continue
            a, tr, b = res
            r0, r1, c0, c1, str_ = sub_grid((b.left, b.bottom, b.right, b.top))
            if r1 <= r0 or c1 <= c0:
                continue
            dst = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
            reproject(a, dst, src_transform=tr, src_crs="EPSG:4326", dst_transform=str_, dst_crs=CRS,
                      resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            ok = np.isfinite(dst); elev[r0:r1, c0:c1][ok] = dst[ok]
            if i % 100 == 0:
                print("  dem", i, flush=True)
    gy, gx = np.gradient(elev, CELL)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    np.savez_compressed(fd, elev=elev.astype(np.float16), slope=slope.astype(np.float16))
    print("dem done, finite:", round(float(np.isfinite(elev).mean()), 3), flush=True)

# ---------- NDVI (MOD13A1, Terra, 16 сут)
G2 = (GH // 2, GW // 2); TR2 = from_origin(X0, Y1, 2 * CELL, 2 * CELL)
YRS = [int(v) for v in os.environ["ONLY_YEARS"].split(",")] if os.environ.get("ONLY_YEARS") else YEARS   # ONLY_YEARS=2018,2019 - доп. процесс
for y in (YRS[::-1] if os.environ.get("REVERSE") else YRS):   # REVERSE=1 - второй параллельный процесс с конца
    fn = os.path.join(KZ, f"ndvi_{y}.npz")
    if os.path.exists(fn):
        continue
    tok = pc_token("modiseuwest", "modis-061-cogs")      # токен SAS живёт ограниченное время - берём заново на каждый год
    j = requests.post(STAC, json={"collections": ["modis-13A1-061"], "bbox": [LON0, LAT0, LON1, LAT1], "limit": 1000,
                                  "datetime": f"{y}-03-01T00:00:00Z/{y}-11-15T23:59:59Z"}, timeout=180).json()
    feats = [f for f in j["features"] if f["id"].startswith("MOD13A1")]
    doys = sorted({int(f["id"].split(".")[1][5:8]) for f in feats})
    print(y, "NDVI composites:", len(doys), "items:", len(feats), flush=True)

    def nd_read(f):
        def g():
            with rasterio.open("/vsicurl/" + f["assets"]["500m_16_days_NDVI"]["href"] + "?" + tok) as s:
                a = s.read(1, out_shape=(1200, 1200), resampling=Resampling.nearest).astype(np.float32)
                a[a < -2000] = np.nan
                return int(f["id"].split(".")[1][5:8]), a, s.transform * Affine.scale(2, 2), s.crs
        return retry(g)
    stack = np.full((len(doys),) + G2, np.nan, np.float32)
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(nd_read, feats):
            if res is None:
                continue
            d, a, tr, crs = res
            dst = np.full(G2, np.nan, np.float32)
            reproject(a, dst, src_transform=tr, src_crs=crs, dst_transform=TR2, dst_crs=CRS,
                      resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            k = doys.index(d); ok = np.isfinite(dst); stack[k][ok] = dst[ok]
    np.savez_compressed(fn, ndvi=np.nan_to_num(stack, nan=-32768).astype(np.int16), doy=np.array(doys))
    print(y, "saved, finite share:", round(float(np.isfinite(stack).mean()), 3), flush=True)
print("done")
