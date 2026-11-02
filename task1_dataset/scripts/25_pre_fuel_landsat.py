"""Шаг 25 (датасет v3). Топливо до пожара для кубов заповедников без композита GEE (23 из 44 событий): Landsat 8/9 C2 L2.

Источник: Microsoft Planetary Computer, коллекция landsat-c2-l2 (без ключа; SAS-токен). Окно: 60 сут до возгорания,
облачность сцены ≤ 70 %. Маска QA_PIXEL: заполнение, расширенное облако, перистые, облако, тень, снег (биты 0–5).
Медиана безоблачных значений по сценам -> те же каналы, что композит GEE (Sentinel-2): blue→B2, green→B3, red→B4,
nir08→B8A, swir16→B11, swir22→B12; отражение = DN·0,0000275 − 0,2. Заполняются только пиксели, где канала нет (NaN).
Из каналов считаются pre_NDVI, pre_NDMI, pre_NBR; dNBR пересчитывается, где есть post_NBR.
В <event>.json добавляется pre_src ("gee" / "landsat_pc" / "gee+landsat_pc"), cov_pre обновляется; cubes_index.csv - тоже.
"""
import os, sys, json
import numpy as np, pandas as pd, requests, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import Affine
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60"); os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3"); os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "NO"); os.environ.setdefault("GDAL_HTTP_VERSION", "1.1")   # HTTP/2 на Windows подвисал
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
POOL = ThreadPoolExecutor(4)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
CH = ["pre_B2", "pre_B3", "pre_B4", "pre_B8A", "pre_B11", "pre_B12", "pre_NDVI", "pre_NDMI", "pre_NBR",
      "elev", "slope_deg", "aspect_sin", "aspect_cos", "dist_ign_km", "prior_burn_days", "post_NBR", "dNBR"]
tok = None


def token():
    global tok
    tok = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2", timeout=60).json()["token"]


def read(href, shp, tr, crs, resampling, dtype=np.float32):
    out = np.full(shp, np.nan if dtype == np.float32 else 1, dtype)
    with rasterio.open("/vsicurl/" + href + "?" + tok) as s:
        b = transform_bounds(crs, s.crs, tr.c, tr.f + shp[0] * tr.e, tr.c + shp[1] * tr.a, tr.f, densify_pts=21)
        w = rasterio.windows.from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        a = s.read(1, window=w)
        reproject(a, out, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=tr, dst_crs=crs,
                  resampling=resampling, src_nodata=0, dst_nodata=np.nan if dtype == np.float32 else 1)
    return out


ev = load_events(); token(); upd = {}
for e in ev.itertuples():
    fj = os.path.join(DATA, "cubes", e.event + ".json"); fn = fj.replace(".json", ".npz")
    meta = json.load(open(fj))
    if meta.get("cov_pre", 0) >= 0.8 or meta.get("pre_src", "").endswith("landsat_pc"):
        continue
    tr = Affine(*meta["transform"]); shp = (meta["H"], meta["W"]); crs = meta["crs"]
    bb = transform_bounds(crs, "EPSG:4326", tr.c, tr.f + shp[0] * tr.e, tr.c + shp[1] * tr.a, tr.f, densify_pts=21)
    t1 = e.ign_time_utc.normalize() - pd.Timedelta(days=1); t0 = t1 - pd.Timedelta(days=60)
    j = requests.post(STAC, json={"collections": ["landsat-c2-l2"], "bbox": list(bb), "limit": 100,
                                  "datetime": f"{t0.date()}/{t1.date()}",
                                  "query": {"eo:cloud_cover": {"lte": 70}, "platform": {"in": ["landsat-8", "landsat-9"]}}}, timeout=120).json()
    items = j.get("features", [])
    stack = []
    print(f"{e.event}: {len(items)} Landsat items", flush=True)
    for it in items[:12]:
        try:
            # каналы сцены читаются параллельно; сцена, не прочитанная за 180 с, пропускается
            fq = POOL.submit(read, it["assets"]["qa_pixel"]["href"], shp, tr, crs, Resampling.nearest, np.uint16)
            fb = [POOL.submit(read, it["assets"][b]["href"], shp, tr, crs, Resampling.bilinear) for b in BANDS]
            qa = fq.result(timeout=180); bad = (qa & 0b111111) > 0
            refl = []
            for f in fb:
                dn = f.result(timeout=180)
                r_ = dn * 0.0000275 - 0.2; r_[bad | (r_ <= 0) | (r_ > 1.5)] = np.nan; refl.append(r_)
            refl = np.stack(refl)
            if np.isfinite(refl[3]).mean() > 0.05:
                stack.append(refl)
            print("   ", it["id"], "clear share", round(float(np.isfinite(refl[3]).mean()), 2), flush=True)
        except FTimeout:
            print("  item timeout", it["id"], flush=True); POOL.shutdown(wait=False, cancel_futures=True); POOL = ThreadPoolExecutor(4); continue
        except Exception as ex:
            print("  item failed", it["id"], str(ex)[:100], flush=True); token()
    if not stack:
        print(f"{e.event}: no clear Landsat scenes in {t0.date()}..{t1.date()} ({len(items)} items)", flush=True); continue
    with np.errstate(all="ignore"):
        comp = np.nanmedian(np.stack(stack), 0)
    d = dict(np.load(fn)); X = d["x"].astype(np.float32)
    hole = ~np.isfinite(X[CH.index("pre_NBR")])
    X[0:6] = np.where(hole[None], comp, X[0:6])
    nd = lambda a, c: (a - c) / (a + c + 1e-9)
    for k, (a, c) in {"pre_NDVI": (3, 2), "pre_NDMI": (3, 4), "pre_NBR": (3, 5)}.items():
        X[CH.index(k)] = np.where(hole, nd(X[a], X[c]), X[CH.index(k)])
    X[CH.index("dNBR")] = X[CH.index("pre_NBR")] - X[CH.index("post_NBR")]
    d["x"] = X.astype(np.float16)
    np.savez_compressed(fn + ".part.npz", **d); os.replace(fn + ".part.npz", fn)
    old = meta.get("cov_pre", 0)
    meta["cov_pre"] = round(float(np.isfinite(X[CH.index("pre_NBR")]).mean()), 3)
    meta["pre_src"] = ("gee+" if old > 0 else "") + "landsat_pc"; meta["pre_landsat_scenes"] = len(stack)
    json.dump(meta, open(fj, "w"), ensure_ascii=False); upd[e.event] = (meta["cov_pre"], meta["pre_src"])
    print(f"{e.event}: scenes {len(stack)}/{len(items)}, cov_pre {old} -> {meta['cov_pre']}", flush=True)

ci = os.path.join(DATA, "cubes_index.csv")
c = pd.read_csv(ci)
if "pre_src" not in c:
    c["pre_src"] = np.where(c.cov_pre > 0, "gee", "")
for k in c.event:
    m = json.load(open(os.path.join(DATA, "cubes", k + ".json")))
    if "pre_src" in m:
        c.loc[c.event == k, ["cov_pre", "pre_src"]] = [m["cov_pre"], m["pre_src"]]
c.to_csv(ci, index=False)
print("cov_pre >= 0.5:", int((c.cov_pre >= 0.5).sum()), "of", len(c))
