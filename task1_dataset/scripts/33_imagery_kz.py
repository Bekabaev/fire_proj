"""Шаг 33 (датасет v4). Состояние гари по снимкам Landsat 8/9 C2 L2 и Sentinel-2 L2A для пожаров GFA Казахстана.

Готовый продукт USGS Landsat Burned Area покрывает только США, поэтому гарь распознаётся здесь так же, как в шаге 30
(проверено на экспертных пожарах): NBR = (NIR − SWIR2) / (NIR + SWIR2), dNBR = NBR до пожара (медиана ≤ 2 безоблачных снимков
за 30 сут до возгорания) − NBR снимка; фон - пиксели окна вне ячеек пожаров этого года (с буфером 3 ячейки);
выгорело, если dNBR − медиана фона > max(0,05, p97 фона − медиана фона), затем фильтр большинства 5 × 5.
Снимки - Planetary Computer (без ключа), читаются в сетке 62,5 м, выровненной с национальной сеткой 500 м (8 × 8 пикселей в ячейке).
Для каждой ячейки 500 м и снимка: доля безоблачного (cf) и доля гари среди безоблачного (bf);
состояние 1 = cf ≥ 0,5 и bf ≥ 0,5, 0 = cf ≥ 0,5 и bf < 0,15, иначе −1 (нет данных / неясно).
Надёжность снимка: доля распознанных ячеек, заведомо сгоревших по VIIRS ≥ 24 ч до снимка (≥ 20 ячеек), ≥ 0,7.
Порядок: пожары test, val, затем train по убыванию площади; только площадь ≥ MIN_KM2 (10 км²), годы ≥ 2013.
Выход: data/ext_kz/imagery/<fire_id>.npz (r0, c0, times_h от 1 янв., sensors, state int8 [K,h,w], bf u8 %, cf u8 %,
       dn i16 = средний dNBR над фоном × 1000 (для подбора порога в шаге 34), recall, reliable),
       data/ext_kz/imagery/_log.csv. Повторный запуск продолжает с места остановки.
"""
import os, sys, time, threading
import numpy as np, pandas as pd, requests, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter, binary_dilation
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from kz_common import KZ, CRS, CELL, X0, Y1, GH, GW

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60"); os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3"); os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "NO"); os.environ.setdefault("GDAL_HTTP_VERSION", "1.1")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
OD = os.path.join(KZ, "imagery"); os.makedirs(OD, exist_ok=True)
SUB = 8; PX = CELL / SUB; PAD = 4                                   # 62,5 м; запас окна 4 ячейки
MIN_KM2 = float(os.environ.get("MIN_KM2", 10)); NW = int(os.environ.get("WORKERS", 6))
COLL = ("sentinel-2-l2a", "landsat-c2-l2")
TOK = {}; TLOCK = threading.Lock()


def token(coll, force=False):
    with TLOCK:
        if force or coll not in TOK or time.time() - TOK[coll][1] > 1800:
            TOK[coll] = (requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{coll}", timeout=60).json()["token"], time.time())
        return TOK[coll][0]


def read(href, coll, shp, tr, rs, nod):
    out = np.full(shp, np.nan, np.float32)
    with rasterio.open("/vsicurl/" + href + "?" + token(coll)) as s:
        b = transform_bounds(CRS, s.crs, tr.c, tr.f + shp[0] * tr.e, tr.c + shp[1] * tr.a, tr.f, densify_pts=21)
        w = rasterio.windows.from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        a = s.read(1, window=w).astype(np.float32)
        reproject(a, out, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=tr, dst_crs=CRS, resampling=rs,
                  src_nodata=nod, dst_nodata=np.nan)
    return out


def search(coll, bb, t0, t1):
    q = {"collections": [coll], "bbox": list(bb), "limit": 200, "datetime": f"{t0:%Y-%m-%dT%H:%M:%SZ}/{t1:%Y-%m-%dT%H:%M:%SZ}",
         "query": {"eo:cloud_cover": {"lte": 70}}}
    if coll == "landsat-c2-l2":
        q["query"]["platform"] = {"in": ["landsat-8", "landsat-9"]}
    for k in range(3):
        try:
            return requests.post(STAC, json=q, timeout=120).json().get("features", [])
        except Exception:
            time.sleep(10 * (k + 1))
    return []


def nbr_scene(it, coll, shp, tr):
    if coll == "sentinel-2-l2a":
        off = 1000.0 if float(it["properties"].get("s2:processing_baseline", "0")) >= 4.0 else 0.0
        nir, sw = [(read(it["assets"][b]["href"], coll, shp, tr, Resampling.average, 0) - off) / 1e4 for b in ("B8A", "B12")]
        scl = read(it["assets"]["SCL"]["href"], coll, shp, tr, Resampling.nearest, 0)
        bad = ~np.isfinite(scl) | np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    else:
        nir, sw = [read(it["assets"][b]["href"], coll, shp, tr, Resampling.average, 0) * 0.0000275 - 0.2 for b in ("nir08", "swir22")]
        qa = read(it["assets"]["qa_pixel"]["href"], coll, shp, tr, Resampling.nearest, 1)
        bad = ~np.isfinite(qa) | ((np.nan_to_num(qa, nan=1).astype(np.uint16) & 0b1111) > 0)
    bad |= ~np.isfinite(nir) | ~np.isfinite(sw) | (nir <= 0) | (sw <= 0) | (nir > 1.2) | (sw > 1.2)
    return np.where(bad, np.nan, (nir - sw) / (nir + sw + 1e-9)).astype(np.float32)


def cells(a, f):
    """[h*SUB, w*SUB] -> [h, w] средним."""
    h, w = a.shape[0] // SUB, a.shape[1] // SUB
    return f(a.reshape(h, SUB, w, SUB), axis=(1, 3))


def process(e, AR, FID, SRC):
    fo = os.path.join(OD, f"{e.fire_id}.npz")
    if os.path.exists(fo):
        return None
    mine = FID == e.fire_id
    rr, cc = np.nonzero(mine[e.rmin - PAD:e.rmax + PAD + 1, e.cmin - PAD:e.cmax + PAD + 1])
    r0 = max(e.rmin - PAD, 0); c0 = max(e.cmin - PAD, 0); r1 = min(e.rmax + PAD + 1, GH); c1 = min(e.cmax + PAD + 1, GW)
    w = (slice(r0, r1), slice(c0, c1)); h, wd = r1 - r0, c1 - c0
    A = AR[w]; M = mine[w]; S = SRC[w]
    if M.sum() < 4:
        return dict(fire_id=e.fire_id, status="no cells")
    tr = from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, PX, PX); shp = (h * SUB, wd * SUB)
    bb = transform_bounds(CRS, "EPSG:4326", tr.c, tr.f - shp[0] * PX, tr.c + shp[1] * PX, tr.f, densify_pts=21)
    jan1 = pd.Timestamp(year=int(e.year), month=1, day=1)
    t0, t1 = np.nanmin(A[M]), np.nanmax(A[M])
    T0 = jan1 + pd.Timedelta(hours=float(t0)); T1 = jan1 + pd.Timedelta(hours=float(t1))
    items = []
    for coll in COLL:
        items += [(coll, it) for it in search(coll, bb, T0 - pd.Timedelta(days=30), T1 + pd.Timedelta(days=5))]
    items.sort(key=lambda x: x[1]["properties"]["datetime"])
    pre, post = [], []
    for coll, it in items:
        th = (pd.Timestamp(it["properties"]["datetime"]).tz_localize(None) - jan1).total_seconds() / 3600
        (pre if th < t0 - 12 else post).append((th, coll, it))
    if not post:
        np.savez_compressed(fo, r0=r0, c0=c0, times_h=np.zeros(0, np.float32), sensors=np.array([]), state=np.zeros((0, h, wd), np.int8))
        return dict(fire_id=e.fire_id, status="no scenes", n_pre=len(pre))
    # NBR до пожара: последние безоблачные снимки (до 2) - медиана
    P = []
    for th, coll, it in reversed(pre):
        try:
            nb = nbr_scene(it, coll, shp, tr)
        except Exception:
            token(coll, True); continue
        if np.isfinite(nb).mean() > 0.3:
            P.append(nb)
        if len(P) >= 2:
            break
    if not P:
        np.savez_compressed(fo, r0=r0, c0=c0, times_h=np.zeros(0, np.float32), sensors=np.array([]), state=np.zeros((0, h, wd), np.int8))
        return dict(fire_id=e.fire_id, status="no pre", n_post=len(post))
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore"); pre_nbr = np.nanmedian(np.stack(P), 0)
    other = (FID[w] > 0)                                                  # ячейки любых пожаров года -> не фон
    bgc = ~binary_dilation(other, iterations=3)
    bg = np.repeat(np.repeat(bgc, SUB, 0), SUB, 1)
    known = M & np.isin(S, (1, 5))                                        # время прихода по VIIRS
    times, sens, ST, BF, CF, DN, REC = [], [], [], [], [], [], []
    for th, coll, it in post:
        if th > t1 + 120 or it["properties"].get("eo:cloud_cover", 0) > 50:
            continue
        try:
            nb = nbr_scene(it, coll, shp, tr)
        except Exception:
            token(coll, True); continue
        d = pre_nbr - nb; ok = np.isfinite(d); okb = ok & bg
        if ok[np.repeat(np.repeat(M, SUB, 0), SUB, 1)].mean() < 0.2 or okb.sum() < 2000:
            continue
        med = np.median(d[okb]); thr = max(0.05, np.percentile(d[okb], 97) - med)
        b = ok & (d - med > thr)
        v = uniform_filter(ok.astype(np.float32), 5); p_ = uniform_filter(b.astype(np.float32), 5)
        b = ok & (p_ >= 0.6 * np.maximum(v, 1e-6))
        cf = cells(ok.astype(np.float32), np.mean); bf = cells(b.astype(np.float32), np.sum) / np.maximum(cells(ok.astype(np.float32), np.sum), 1)
        st = np.where(cf >= 0.5, np.where(bf >= 0.5, 1, np.where(bf < 0.15, 0, -1)), -1).astype(np.int8)
        dn = cells(np.where(ok, d - med, 0), np.sum) / np.maximum(cells(ok.astype(np.float32), np.sum), 1)   # средний dNBR над фоном
        kk = known & (A < th - 24) & (st >= 0)
        rec = float((st[kk] == 1).mean()) if kk.sum() >= 20 else np.nan
        if times and th - times[-1] < 1:                                 # соседние сцены одного пролёта - объединяем
            m = ST[-1] < 0; ST[-1][m] = st[m]; BF[-1] = np.maximum(BF[-1], bf); m = cf > CF[-1]; DN[-1][m] = dn[m]; CF[-1] = np.maximum(CF[-1], cf)
            continue
        times.append(th); sens.append(coll.split("-")[0] + ":" + it["id"][:4]); ST.append(st); BF.append(bf); CF.append(cf); DN.append(dn); REC.append(rec)
    K = len(times)
    rec = np.array(REC, np.float32)
    np.savez_compressed(fo, r0=r0, c0=c0, times_h=np.array(times, np.float32), sensors=np.array(sens),
                        state=np.array(ST, np.int8).reshape(K, h, wd), bf=(np.array(BF).reshape(K, h, wd) * 100).astype(np.uint8),
                        cf=(np.array(CF).reshape(K, h, wd) * 100).astype(np.uint8),
                        dn=(np.clip(np.array(DN).reshape(K, h, wd), -1, 1) * 1000).astype(np.int16), recall=rec, reliable=np.nan_to_num(rec) >= 0.7)
    return dict(fire_id=e.fire_id, status="ok", n_pre=len(pre), n_post=len(post), snapshots=K, reliable=int((np.nan_to_num(rec) >= 0.7).sum()))


if __name__ == "__main__":
    ev = pd.read_csv(os.path.join(KZ, "events_kz.csv"))
    ev = ev[(ev.area_km2 >= MIN_KM2) & (ev.year >= 2013)].copy()
    ev["pri"] = ev.split.map({"test": 0, "val": 1}).fillna(2)
    ev = ev.sort_values(["pri", "area_km2"], ascending=[True, False])
    print(f"fires: {len(ev)} (test {int((ev.pri == 0).sum())}, val {int((ev.pri == 1).sum())})", flush=True)
    flog = os.path.join(OD, "_log.csv"); n = 0; t_start = time.time()
    for pri in (0, 1, 2):
        for y in sorted(ev.year.unique()):
            E = ev[(ev.year == y) & (ev.pri == pri)]
            if E.empty or all(os.path.exists(os.path.join(OD, f"{f}.npz")) for f in E.fire_id):
                continue
            a = np.load(os.path.join(KZ, "arrival", f"arr_{y}.npz"))
            AR = np.full((GH, GW), np.nan, np.float32); FID = np.zeros((GH, GW), np.int32); SRC = np.zeros((GH, GW), np.uint8)
            AR[a["r"], a["c"]] = a["arr_h"]; FID[a["r"], a["c"]] = a["fire_id"]; SRC[a["r"], a["c"]] = a["src"]
            with ThreadPoolExecutor(NW) as ex:
                futs = {ex.submit(process, e, AR, FID, SRC): e.fire_id for e in E.itertuples()}
                for f, fid in futs.items():
                    try:
                        r = f.result()
                    except Exception as x:
                        r = dict(fire_id=fid, status="error " + str(x)[:80])
                    if r is None:
                        continue
                    r.update(year=y, split=["test", "val", "train"][pri]); n += 1
                    pd.DataFrame([r]).to_csv(flog, mode="a", header=not os.path.exists(flog), index=False)
                    if n % 20 == 0:
                        print(f"{n} fires, {(time.time() - t_start) / 60:.0f} min, last {r}", flush=True)
    print("done", flush=True)
