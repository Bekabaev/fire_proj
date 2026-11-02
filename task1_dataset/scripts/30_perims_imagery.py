"""Шаг 30 (датасет v3). Промежуточные контуры экспертных пожаров по снимкам Sentinel-2 / Landsat - независимая проверка динамики.

До сих пор время прихода огня внутри экспертного контура восстанавливалось только по термоточкам (шаг 7), а контур был лишь
финальный. Здесь для каждого пожара, горевшего дольше 12 ч, берутся все снимки Sentinel-2 L2A и Landsat 7/8/9 C2 L2
(Planetary Computer, без ключа) между возгоранием и концом пожара + 2 сут и строится карта «выгорело к моменту снимка» на сетке куба 30 м:
- NBR = (NIR − SWIR2) / (NIR + SWIR2): S2 - B8A, B12 (поправка −1000 для базовой линии ≥ 04.00), Landsat - nir08, swir22;
- маска облаков: S2 SCL 0/1/3/8/9/10/11 (класс 2 «тёмные пиксели» НЕ маскируется - так часто помечаются свежие гари),
  Landsat QA_PIXEL биты 0–3 (тени облаков не маскируются: гарь тоже тёмная, а NBR в тени почти не меняется);
- dNBR = pre_NBR (куб, до пожара) − NBR снимка; фон - пиксели куба вне контура (без ignore): выгорело, если
  dNBR − медиана фона > max(0,05, p97 фона − медиана фона) - порог учитывает сезонное высыхание травы на том же снимке.
Оценка (только внутри экспертного контура, безоблачные пиксели): совпадение «состояния к моменту снимка» по
(а) экспертному времени прихода (термоточки, шаг 7), (б) суточной метке MCD64A1 (v2), (в) метке v3 по GFA (шаг 18, ячейки 500 м).
Распознавание гари неполное, поэтому «нет гари» используется только на надёжных снимках (распознано ≥ 70 % пикселей, заведомо
сгоревших по термоточкам ≥ 24 ч до снимка); гарь очищается фильтром большинства 5 × 5.
Уточнение времени прихода: верхняя граница - первый снимок, где выгорело; нижняя - последний надёжный снимок, где не выгорело;
время, выходящее за границы более чем на 12 ч, сдвигается внутрь (если границы непротиворечивы), источник = 6 «уточнено по снимку»;
позже сдвигается только время без термоточки (интерполяция / дата MCD64A1) - время по термоточке является прямым наблюдением.
Выход: data/perims/<event>.npz (times_h от возгорания, sensors, burned int8 [K,H,W]: 1/0/−1 нет данных, thr),
       data/arrival_img/<event>.npz (arrival_h, arrival_src с кодом 6), data/qa_perims.csv, data/qa_perims.txt
"""
import os, sys, json
import numpy as np, pandas as pd, requests, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import KZ, CRS, CELL, X0, Y1, load_doy

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60"); os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3"); os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "NO"); os.environ.setdefault("GDAL_HTTP_VERSION", "1.1")
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
POOL = ThreadPoolExecutor(3)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PD = os.path.join(DATA, "perims"); AD = os.path.join(DATA, "arrival_img"); os.makedirs(PD, exist_ok=True); os.makedirs(AD, exist_ok=True)
TOK = {}


def token(coll):
    TOK[coll] = requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{coll}", timeout=60).json()["token"]


def read(href, coll, shp, tr, crs, rs, nod):
    out = np.full(shp, np.nan, np.float32)
    with rasterio.open("/vsicurl/" + href + "?" + TOK[coll]) as s:
        b = transform_bounds(crs, s.crs, tr.c, tr.f + shp[0] * tr.e, tr.c + shp[1] * tr.a, tr.f, densify_pts=21)
        w = rasterio.windows.from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        a = s.read(1, window=w).astype(np.float32)
        reproject(a, out, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=tr, dst_crs=crs, resampling=rs,
                  src_nodata=nod, dst_nodata=np.nan)
    return out


def search(coll, bb, t0, t1, extra=None):
    q = {"collections": [coll], "bbox": list(bb), "limit": 200, "datetime": f"{t0:%Y-%m-%dT%H:%M:%SZ}/{t1:%Y-%m-%dT%H:%M:%SZ}",
         "query": {"eo:cloud_cover": {"lte": 80}, **(extra or {})}}
    return requests.post(STAC, json=q, timeout=120).json().get("features", [])


def nbr_scene(it, coll, shp, tr, crs):
    """NBR снимка на сетке куба (NaN - облако / нет данных)."""
    if coll == "sentinel-2-l2a":
        off = 1000.0 if float(it["properties"].get("s2:processing_baseline", "0")) >= 4.0 else 0.0
        fs = [POOL.submit(read, it["assets"][b]["href"], coll, shp, tr, crs, Resampling.bilinear, 0) for b in ("B8A", "B12")]
        fq = POOL.submit(read, it["assets"]["SCL"]["href"], coll, shp, tr, crs, Resampling.nearest, 0)
        nir, sw = [(f.result(timeout=240) - off) / 1e4 for f in fs]; scl = fq.result(timeout=240)
        bad = ~np.isfinite(scl) | np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    else:
        fs = [POOL.submit(read, it["assets"][b]["href"], coll, shp, tr, crs, Resampling.bilinear, 0) for b in ("nir08", "swir22")]
        fq = POOL.submit(read, it["assets"]["qa_pixel"]["href"], coll, shp, tr, crs, Resampling.nearest, 1)
        nir, sw = [f.result(timeout=240) * 0.0000275 - 0.2 for f in fs]; qa = fq.result(timeout=240)
        bad = ~np.isfinite(qa) | ((np.nan_to_num(qa, nan=1).astype(np.uint16) & 0b1111) > 0)
    bad |= ~np.isfinite(nir) | ~np.isfinite(sw) | (nir <= 0) | (sw <= 0) | (nir > 1.2) | (sw > 1.2)
    return np.where(bad, np.nan, (nir - sw) / (nir + sw + 1e-9)).astype(np.float32)


def f1(pred, true):
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


ev = load_events(); rows = []
for c in ("sentinel-2-l2a", "landsat-c2-l2"):
    token(c)
for e in ev.itertuples():
    fo = os.path.join(PD, e.event + ".npz")
    meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json"))); H, W = meta["H"], meta["W"]; tr = Affine(*meta["transform"]); crs = meta["crs"]
    cb = np.load(os.path.join(DATA, "cubes", e.event + ".npz")); lab = cb["label"].astype(bool); ign = cb["ignore"] > 0
    ar = np.load(os.path.join(DATA, "arrival", e.event + ".npz")); A = ar["arrival_h"]; SRC = ar["arrival_src"]
    if np.isfinite(A[lab]).sum() <= 1:
        continue
    t_end = float(np.nanmax(A[lab]))
    if t_end < 12:
        continue
    pre = cb["x"][meta["channels"].index("pre_NBR")].astype(np.float32)
    bg = ~lab & ~ign & np.isfinite(pre)
    if os.path.exists(fo):
        z = np.load(fo); times, sens, B = list(z["times_h"]), list(z["sensors"]), list(z["burned"])
    else:
        bb = transform_bounds(crs, "EPSG:4326", tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f, densify_pts=21)
        T0 = e.ign_time_utc; T1 = T0 + pd.Timedelta(hours=t_end + 48)
        items = [("sentinel-2-l2a", it) for it in search("sentinel-2-l2a", bb, T0, T1)]
        items += [("landsat-c2-l2", it) for it in search("landsat-c2-l2", bb, T0, T1, {"platform": {"in": ["landsat-7", "landsat-8", "landsat-9"]}})]
        items.sort(key=lambda x: x[1]["properties"]["datetime"])
        print(f"{e.event}: t_end {t_end:.0f} h, {len(items)} scenes", flush=True)
        times, sens, B = [], [], []
        for coll, it in items:
            th = (pd.Timestamp(it["properties"]["datetime"]).tz_localize(None) - T0).total_seconds() / 3600
            try:
                nb = nbr_scene(it, coll, (H, W), tr, crs)
            except FTimeout:
                print("   timeout", it["id"], flush=True); POOL.shutdown(wait=False, cancel_futures=True); POOL = ThreadPoolExecutor(3); continue
            except Exception as ex:
                print("   failed", it["id"], str(ex)[:90], flush=True); token(coll); continue
            d = pre - nb; ok = np.isfinite(d)
            clear_in = ok[lab].mean(); okb = ok & bg
            if clear_in < 0.2 or okb.sum() < 500:
                print(f"   {it['id'][:40]:40s} {th:7.1f} h  clear in contour {clear_in:.2f} - skip", flush=True); continue
            med = np.median(d[okb]); thr = max(0.05, np.percentile(d[okb], 97) - med)
            b = np.full((H, W), -1, np.int8); b[ok] = (d[ok] - med > thr).astype(np.int8)
            times.append(th); sens.append(coll.split("-")[0] + ":" + it["id"][:4]); B.append(b)
            print(f"   {it['id'][:40]:40s} {th:7.1f} h  clear {clear_in:.2f}  thr {thr:.3f}  burned in contour {(b[lab] == 1).sum() / max((b[lab] >= 0).sum(), 1):.2f}"
                  f"  outside {(b[bg] == 1).sum() / max((b[bg] >= 0).sum(), 1):.3f}", flush=True)
        # несколько снимков в один час (соседние сцены одного пролёта) объединяются
        if B:
            o = np.argsort(times); times = [times[i] for i in o]; sens = [sens[i] for i in o]; B = [B[i] for i in o]
            mt, ms, mb = [], [], []
            for t, s, b in zip(times, sens, B):
                if mt and t - mt[-1] < 1:
                    m = mb[-1] < 0; mb[-1][m] = b[m]
                else:
                    mt.append(t); ms.append(s); mb.append(b.copy())
            times, sens, B = mt, ms, mb
        np.savez_compressed(fo, times_h=np.array(times, np.float32), sensors=np.array(sens), burned=np.array(B, np.int8).reshape(-1, H, W))
    if not B:
        rows.append(dict(event=e.event, t_end_h=round(t_end, 1), snapshots=0)); continue
    # --- очистка и надёжность снимков. Распознавание гари неполное (слабый сигнал в сухой степи / лесостепи): «нет гари на снимке»
    # не доказывает, что огонь ещё не дошёл. Поэтому: гарь - после фильтра большинства 5 × 5 (≥ 60 % соседей); «ещё не сгорело»
    # используется только на надёжных снимках - где распознано ≥ 70 % заведомо сгоревших пикселей (термоточка ≥ 24 ч до снимка).
    Bc, rel, rec = [], [], []
    for t, b in zip(times, B):
        v = uniform_filter((b >= 0).astype(np.float32), 5); p_ = uniform_filter((b == 1).astype(np.float32), 5)
        c = np.where(b < 0, -1, (p_ >= 0.6 * np.maximum(v, 1e-6)) & (v > 0)).astype(np.int8); Bc.append(c)
        conf = lab & (SRC == 1) & (A < t - 24) & (c >= 0)
        rc = float((c[conf] == 1).mean()) if conf.sum() >= 200 else np.nan
        rec.append(rc); rel.append(bool(rc >= 0.7))
    # --- уточнение времени прихода: сдвиг только при расхождении > 12 ч
    lo = np.full((H, W), -np.inf, np.float32); hi = np.full((H, W), np.inf, np.float32)
    for t, c, ok in zip(times, Bc, rel):
        hi = np.where(c == 1, np.minimum(hi, t), hi)
        if ok:
            lo = np.where(c == 0, np.maximum(lo, t), lo)
    cons = lab & (lo < hi)
    A2 = A.copy(); S2 = SRC.copy()
    late = cons & np.isfinite(A) & (A > hi + 12); early = cons & np.isfinite(A) & (A < lo - 12) & (SRC != 1)      # время по термоточке (src 1) позже не сдвигается
    A2[late] = np.where(np.isfinite(lo[late]), (np.maximum(lo[late], 0) + hi[late]) / 2, np.maximum(hi[late] - 6, 0))
    A2[early] = np.where(np.isfinite(hi[early]), (lo[early] + hi[early]) / 2, lo[early] + 6)
    S2[late | early] = 6
    np.savez_compressed(os.path.join(AD, e.event + ".npz"), arrival_h=A2, arrival_src=S2, mcd64_h=ar["mcd64_h"])
    # --- оценка меток: экспертное время, суточная MCD64A1 (v2), v3 по GFA (500 м)
    y = e.ign_time_utc.year; t_ign = (e.ign_time_utc - pd.Timestamp(year=y, month=1, day=1)).total_seconds() / 3600
    mcd = ar["mcd64_h"]
    fa = os.path.join(KZ, "arrival", f"arr_{y}.npz"); g3 = None
    if os.path.exists(fa):
        a3 = np.load(fa)
        bx = transform_bounds(crs, CRS, tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f, densify_pts=21)
        c0 = int((bx[0] - X0) // CELL); c1 = int((bx[2] - X0) // CELL) + 1; r0 = int((Y1 - bx[3]) // CELL); r1 = int((Y1 - bx[1]) // CELL) + 1
        m = (a3["r"] >= r0) & (a3["r"] < r1) & (a3["c"] >= c0) & (a3["c"] < c1)
        G = np.full((r1 - r0, c1 - c0), np.nan, np.float32); G[a3["r"][m] - r0, a3["c"][m] - c0] = a3["arr_h"][m] - t_ign
        g3 = np.full((H, W), np.nan, np.float32)
        from rasterio.transform import from_origin
        reproject(G, g3, src_transform=from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL), src_crs=CRS, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    for k, (t, b) in enumerate(zip(times, Bc)):
        if t > t_end + 24:
            continue
        ev_m = lab & (b >= 0)
        if ev_m.sum() < 100:
            continue
        obs = b[ev_m] == 1
        r = dict(event=e.event, t_end_h=round(t_end, 1), snapshot=k, t_h=round(t, 1), sensor=sens[k], pixels=int(ev_m.sum()),
                 recall_known=round(rec[k], 3) if np.isfinite(rec[k]) else np.nan, reliable=rel[k],
                 burned_share_img=round(float(obs.mean()), 3))
        for name, arr in (("expert", A), ("expert_img", A2), ("mcd64_day", mcd), ("v3_gfa", g3)):
            if arr is None:
                continue
            st = np.nan_to_num(arr[ev_m], nan=np.inf) <= t
            r[f"f1_{name}"] = round(float(f1(st, obs)), 3); r[f"acc_{name}"] = round(float((st == obs).mean()), 3)
            r[f"late_{name}"] = round(float((obs & ~st).mean()), 3)      # на снимке уже выгорело, по метке - ещё нет
            r[f"early_{name}"] = round(float((~obs & st).mean()), 3)     # по метке выгорело, на снимке - нет (значимо только на надёжных)
            r[f"late_share_{name}"] = round(float((obs & ~st).sum() / max(obs.sum(), 1)), 3)   # доля видимой гари, которую метка «не знает»
        rows.append(r)
    print(f"{e.event}: snapshots {len(times)} (reliable {sum(rel)}), recall on known-burned {[round(x, 2) for x in rec]}, "
          f"adjusted earlier {int(late.sum())} / later {int(early.sum())} px of {int(lab.sum())}", flush=True)

q = pd.DataFrame(rows); q.to_csv(os.path.join(DATA, "qa_perims.csv"), index=False)
s = q.dropna(subset=["f1_expert"]) if "f1_expert" in q else q.iloc[:0]
mid = s[(s.burned_share_img > 0.02) & (s.burned_share_img < 0.98)]       # снимки посреди пожара (не начало / не конец)
lines = [f"events with snapshots: {s.event.nunique()}, snapshots inside fire window: {len(s)}, mid-fire: {len(mid)}, reliable mid-fire: {int(mid.reliable.sum())}",
         "late_share = доля гари, видимой на снимке, которую метка к этому моменту ещё не считает сгоревшей (запаздывание метки) - все снимки;",
         "F1 / acc / early - только надёжные снимки (распознано ≥ 70 % заведомо сгоревшего)"]
rm = mid[mid.reliable]
for name in ("expert", "mcd64_day", "v3_gfa", "expert_img"):
    if f"f1_{name}" in mid and mid[f"f1_{name}"].notna().any():
        w = mid.burned_share_img * mid.pixels; wr = rm.pixels
        lines.append(f"  {name:11s} late_share {np.average(mid[f'late_share_{name}'].fillna(0), weights=w):.3f} (n={len(mid)})"
                     + (f" | reliable: F1 {np.average(rm[f'f1_{name}'].fillna(0), weights=wr):.3f}, acc {np.average(rm[f'acc_{name}'].fillna(0), weights=wr):.3f}, "
                        f"early {np.average(rm[f'early_{name}'].fillna(0), weights=wr):.3f} (n={len(rm)})" if len(rm) else ""))
open(os.path.join(DATA, "qa_perims.txt"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
