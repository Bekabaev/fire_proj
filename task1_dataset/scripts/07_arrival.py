"""Шаг 7. Время прихода огня (arrival time) на сетке куба 30 м: экспертный контур + FIRMS + MCD64A1.

Внутри экспертного контура события каждому пикселю назначается время прихода огня (часы от возгорания):
  src=1  FIRMS: самая ранняя из 5 ближайших термоточек в радиусе 1.5 км (термоточки в контуре, от −24 ч до конца пожара)
  src=2  MCD64A1 (суточная дата выгорания MODIS 500 м, Planetary Computer), если рядом нет термоточек; время = 08 UTC даты
  src=3  ближайшая термоточка дальше 1.5 км (низкая надёжность), если нет ни того, ни другого
Пиксель возгорания = 0 ч. Вне контура - NaN (не сгорело в этом пожаре).
Выход: data/arrival/<event>.npz (arrival_h float32, arrival_src uint8, mcd64_h float32), data/arrival_summary.csv
"""
import os, sys, json
import numpy as np, pandas as pd, requests, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import Affine
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

OD = os.path.join(DATA, "arrival"); os.makedirs(OD, exist_ok=True)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOK = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs", timeout=60).json()["token"]
R_FIRMS, K_NN = 1500.0, 5


def mcd64(meta, e, t_end):
    """Самая ранняя дата выгорания MCD64A1 в окне [возгорание − 2 сут, конец] -> часы от возгорания на сетке куба."""
    t = meta["transform"]; tr = Affine(*t); H, W = meta["H"], meta["W"]
    x0, y1 = t[2], t[5]; bb = transform_bounds(meta["crs"], "EPSG:4326", x0, y1 + H * t[4], x0 + W * t[0], y1, densify_pts=21)
    d0 = e.ign_time_utc.normalize() - pd.Timedelta(days=2)
    j = requests.post(STAC, json={"collections": ["modis-64A1-061"], "bbox": list(bb), "limit": 50,
                                  "datetime": f"{d0.replace(day=1).date()}/{t_end.date()}"}, timeout=120).json()
    out = np.full((H, W), np.nan, np.float32)
    for f in j["features"]:
        with rasterio.open("/vsicurl/" + f["assets"]["Burn_Date"]["href"] + "?" + TOK) as s:
            doy = s.read(1).astype(np.float32)
            year = pd.Timestamp(f["properties"]["start_datetime"]).year
            dst = np.full((H, W), np.nan, np.float32)
            reproject(np.where(doy > 0, doy, np.nan), dst, src_transform=s.transform, src_crs=s.crs, dst_transform=tr,
                      dst_crs=meta["crs"], resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        # часы от возгорания: (DOY − 1) суток от 1 января + 08 UTC (дневной пролёт)
        h = ((dst - 1) * 24 + 8 - (e.ign_time_utc - pd.Timestamp(year=year, month=1, day=1)).total_seconds() / 3600).astype(np.float32)
        ok = np.isfinite(h) & (h >= -48) & (h <= (t_end - e.ign_time_utc).total_seconds() / 3600 + 24)
        out = np.where(ok & (np.isnan(out) | (h < out)), h, out)
    return out, len(j["features"])


ev = load_events(); rows = []
for e in ev.itertuples():
    meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json")))
    d = np.load(os.path.join(DATA, "cubes", e.event + ".npz")); lab = d["label"].astype(bool); H, W = lab.shape
    t = meta["transform"]; x0, y1, px = t[2], t[5], t[0]
    t_end = e.image_date + pd.Timedelta(hours=6) if e.duration_ok else e.ign_time_utc + pd.Timedelta(days=10)
    a = read_firms(e.event)
    if len(a):
        a = a[a.in_poly & (a.h_since_ign >= -24) & (a.time_utc <= t_end)]
    mh, n_items = mcd64(meta, e, t_end)
    arr = np.full((H, W), np.nan, np.float32); src = np.zeros((H, W), np.uint8)
    rr, cc = np.nonzero(lab)
    xs = x0 + (cc + 0.5) * px; ys = y1 - (rr + 0.5) * px
    if len(a):
        tree = cKDTree(np.c_[a.x.values, a.y.values]); hh = a.h_since_ign.values
        dist, ii = tree.query(np.c_[xs, ys], k=min(K_NN, len(a)))
        dist = dist.reshape(len(xs), -1); ii = ii.reshape(len(xs), -1)
        tk = np.where(dist <= R_FIRMS, hh[ii], np.inf).min(1)
        near = np.isfinite(tk)
        arr[rr[near], cc[near]] = np.maximum(tk[near], 0); src[rr[near], cc[near]] = 1
    m_ok = (src[rr, cc] == 0) & np.isfinite(mh[rr, cc])
    arr[rr[m_ok], cc[m_ok]] = np.maximum(mh[rr[m_ok], cc[m_ok]], 0); src[rr[m_ok], cc[m_ok]] = 2
    rest = src[rr, cc] == 0
    if rest.any() and len(a):
        arr[rr[rest], cc[rest]] = np.maximum(hh[ii[rest, 0]], 0); src[rr[rest], cc[rest]] = 3
    ir, ic = meta["ign_rc"]
    if 0 <= ir < H and 0 <= ic < W:
        arr[ir, ic] = 0.0
    np.savez_compressed(os.path.join(OD, e.event + ".npz"), arrival_h=arr, arrival_src=src, mcd64_h=mh)
    n = lab.sum(); fr = lambda k: round(float((src[lab] == k).sum() / n), 3)
    both = lab & (src == 1) & np.isfinite(mh)
    mae = float(np.median(np.abs(arr[both] - mh[both])) / 24) if both.any() else np.nan
    v = arr[lab]
    q = dict(event=e.event, n_firms=len(a), mcd64_items=n_items, frac_firms=fr(1), frac_mcd64=fr(2), frac_far=fr(3),
             frac_unknown=round(float(np.isnan(v).mean()), 3), mcd64_detect_frac=round(float(np.isfinite(mh[lab]).mean()), 3),
             firms_vs_mcd64_med_abs_days=round(mae, 2) if np.isfinite(mae) else np.nan,
             t50_h=round(float(np.nanpercentile(v, 50)), 1) if np.isfinite(v).any() else np.nan,
             t90_h=round(float(np.nanpercentile(v, 90)), 1) if np.isfinite(v).any() else np.nan,
             t_max_h=round(float(np.nanmax(v)), 1) if np.isfinite(v).any() else np.nan)
    for hh_ in (24, 48, 72):
        q[f"burned_by_{hh_}h"] = round(float(np.nanmean(v <= hh_)), 3)
    rows.append(q)
    print(" ".join(f"{k}={v}" for k, v in q.items()), flush=True)
s = pd.DataFrame(rows); s.to_csv(os.path.join(DATA, "arrival_summary.csv"), index=False)
print("\nmedian share of polygon timed by FIRMS:", s.frac_firms.median(), "| by MCD64A1:", s.frac_mcd64.median(),
      "| far NN:", s.frac_far.median(), "| MCD64A1 detects (median share of polygon):", s.mcd64_detect_frac.median())
print("FIRMS vs MCD64A1 median |Δ| days (median over events):", s.firms_vs_mcd64_med_abs_days.median())
print("median t50 / t90 (h):", s.t50_h.median(), s.t90_h.median(), "| burned by 24/48/72 h (median):",
      s.burned_by_24h.median(), s.burned_by_48h.median(), s.burned_by_72h.median())
