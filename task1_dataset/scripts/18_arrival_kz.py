"""Шаг 18 (датасет v3). Субсуточное время прихода огня для пожаров Казахстана (GFA) по термоточкам FIRMS.

Главное ограничение v2: метка GFA/MCD64A1 - только день выгорания (точность ±1 сут), и MCD64A1 пропускает часть площади
степных пожаров. Здесь каждой ячейке 500 м пожара назначается время прихода огня в часах UTC от 1 января (как шаг 7 для заповедников):
  src=1  VIIRS (375 м, S-NPP / NOAA-20): самая ранняя термоточка в радиусе 600 м от центра ячейки,
         местные сутки термоточки в [дата MCD64A1 − 5, дата + 1] (дата MCD64A1 запаздывает, шаг 24)
  src=2  MODIS (1 км): то же, радиус 1000 м, если нет VIIRS
  src=3  интерполяция: медиана времени ≤ 8 ближайших ячеек этого же пожара с src 1–2 в радиусе 3 км (согласованных по дате)
         Окно −5 сут и радиус 3 км выбраны по проверке на экспертных контурах (шаг 24): при −2 сут / 1,5 км ячейки src=4
         (30 % площади) опаздывали на ~72 ч против экспертного времени; теперь src=4 ≈ 13 % площади.
  src=4  только дата MCD64A1: время = (DOY − 1)·24 ч + сдвиг; сдвиг - медиана (время VIIRS − начало даты MCD64A1) по ячейкам src=1
  src=5  расширение: ячейка вне контура MCD64A1 (в этом году не выгорала по MCD64A1), в которой есть термоточки VIIRS
         (confidence n/h) в окне пожара, связанная с контуром пожара (8-связность, разрыв ≤ 1 ячейки); время - самая ранняя термоточка
Неопределённость (ч, для весов при обучении): src1 6, src2 12, src3 12, src4 24, src5 6.
Местные сутки = UTC+6. Термоточки - только type 0 (растительность), MODIS confidence ≥ 30, VIIRS - без 'l' для src=5.
Выход: data/ext_kz/arrival/arr_<год>.npz (r, c, fire_id int32, arr_h float32, src uint8), data/ext_kz/arrival_summary.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from rasterio.warp import transform as warp_transform
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OD = os.environ.get("ARR_DIR", os.path.join(KZ, "arrival")); os.makedirs(OD, exist_ok=True)
R_V, R_M = 600.0, 1000.0
R_INT = float(os.environ.get("R_INT", 3000)); LAG = int(os.environ.get("LAG", 5))   # LAG - на сколько суток термоточка может опережать дату MCD64A1
UNC = {1: 6, 2: 12, 3: 12, 4: 24, 5: 6}
st8 = np.ones((3, 3), bool)


def read_af(y):
    fa = pd.read_csv(os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz"), parse_dates=["time_utc"])
    fa = fa[fa["type"].fillna(0) == 0].copy()
    viirs = fa.source.str.startswith("VIIRS")
    cnum = pd.to_numeric(fa.conf, errors="coerce")
    fa = fa[viirs | (cnum >= 30)].copy(); viirs = fa.source.str.startswith("VIIRS")
    px, py = warp_transform("EPSG:4326", CRS, fa.lon.tolist(), fa.lat.tolist())
    fa["x"], fa["y"] = np.array(px), np.array(py)
    fa["gr"] = ((Y1 - fa.y) // CELL).astype(int); fa["gc"] = ((fa.x - X0) // CELL).astype(int)
    fa["t_h"] = ((fa.time_utc - pd.Timestamp(year=y, month=1, day=1)).dt.total_seconds() / 3600).astype(np.float32)
    fa["ldoy"] = np.floor((fa.t_h + 6) / 24).astype(int) + 1          # местный день года
    fa["viirs"] = viirs.values; fa["conf_ok"] = ~(viirs & (fa.conf.astype(str) == "l"))
    return fa[(fa.gr >= 0) & (fa.gr < GH) & (fa.gc >= 0) & (fa.gc < GW)].reset_index(drop=True)


def earliest(tree, t, ld, xy, cdoy, r):
    """Самая ранняя термоточка в радиусе r с местными сутками в [cdoy − 2, cdoy + 1]; NaN - нет."""
    out = np.full(len(xy), np.nan, np.float32)
    if tree is None:
        return out
    for i, js in enumerate(tree.query_ball_point(xy, r)):
        if js:
            js = np.asarray(js); ok = (ld[js] >= cdoy[i] - LAG) & (ld[js] <= cdoy[i] + 1)
            if ok.any():
                out[i] = t[js[ok]].min()
    return out


ev_all = pd.read_csv(os.path.join(KZ, "events_kz.csv"))
summ = []
YRS = [int(v) for v in os.environ["ONLY_YEARS"].split(",")] if os.environ.get("ONLY_YEARS") else YEARS[1:]
for y in YRS:
    fo = os.path.join(OD, f"arr_{y}.npz")
    if os.path.exists(fo) and not os.environ.get("ARR_DIR"):
        continue
    doy = np.asarray(load_doy(y)); fid = np.load(os.path.join(KZ, f"fire_id_{y}.npy"))
    fa = read_af(y); V = fa[fa.viirs]; M = fa[~fa.viirs]
    rr, cc = np.nonzero(fid > 0); F = fid[rr, cc]; D = doy[rr, cc].astype(int)
    xy = np.c_[X0 + (cc + 0.5) * CELL, Y1 - (rr + 0.5) * CELL]
    tv = cKDTree(V[["x", "y"]].values) if len(V) else None; tm = cKDTree(M[["x", "y"]].values) if len(M) else None
    arr = earliest(tv, V.t_h.values, V.ldoy.values, xy, D, R_V); src = np.where(np.isfinite(arr), 1, 0).astype(np.uint8)
    am = earliest(tm, M.t_h.values, M.ldoy.values, xy, D, R_M)
    k = (src == 0) & np.isfinite(am); arr[k] = am[k]; src[k] = 2
    # src 3: интерполяция по ближайшим датированным ячейкам того же пожара
    dated = src > 0
    if dated.any():
        td = cKDTree(xy[dated]); fd = F[dated]; ad = arr[dated]
        q = np.nonzero(~dated)[0]
        dist, ii = td.query(xy[q], k=min(8, int(dated.sum())), distance_upper_bound=R_INT)
        dist = dist.reshape(len(q), -1); ii = ii.reshape(len(q), -1)
        okn = np.isfinite(dist); ii_ = np.where(okn, ii, 0)
        okn &= fd[ii_] == F[q][:, None]
        ld_n = np.floor((ad[ii_] + 6) / 24) + 1
        okn &= (ld_n >= D[q][:, None] - LAG) & (ld_n <= D[q][:, None] + 1)
        vals = np.where(okn, ad[ii_], np.nan)
        has = okn.any(1)
        arr[q[has]] = np.nanmedian(vals[has], 1); src[q[has]] = 3
    # src 4: дата MCD64A1 + калиброванный сдвиг
    s1 = src == 1
    off = float(np.median(arr[s1] - (D[s1] - 1) * 24)) if s1.any() else 6.0
    k = src == 0; arr[k] = (D[k] - 1) * 24 + off; src[k] = 4
    # src 5: расширение термоточками VIIRS вне контура MCD64A1
    Vok = V[V.conf_ok]
    vcell = Vok.groupby(["gr", "gc", "ldoy"]).t_h.min().reset_index()
    ext_r, ext_c, ext_f, ext_t = [], [], [], []
    taken = np.zeros((GH, GW), bool)
    E = ev_all[ev_all.year == y].sort_values("area_km2", ascending=False)
    first = pd.Series(arr).groupby(F).min(); last = pd.Series(arr).groupby(F).max()
    pad = 10
    vg = vcell.set_index(["gr", "gc"]).sort_index()
    for e in E.itertuples():
        if e.fire_id not in first.index:
            continue
        r0, r1, c0, c1 = max(e.rmin - pad, 0), min(e.rmax + pad + 1, GH), max(e.cmin - pad, 0), min(e.cmax + pad + 1, GW)
        l0 = int(np.floor((first[e.fire_id] + 6) / 24)) + 1 - 1; l1 = int(np.floor((last[e.fire_id] + 6) / 24)) + 1 + 1
        w = vcell[(vcell.gr >= r0) & (vcell.gr < r1) & (vcell.gc >= c0) & (vcell.gc < c1) & (vcell.ldoy >= l0) & (vcell.ldoy <= l1)]
        if w.empty:
            continue
        w = w.groupby(["gr", "gc"]).t_h.min().reset_index()
        free = (doy[w.gr, w.gc] == 0) & (fid[w.gr, w.gc] == 0) & ~taken[w.gr, w.gc]
        w = w[free]
        if w.empty:
            continue
        foot = fid[r0:r1, c0:c1] == e.fire_id
        det = np.zeros_like(foot); det[w.gr - r0, w.gc - c0] = True
        lab, _ = ndimage.label(ndimage.binary_dilation(foot | det, st8), structure=st8)
        keep_lab = np.unique(lab[foot]); keep_lab = keep_lab[keep_lab > 0]
        sel = np.isin(lab[w.gr - r0, w.gc - c0], keep_lab)
        w = w[sel]
        if w.empty:
            continue
        taken[w.gr, w.gc] = True
        ext_r.append(w.gr.values); ext_c.append(w.gc.values); ext_f.append(np.full(len(w), e.fire_id, np.int32)); ext_t.append(w.t_h.values)
    if ext_r:
        er, ec, ef, et = map(np.concatenate, (ext_r, ext_c, ext_f, ext_t))
    else:
        er = ec = ef = np.zeros(0, int); et = np.zeros(0, np.float32)
    R_ = np.r_[rr, er].astype(np.int32); C_ = np.r_[cc, ec].astype(np.int32); FF = np.r_[F, ef].astype(np.int32)
    A_ = np.r_[arr, et].astype(np.float32); S_ = np.r_[src, np.full(len(er), 5, np.uint8)].astype(np.uint8)
    np.savez_compressed(fo, r=R_, c=C_, fire_id=FF, arr_h=A_, src=S_, offset_h=off)
    lag = (D[s1] - 1) * 24 - arr[s1]
    row = dict(year=y, fire_cells=len(rr), ext_cells=len(er), offset_h=round(off, 1),
               **{f"src{k}": round(float((S_ == k).mean()), 3) for k in range(1, 6)},
               mcd_minus_viirs_days_median=round(float(np.median(lag) / 24), 2), af_viirs=len(V), af_modis=len(M))
    summ.append(row); print(row, flush=True)

if summ:
    s = pd.DataFrame(summ); f = os.path.join(OD, "..", "arrival_summary.csv") if not os.environ.get("ARR_DIR") else os.path.join(OD, "summary.csv")
    if os.path.exists(f):
        s = pd.concat([pd.read_csv(f), s]).drop_duplicates("year", keep="last").sort_values("year")
    s.to_csv(f, index=False); print(s.to_string())
print("done")
