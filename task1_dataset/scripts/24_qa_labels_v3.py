"""Шаг 24 (датасет v3). Проверка меток v3 (шаг 18) по экспертным контурам заповедников (шаги 3, 7) - независимая граница.

Для каждого экспертного пожара 2013–2024 экспертный контур (куб 30 м) и экспертное время прихода (только src=1 - термоточки
внутри контура) переносятся на национальную сетку 500 м (доля контура в ячейке; минимум времени). Сравниваются ячейки v3
любых пожаров GFA в окне события [возгорание − 2 сут, конец + 2 сут]:
- полнота площади: доля ячеек контура (доля ≥ 0,5), покрытых MCD64A1 (src 1–4) и MCD64A1 + расширение VIIRS (src 1–5);
- точность: доля ячеек v3 в окне куба, попавших в контур (доля > 0);
- ошибка времени |v3 − эксперт| (ч) по источнику v3 и для суточной метки v2 (полдень местной даты MCD64A1).
  Важно: время src 1–2 и экспертное src=1 построены по одним и тем же термоточкам - для них сравнение не независимо;
  независимы src 3/4 и суточная метка v2.
Выход: data/ext_kz/qa_labels_v3.csv (по событиям), data/ext_kz/qa_labels_v3.txt (сводка)
"""
import os, sys, json
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine, from_origin
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

ev = load_events(); rows = []; terr = []
for e in ev.itertuples():
    y = e.ign_time_utc.year
    fa = os.path.join(os.environ.get("ARR_DIR", os.path.join(KZ, "arrival")), f"arr_{y}.npz")
    if not os.path.exists(fa):
        continue
    meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json"))); H, W = meta["H"], meta["W"]
    lab = np.load(os.path.join(DATA, "cubes", e.event + ".npz"))["label"].astype(np.float32)
    ar = np.load(os.path.join(DATA, "arrival", e.event + ".npz"))
    t0 = (e.ign_time_utc - pd.Timestamp(year=y, month=1, day=1)).total_seconds() / 3600
    xt = np.where((ar["arrival_src"] == 1) & (lab > 0), ar["arrival_h"] + t0, np.nan).astype(np.float32)
    tr = Affine(*meta["transform"])
    # окно национальной сетки, покрывающее куб
    from rasterio.warp import transform_bounds
    bx = transform_bounds(meta["crs"], CRS, tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f, densify_pts=21)
    c0 = int((bx[0] - X0) // CELL); c1 = int((bx[2] - X0) // CELL) + 1; r0 = int((Y1 - bx[3]) // CELL); r1 = int((Y1 - bx[1]) // CELL) + 1
    gt = from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL); sh = (r1 - r0, c1 - c0)
    fr = np.zeros(sh, np.float32); reproject(lab, fr, src_transform=tr, src_crs=meta["crs"], dst_transform=gt, dst_crs=CRS, resampling=Resampling.average)
    tx = np.full(sh, np.nan, np.float32)
    reproject(xt, tx, src_transform=tr, src_crs=meta["crs"], dst_transform=gt, dst_crs=CRS, resampling=Resampling.min, src_nodata=np.nan, dst_nodata=np.nan)
    fr_cov = np.zeros(sh, np.float32)          # доля ячейки, покрытая кубом
    reproject(np.ones(lab.shape, np.float32), fr_cov, src_transform=tr, src_crs=meta["crs"], dst_transform=gt, dst_crs=CRS, resampling=Resampling.average)
    a = np.load(fa); m = (a["r"] >= r0) & (a["r"] < r1) & (a["c"] >= c0) & (a["c"] < c1)
    t_end = (e.image_date - pd.Timestamp(year=y, month=1, day=1)).total_seconds() / 3600 + 30 if e.duration_ok else t0 + 240
    m &= (a["arr_h"] >= t0 - 48) & (a["arr_h"] <= t_end + 48)
    V = np.full(sh, np.nan, np.float32); S = np.zeros(sh, np.uint8)
    V[a["r"][m] - r0, a["c"][m] - c0] = a["arr_h"][m]; S[a["r"][m] - r0, a["c"][m] - c0] = a["src"][m]
    doy = np.asarray(load_doy(y))[r0:r1, c0:c1].astype(np.float32)
    in_cube = fr_cov > 0.9
    ex = (fr >= 0.5) & in_cube; anyx = fr > 0
    n = int(ex.sum())
    if n == 0:
        continue
    v3 = (S > 0) & in_cube
    row = dict(event=e.event, year=y, expert_cells=n,
               recall_mcd=round(float(((S >= 1) & (S <= 4))[ex].mean()), 3), recall_mcd_ext=round(float((S > 0)[ex].mean()), 3),
               precision_mcd=round(float(anyx[v3 & (S <= 4)].mean()), 3) if (v3 & (S <= 4)).any() else np.nan,
               precision_ext_only=round(float(anyx[v3 & (S == 5)].mean()), 3) if (v3 & (S == 5)).any() else np.nan,
               ext_cells=int((v3 & (S == 5)).sum()))
    ok = np.isfinite(tx) & np.isfinite(V)
    for k in range(1, 6):
        mk = ok & (S == k)
        row[f"abs_err_h_src{k}"] = round(float(np.median(np.abs(V[mk] - tx[mk]))), 1) if mk.sum() >= 3 else np.nan
        terr += [dict(event=e.event, src=k, err=float(d)) for d in (V[mk] - tx[mk])]
    mv2 = np.isfinite(tx) & (doy > 0)
    v2t = (doy - 1) * 24 + 6
    row["abs_err_h_v2_day"] = round(float(np.median(np.abs(v2t[mv2] - tx[mv2]))), 1) if mv2.sum() >= 3 else np.nan
    terr += [dict(event=e.event, src=0, err=float(d)) for d in (v2t[mv2] - tx[mv2])]
    rows.append(row); print(row, flush=True)

QO = os.environ.get("ARR_DIR", KZ)
q = pd.DataFrame(rows); q.to_csv(os.path.join(QO, "qa_labels_v3.csv"), index=False)
T = pd.DataFrame(terr)
lines = [f"events: {len(q)}, expert cells: {int(q.expert_cells.sum())}",
         f"area recall (cell-weighted): MCD64A1 {np.average(q.recall_mcd, weights=q.expert_cells):.3f} -> MCD64A1+VIIRS ext {np.average(q.recall_mcd_ext, weights=q.expert_cells):.3f}",
         f"precision (median over events): MCD64A1 cells {q.precision_mcd.median():.3f}, VIIRS-ext cells {q.precision_ext_only.median():.3f}; ext cells total {int(q.ext_cells.sum())}",
         "timing |error| vs expert FIRMS arrival, hours (all cells pooled): median / p75 / n"]
names = {0: "v2 day label (MCD64A1 date, local noon)", 1: "v3 src1 VIIRS (not independent)", 2: "v3 src2 MODIS (not independent)",
         3: "v3 src3 interpolated", 4: "v3 src4 MCD64A1 day + offset", 5: "v3 src5 VIIRS ext (not independent)"}
for k, g in T.groupby("src"):
    lines.append(f"  {names[k]}: {g.err.abs().median():.1f} / {g.err.abs().quantile(.75):.1f} / {len(g)}  (bias {g.err.median():+.1f})")
open(os.path.join(QO, "qa_labels_v3.txt"), "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
