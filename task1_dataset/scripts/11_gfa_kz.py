"""Шаг 11 (расширение выборки). Global Fire Atlas v2 (Andela et al. 2019; Zenodo 14065246, CC BY 4.0) по Казахстану.

A. Растр даты выгорания GFA (день года, 500 м MODIS, по MCD64A1 C6.1) -> национальная сетка 500 м EPSG:6933 -> data/ext_kz/doy_<год>.npy
B. Точки возгорания GFA (по одной на пожар) внутри Казахстана, 2012–2024 -> отбор пожаров для срезов:
   длительность ≥ 2 сут (есть динамика) и площадь ≥ 1 км² (≥ 4 ячейки). Для каждого пожара - компонента связности
   выгоревших ячеек (8-связность, даты в [start−1, end+1]), содержащая точку возгорания -> площадь, bbox, длительность по растру.
Выход: data/ext_kz/events_kz.csv, data/ext_kz/fire_id_<год>.npy (номер пожара в ячейке, int32; 0 = нет)
"""
import os, sys, glob, zipfile
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

# --- A. даты выгорания на национальной сетке
for y in YEARS:
    f = glob.glob(os.path.join(GFA_DIR, "day_of_burn", f"*day_of_burn_{y}.tif"))
    if os.path.exists(doy_path(y)) or not f:
        continue
    with rasterio.open(f[0]) as s:
        bb = transform_bounds(CRS, s.crs, X0, Y1 - GH * CELL, X0 + GW * CELL, Y1, densify_pts=50)
        w = from_bounds(*bb, s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w).astype(np.float32); a[a <= 0] = np.nan
        out = np.full((GH, GW), np.nan, np.float32)
        reproject(a, out, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=GTR, dst_crs=CRS,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    np.save(doy_path(y), np.nan_to_num(out).astype(np.int16))
    print(y, "burned cells:", int(np.isfinite(out).sum()), flush=True)

# --- B. пожары
kz = gpd.read_file(KZ_BOUNDARY).to_crs(4326).geometry.union_all()
zf = os.path.join(GFA_DIR, "SHP_ignitions.zip")
names = zipfile.ZipFile(zf).namelist()
rows = []
st8 = np.ones((3, 3), bool)
for y in YEARS:
    shp = [n for n in names if n.endswith(f"_{y}.shp")]
    if not shp or not os.path.exists(doy_path(y)):
        print(y, "skip (no data)"); continue
    g = gpd.read_file(f"zip://{zf}!{shp[0]}", bbox=(46.6, 40.6, 87.3, 55.4))
    g = g[g.within(kz)].copy()
    if y == YEARS[0] or not rows:
        print("GFA ignition columns:", list(g.columns))
    cols = {c.lower(): c for c in g.columns}
    dur = g[cols["duration"]]; size = g[cols["size"]]
    sel = g[(dur >= 2) & (size >= 1.0)].copy()
    doy = np.asarray(load_doy(y)); fid = np.zeros((GH, GW), np.int32)
    r, c = rc_of_lonlat(sel.geometry.x.values, sel.geometry.y.values)
    sdoy = sel[cols["start_doy"]].values.astype(int); edoy = sel[cols["end_doy"]].values.astype(int)
    order = np.argsort(-sel[cols["size"]].values)       # крупные пожары первыми забирают свои ячейки
    n_ok = 0
    for i in order:
        rr, cc = r[i], c[i]
        if not (0 <= rr < GH and 0 <= cc < GW):
            continue
        r0, r1, c0, c1 = max(rr - 300, 0), min(rr + 300, GH), max(cc - 300, 0), min(cc + 300, GW)   # ± 150 км
        sub = doy[r0:r1, c0:c1]; m = (sub >= sdoy[i] - 1) & (sub <= edoy[i] + 1) & (fid[r0:r1, c0:c1] == 0)
        # точка возгорания может попасть в соседнюю ячейку - ищем ближайшую подходящую в радиусе 2 ячеек
        lr, lc = rr - r0, cc - c0
        win = m[max(lr - 2, 0):lr + 3, max(lc - 2, 0):lc + 3]
        if not win.any():
            continue
        wr, wc = np.argwhere(win)[0]; lr, lc = max(lr - 2, 0) + wr, max(lc - 2, 0) + wc
        lab, _ = ndimage.label(m, structure=st8); comp = lab == lab[lr, lc]
        n = int(comp.sum())
        if n < 4:
            continue
        uid = int(y * 100000 + len(rows) + 1)
        fid[r0:r1, c0:c1][comp] = uid
        pr, pc = np.nonzero(comp); d = sub[comp]
        rows.append(dict(fire_id=uid, year=y, lon=sel.geometry.x.values[i], lat=sel.geometry.y.values[i], ign_r=rr, ign_c=cc,
                         start_doy=int(sdoy[i]), end_doy=int(edoy[i]), gfa_size_km2=float(sel[cols["size"]].values[i]),
                         gfa_duration=int(sel[cols["duration"]].values[i]),
                         gfa_speed=float(sel[cols["speed"]].values[i]) if "speed" in cols else np.nan,
                         gfa_direction=str(sel[cols["direction"]].values[i]) if "direction" in cols else "",
                         gfa_landcover=str(sel[cols["landcover"]].values[i]) if "landcover" in cols else "",
                         n_cells=n, area_km2=n * 0.25, doy_min=int(d.min()), doy_max=int(d.max()), n_days=int(d.max() - d.min() + 1),
                         rmin=r0 + pr.min(), rmax=r0 + pr.max(), cmin=c0 + pc.min(), cmax=c0 + pc.max()))
        n_ok += 1
    np.save(os.path.join(KZ, f"fire_id_{y}.npy"), fid)
    print(f"{y}: GFA fires in KZ {len(g)}, duration>=2d & size>=1km2: {len(sel)}, matched to raster: {n_ok}", flush=True)
ev = pd.DataFrame(rows)
ev["split"] = np.where(ev.year <= 2021, "train", np.where(ev.year == 2022, "val", "test"))
ev.to_csv(os.path.join(KZ, "events_kz.csv"), index=False)
print("\nfires:", len(ev), "| by split:", ev.split.value_counts().to_dict())
print("area km2 quantiles:", ev.area_km2.quantile([.1, .5, .9, .99]).round(1).to_dict(), "| n_days median:", ev.n_days.median())
print(ev.groupby("year").agg(fires=("fire_id", "size"), area=("area_km2", "sum")).round(0).to_string())
