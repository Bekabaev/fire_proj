"""Шаг 14 (расширение выборки). Срезы пожаров Казахстана (Global Fire Atlas) в формате FireSpread_Report_2:
сцена 96 × 96 ячеек 500 м EPSG:6933, «состояние пожара в конце суток d -> новое выгорание за 24/48/72 ч».

Состояние: ячейки этого пожара с датой выгорания ≤ d. Цель: ячейки этого пожара с d < дата ≤ d + 1/2/3.
valid: ячейка покрыта данными, ещё не сгорела в этом пожаре и не сгорела в (d, d+3] в другом пожаре (чужой рост не размечаем).
Каналы (15): burned_frac, af_count_24h, af_frp_24h, af_count_all (FIRMS, тип 0 = растительность), prior_burn (1 - сгорело ранее
в этом году или в прошлом году, 0.5 - два года назад), fbfm40, grass/crop/shrub_tree/wetland/nonburnable (доли WorldCover),
elev, slope_deg, ndvi (последний 16-сут композит MOD13A1 до начала пожара), dist_ign_km.
Срезы: d = первый … последний день горения пожара, не более 10 на пожар (равномерно). Годы срезов 2013–2024 (2012 - только прошлые гари).
t = конец суток d по местному времени (≈ 18 UTC); погода на срез добавляется шагом 15.
Год обрабатывается, когда для него готовы термоточки (шаг 10) и NDVI (шаг 12); готовые годы пропускаются.
Выход: data/ext_kz/slices/<год>.npz (X [S,15,96,96] f16, Y [S,3,96,96] u8, valid [S,96,96] bool), data/ext_kz/slices/idx_<год>.csv
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *
from rasterio.warp import transform as warp_transform

OD = os.path.join(KZ, "slices"); os.makedirs(OD, exist_ok=True)
CH = ["burned_frac", "af_count_24h", "af_frp_24h", "af_count_all", "prior_burn", "fbfm40", "grass_frac", "crop_frac",
      "shrub_tree_frac", "wetland_frac", "nonburnable_frac", "elev", "slope_deg", "ndvi", "dist_ign_km"]
HZ = (24, 48, 72); MAX_SL = 10
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv"))
lc = np.load(os.path.join(KZ, "lc.npz")); FR = lc["frac"]; FB = lc["fbfm40"]; COV = lc["covered"]
dem = np.load(os.path.join(KZ, "dem.npz")); ELEV = dem["elev"]; SLOPE = dem["slope"]


idx = []
for y in YEARS[1:]:
    fo = os.path.join(OD, f"{y}.npz"); fi = os.path.join(OD, f"idx_{y}.csv")
    E = ev[ev.year == y]
    need = [doy_path(y), os.path.join(KZ, f"ndvi_{y}.npz"), os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz")]
    if os.path.exists(fi):
        idx.append(pd.read_csv(fi)); continue
    if len(E) == 0 or not all(map(os.path.exists, need)):
        print(y, "not ready"); continue
    rows = []
    doy = np.asarray(load_doy(y)); fid = np.load(os.path.join(KZ, f"fire_id_{y}.npy"))
    p1 = np.asarray(load_doy(y - 1)) > 0 if os.path.exists(doy_path(y - 1)) else np.zeros_like(doy, bool)
    p2 = np.asarray(load_doy(y - 2)) > 0 if os.path.exists(doy_path(y - 2)) else np.zeros_like(doy, bool)
    nd = np.load(os.path.join(KZ, f"ndvi_{y}.npz")); NDV = np.pad(nd["ndvi"], ((0, 0), (0, 2), (0, 2)), constant_values=-32768); NDOY = nd["doy"]
    fa = pd.read_csv(os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz"), parse_dates=["time_utc"])
    fa = fa[fa["type"].fillna(0) == 0]
    px, py = warp_transform("EPSG:4326", CRS, fa.lon.tolist(), fa.lat.tolist())
    fa["gr"] = ((Y1 - np.array(py)) // CELL).astype(int); fa["gc"] = ((np.array(px) - X0) // CELL).astype(int)
    fa = fa[(fa.gr >= 0) & (fa.gr < GH) & (fa.gc >= 0) & (fa.gc < GW)]
    fa["doy_loc"] = (fa.time_utc + pd.Timedelta(hours=6)).dt.dayofyear      # местные сутки (UTC+6)
    fa_by_day = {d: g for d, g in fa.groupby("doy_loc")}
    Xs, Ys, Vs = [], [], []
    for e in E.itertuples():
        days = np.arange(e.doy_min, e.doy_max + 1)
        if len(days) > MAX_SL:
            days = np.unique(np.linspace(e.doy_min, e.doy_max, MAX_SL).round().astype(int))
        # NDVI: последний композит, закончившийся до начала пожара (начало + 16 сут ≤ первый день горения)
        kn = np.nonzero(NDOY + 16 <= e.doy_min)[0]
        bw = (slice(e.rmin, e.rmax + 1), slice(e.cmin, e.cmax + 1)); fb_ = fid[bw] == e.fire_id; db_ = doy[bw]
        for d in days:
            rr, cc = np.nonzero(fb_ & (db_ <= d) & (db_ > 0))
            if len(rr) == 0:
                continue
            cr, ccn = e.rmin + int(rr.mean()), e.cmin + int(cc.mean())
            r0 = int(np.clip(cr - N // 2, 0, GH - N)); c0 = int(np.clip(ccn - N // 2, 0, GW - N)); w = (slice(r0, r0 + N), slice(c0, c0 + N))
            D = doy[w]; F = fid[w]; mine = F == e.fire_id
            S = mine & (D <= d) & (D > 0)
            Yk = np.stack([(mine & (D > d) & (D <= d + k)) for k in (1, 2, 3)]).astype(np.uint8)
            other_new = (~mine) & (D > d) & (D <= d + 3)
            V = COV[w] & ~S & ~other_new
            prior = np.where(p1[w] | ((~mine) & (D > 0) & (D <= d)), 1.0, np.where(p2[w], 0.5, 0.0))
            afc = np.zeros((N, N), np.float32); afp = np.zeros((N, N), np.float32); afa = np.zeros((N, N), np.float32)
            for dd in range(e.doy_min - 1, d + 1):
                q = fa_by_day.get(dd)
                if q is None:
                    continue
                q = q[(q.gr >= r0) & (q.gr < r0 + N) & (q.gc >= c0) & (q.gc < c0 + N)]
                np.add.at(afa, (q.gr.values - r0, q.gc.values - c0), 1)
                if dd == d:
                    np.add.at(afc, (q.gr.values - r0, q.gc.values - c0), 1); np.add.at(afp, (q.gr.values - r0, q.gc.values - c0), q.frp.fillna(0).values)
            if len(kn):
                nv = NDV[kn[-1], r0 // 2:(r0 + N) // 2 + 1, c0 // 2:(c0 + N) // 2 + 1].astype(np.float32)
                nv = np.repeat(np.repeat(nv, 2, 0), 2, 1)[(r0 % 2):(r0 % 2) + N, (c0 % 2):(c0 % 2) + N]
                nv = np.where(nv < -20000, np.nan, nv * 1e-4)
            else:
                nv = np.full((N, N), np.nan, np.float32)
            gr, gc = np.mgrid[r0:r0 + N, c0:c0 + N]
            dist = np.hypot(gr - e.ign_r, gc - e.ign_c) * CELL / 1000
            L = dict(burned_frac=S.astype(np.float32), af_count_24h=afc, af_frp_24h=afp, af_count_all=afa, prior_burn=prior,
                     fbfm40=FB[w].astype(np.float32), grass_frac=FR[0][w] / 100, crop_frac=FR[1][w] / 100, shrub_tree_frac=FR[2][w] / 100,
                     wetland_frac=FR[3][w] / 100, nonburnable_frac=FR[4][w] / 100, elev=ELEV[w].astype(np.float32),
                     slope_deg=SLOPE[w].astype(np.float32), ndvi=nv, dist_ign_km=dist)
            Xs.append(np.stack([L[c] for c in CH]).astype(np.float16)); Ys.append(Yk); Vs.append(V)
            t = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=int(d) - 1, hours=18)
            row = dict(fire_id=e.fire_id, year=y, split=e.split, slice=len(Xs) - 1, doy=int(d), k_day=int(d - e.doy_min), t_utc=t,
                       r0=r0, c0=c0, lat=e.lat, lon=e.lon, fire_area_km2=e.area_km2, state_km2=float(S.sum() * 0.25),
                       af_24h=int(afc.sum()), valid_frac=float(V.mean()),
                       **{f"new_{h}h_km2": float(Yk[i].sum() * 0.25) for i, h in enumerate(HZ)},
                       **{f"new_{h}h_km2_total": float((fb_ & (db_ > d) & (db_ <= d + i + 1)).sum() * 0.25) for i, h in enumerate(HZ)},
                       area_burnable_frac=float(1 - FR[4][w].mean() / 100))
            rows.append(row)
    if Xs:
        np.savez_compressed(fo, X=np.stack(Xs), Y=np.stack(Ys), valid=np.stack(Vs))
        pd.DataFrame(rows).to_csv(fi, index=False); idx.append(pd.DataFrame(rows))
    print(f"{y}: fires {len(E)}, slices {len(Xs)}", flush=True)
if not idx:
    sys.exit()
si = pd.concat(idx, ignore_index=True); si.to_csv(os.path.join(KZ, "slices_base.csv"), index=False)
json.dump(dict(channels=CH, targets=[f"new_{h}h" for h in HZ], cell_m=CELL, size=N, crs=CRS, grid=dict(X0=X0, Y1=Y1, GW=GW, GH=GH)),
          open(os.path.join(KZ, "slices_channels.json"), "w"), indent=1)
print("slices:", len(si), "| fires:", si.fire_id.nunique(), "| by split:", si.groupby("split").agg(slices=("slice", "size"), fires=("fire_id", "nunique")).to_dict("index"))
for h in HZ:
    print(f"{h}h: share of slices with growth {(si[f'new_{h}h_km2'] > 0).mean():.2f}, median growth (cells) "
          f"{(si[f'new_{h}h_km2'] / 0.25).median():.0f}, in-scene share (area) {si[f'new_{h}h_km2'].sum() / si[f'new_{h}h_km2_total'].sum():.3f}")
