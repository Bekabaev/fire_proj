"""Шаг 22 (датасет v3). Субсуточные срезы пожаров Казахстана: «состояние на момент t -> новое выгорание за 12/24/48/72 ч».

Отличия от v2 (шаг 14):
- время прихода огня по ячейкам из шага 18 (VIIRS/MODIS, интерполяция, дата MCD64A1) вместо дня MCD64A1;
  моменты прогноза t - каждые 12 ч в 00 и 12 UTC (после ночного и дневного пролётов VIIRS), не более 12 на пожар;
- горизонт 12 ч добавлен (цели: new_12h, new_24h, new_48h, new_72h);
- метки включают ячейки-расширения по VIIRS (src=5) вне контура MCD64A1; карта источника метки `lsrc` сохраняется,
  поэтому вариант «только MCD64A1» = Y & (lsrc != 5), веса - по неопределённости источника (см. шаг 18);
- новые каналы: возраст выгорания, термоточки за 12 ч, лет с прошлой гари, аномалия NDVI, пик NDVI прошлого года,
  расстояние до поселений / дорог, плотность дорог (заместители тушения);
- крупные пожары (протяжённость > 80 ячеек = 40 км) дополнительно сохраняются сценой 96 × 96 ячеек 1 км (96 км), slices_1km/.
Каналы (23): см. CH. Термоточки и состояние - только то, что наблюдено к t (термоточки ≤ t).
valid: ячейка покрыта WorldCover, ещё не сгорела в этом пожаре к t, не сгорела в (t, t+72] в другом пожаре.
Выход: data/ext_kz/slices_v3/<год>_<k>.npz (X [S,23,96,96] f16, Y [S,4,96,96] u8, valid bool, lsrc u8),
       data/ext_kz/slices_v3/idx_<год>.csv, то же для slices_1km/, data/ext_kz/slices_v3_base.csv, slices_v3_channels.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *
from rasterio.warp import transform as warp_transform

OD = os.path.join(KZ, "slices_v3"); OD1 = os.path.join(KZ, "slices_1km"); os.makedirs(OD, exist_ok=True); os.makedirs(OD1, exist_ok=True)
CH = ["burned_frac", "burn_age_h", "af_count_12h", "af_frp_12h", "af_count_24h", "af_count_all", "prior_burn", "years_since_burn",
      "fbfm40", "grass_frac", "crop_frac", "shrub_tree_frac", "wetland_frac", "nonburnable_frac", "elev", "slope_deg",
      "ndvi", "ndvi_anom", "ndvi_peak_prev", "dist_ign_km", "dist_settlement_km", "dist_road_km", "road_density"]
HZ = (12, 24, 48, 72); MAX_SL = 12; CHUNK = 4000; BIG = 80
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv"))
lc = np.load(os.path.join(KZ, "lc.npz")); FR = lc["frac"]; FB = lc["fbfm40"]; COV = lc["covered"]
dem = np.load(os.path.join(KZ, "dem.npz")); ELEV = dem["elev"]; SLOPE = dem["slope"]
osm = np.load(os.path.join(KZ, "osm.npz")); DSET = osm["dist_settlement_km"]; DROAD = osm["dist_road_km"]; RDEN = osm["road_density"]
cl = np.load(os.path.join(KZ, "ndvi_clim.npz")); CLIM = np.pad(cl["ndvi"], ((0, 0), (0, 2), (0, 2)), constant_values=-32768); CLIM_DOY = list(cl["doy"])


def up2(a, r0, c0, n):
    """Поле сетки 1 км -> окно n × n сетки 500 м с началом (r0, c0)."""
    s = a[r0 // 2:(r0 + n) // 2 + 1, c0 // 2:(c0 + n) // 2 + 1].astype(np.float32)
    s = np.repeat(np.repeat(s, 2, 0), 2, 1)[(r0 % 2):(r0 % 2) + n, (c0 % 2):(c0 % 2) + n]
    return np.where(s < -20000, np.nan, s * 1e-4)


def coarsen(a, how="mean"):
    """[..., 2n, 2n] -> [..., n, n] блоками 2 × 2."""
    sh = a.shape[:-2] + (a.shape[-2] // 2, 2, a.shape[-1] // 2, 2)
    b = a.reshape(sh)
    if how == "sum":
        return b.sum((-3, -1))
    if how == "first":
        return b[..., 0, :, 0]
    if how == "nanmean":
        with np.errstate(all="ignore"):
            return np.nanmean(b, (-3, -1))
    return b.mean((-3, -1))


class Writer:
    def __init__(self, od, y):
        self.od, self.y, self.k, self.buf, self.rows = od, y, 0, ([], [], [], []), []

    def add(self, X, Y, V, L, row):
        row = dict(row, file=f"{self.y}_{self.k}.npz", slice=len(self.buf[0])); self.rows.append(row)
        for b, a in zip(self.buf, (X, Y, V, L)):
            b.append(a)
        if len(self.buf[0]) >= CHUNK:
            self.flush()

    def flush(self):
        if self.buf[0]:
            np.savez_compressed(os.path.join(self.od, f"{self.y}_{self.k}.npz"), X=np.stack(self.buf[0]), Y=np.stack(self.buf[1]),
                                valid=np.stack(self.buf[2]), lsrc=np.stack(self.buf[3]))
            self.k += 1; self.buf = ([], [], [], [])

    def close(self):
        self.flush()
        pd.DataFrame(self.rows).to_csv(os.path.join(self.od, f"idx_{self.y}.csv"), index=False)
        return pd.DataFrame(self.rows)


def scene(y, e, t, r0, c0, n, ctx):
    """Каналы, цели, valid, lsrc для окна n × n сетки 500 м с началом (r0, c0)."""
    AR, FID, SRC, fa, p1, p2, LB, NDV, NPK, kn, NDOY = ctx
    w = (slice(r0, r0 + n), slice(c0, c0 + n))
    A = AR[w]; F = FID[w]; mine = F == e.fire_id
    S = mine & (A <= t)
    Yk = np.stack([mine & (A > t) & (A <= t + h) for h in HZ]).astype(np.uint8)
    other_new = (~mine) & (F > 0) & (A > t) & (A <= t + 72)
    V = COV[w] & ~S & ~other_new
    L = np.where(mine, SRC[w], 0).astype(np.uint8)
    age = np.where(S, np.minimum(t - A, 240), 0).astype(np.float32)
    prior = np.where(p1[w] | ((~mine) & (F > 0) & (A <= t)), 1.0, np.where(p2[w], 0.5, 0.0))
    lb = LB[w]; ysb = np.where(lb > 0, y - lb, 13).astype(np.float32)
    q = fa[(fa.gr >= r0) & (fa.gr < r0 + n) & (fa.gc >= c0) & (fa.gc < c0 + n) & (fa.t_h <= t) & (fa.t_h > ctx_t0(e))]
    ii, jj = q.gr.values - r0, q.gc.values - c0
    afa = np.zeros((n, n), np.float32); np.add.at(afa, (ii, jj), 1)
    m12 = q.t_h.values > t - 12; m24 = q.t_h.values > t - 24
    af12 = np.zeros((n, n), np.float32); np.add.at(af12, (ii[m12], jj[m12]), 1)
    fp12 = np.zeros((n, n), np.float32); np.add.at(fp12, (ii[m12], jj[m12]), q.frp.fillna(0).values[m12])
    af24 = np.zeros((n, n), np.float32); np.add.at(af24, (ii[m24], jj[m24]), 1)
    if len(kn):
        nv = up2(NDV[kn[-1]], r0, c0, n); an = nv - up2(CLIM[CLIM_DOY.index(NDOY[kn[-1]])], r0, c0, n)
    else:
        nv = an = np.full((n, n), np.nan, np.float32)
    pk = up2(NPK, r0, c0, n) if NPK is not None else np.full((n, n), np.nan, np.float32)
    gr, gc = np.mgrid[r0:r0 + n, c0:c0 + n]
    dist = np.hypot(gr - e.ign_r, gc - e.ign_c) * CELL / 1000
    Lr = dict(burned_frac=S.astype(np.float32), burn_age_h=age, af_count_12h=af12, af_frp_12h=fp12, af_count_24h=af24, af_count_all=afa,
              prior_burn=prior, years_since_burn=ysb, fbfm40=FB[w].astype(np.float32), grass_frac=FR[0][w] / 100, crop_frac=FR[1][w] / 100,
              shrub_tree_frac=FR[2][w] / 100, wetland_frac=FR[3][w] / 100, nonburnable_frac=FR[4][w] / 100,
              elev=ELEV[w].astype(np.float32), slope_deg=SLOPE[w].astype(np.float32), ndvi=nv, ndvi_anom=an, ndvi_peak_prev=pk,
              dist_ign_km=dist, dist_settlement_km=DSET[w].astype(np.float32), dist_road_km=DROAD[w].astype(np.float32),
              road_density=RDEN[w].astype(np.float32))
    return np.stack([Lr[c] for c in CH]).astype(np.float32), Yk, V, L


def ctx_t0(e):
    return (e.doy_min - 2) * 24.0


idx, idx1 = [], []
YRS = [int(v) for v in os.environ["ONLY_YEARS"].split(",")] if os.environ.get("ONLY_YEARS") else YEARS[1:]
for y in YRS:
    fi = os.path.join(OD, f"idx_{y}.csv")
    if os.path.exists(fi):
        idx.append(pd.read_csv(fi)); f1 = os.path.join(OD1, f"idx_{y}.csv")
        if os.path.exists(f1):
            idx1.append(pd.read_csv(f1))
        continue
    E = ev[ev.year == y]
    a = np.load(os.path.join(KZ, "arrival", f"arr_{y}.npz"))
    AR = np.full((GH, GW), np.nan, np.float32); FID = np.zeros((GH, GW), np.int32); SRC = np.zeros((GH, GW), np.uint8)
    AR[a["r"], a["c"]] = a["arr_h"]; FID[a["r"], a["c"]] = a["fire_id"]; SRC[a["r"], a["c"]] = a["src"]
    cells = pd.DataFrame(dict(f=a["fire_id"], r=a["r"], c=a["c"], t=a["arr_h"]))
    ext = cells.groupby("f").agg(rmin=("r", "min"), rmax=("r", "max"), cmin=("c", "min"), cmax=("c", "max"), t0=("t", "min"), t1=("t", "max"))
    p1 = np.asarray(load_doy(y - 1)) > 0 if os.path.exists(doy_path(y - 1)) else np.zeros((GH, GW), bool)
    p2 = np.asarray(load_doy(y - 2)) > 0 if os.path.exists(doy_path(y - 2)) else np.zeros((GH, GW), bool)
    LB = np.load(os.path.join(KZ, f"last_burn_{y}.npy"))
    nd = np.load(os.path.join(KZ, f"ndvi_{y}.npz")); NDV = np.pad(nd["ndvi"], ((0, 0), (0, 2), (0, 2)), constant_values=-32768); NDOY = nd["doy"]
    fpk = os.path.join(KZ, f"ndvi_peak_{y - 1}.npy"); NPK = np.pad(np.load(fpk), ((0, 2), (0, 2)), constant_values=-32768) if os.path.exists(fpk) else None
    fa = pd.read_csv(os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz"), parse_dates=["time_utc"])
    fa = fa[fa["type"].fillna(0) == 0]
    px, py = warp_transform("EPSG:4326", CRS, fa.lon.tolist(), fa.lat.tolist())
    fa["gr"] = ((Y1 - np.array(py)) // CELL).astype(int); fa["gc"] = ((np.array(px) - X0) // CELL).astype(int)
    fa["t_h"] = (fa.time_utc - pd.Timestamp(year=y, month=1, day=1)).dt.total_seconds() / 3600
    fa = fa[(fa.gr >= 0) & (fa.gr < GH) & (fa.gc >= 0) & (fa.gc < GW)][["gr", "gc", "t_h", "frp"]].sort_values("gr").reset_index(drop=True)
    W, W1 = Writer(OD, y), Writer(OD1, y)
    for e in E.itertuples():
        if e.fire_id not in ext.index:
            continue
        x = ext.loc[e.fire_id]
        kn = np.nonzero(NDOY + 16 <= e.doy_min)[0]
        ctx = (AR, FID, SRC, fa[(fa.gr >= x.rmin - 200) & (fa.gr <= x.rmax + 200)], p1, p2, LB, NDV, NPK, kn, NDOY)
        ts = np.arange(np.floor(x.t0 / 12) * 12 + 12, x.t1 + 1e-3, 12.0)     # первый срок 00/12 UTC после первого прихода огня
        if len(ts) == 0:
            continue
        if len(ts) > MAX_SL:
            ts = ts[np.unique(np.linspace(0, len(ts) - 1, MAX_SL).round().astype(int))]
        big = max(x.rmax - x.rmin, x.cmax - x.cmin) > BIG
        mine_all = cells[cells.f == e.fire_id]
        for t in ts:
            st = mine_all[mine_all.t <= t]
            if st.empty:
                continue
            cr, ccn = int(st.r.mean()), int(st.c.mean())
            r0 = int(np.clip(cr - N // 2, 0, GH - N)); c0 = int(np.clip(ccn - N // 2, 0, GW - N))
            X, Yk, V, L = scene(y, e, t, r0, c0, N, ctx)
            fut = mine_all[(mine_all.t > t)]
            tt = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(hours=float(t))
            row = dict(fire_id=e.fire_id, year=y, split=e.split, t_utc=tt, t_h=float(t), hour_utc=tt.hour,
                       k_12h=int((t - ts[0]) // 12), r0=r0, c0=c0, lat=e.lat, lon=e.lon, fire_area_km2=e.area_km2,
                       fire_area_ext_km2=len(mine_all) * 0.25, state_km2=float(X[0].sum() * 0.25), af_12h=int(X[2].sum()),
                       valid_frac=float(V.mean()), big_fire=bool(big),
                       state_src_af=float((L[X[0] > 0] <= 2).mean()) if X[0].any() else np.nan)
            for i, h in enumerate(HZ):
                yk = Yk[i].astype(bool); tot = int(((fut.t <= t + h)).sum())
                row[f"new_{h}h_km2"] = float(yk.sum() * 0.25); row[f"new_{h}h_km2_total"] = tot * 0.25
                row[f"new_{h}h_in_scene"] = float(yk.sum() / tot) if tot else np.nan
                row[f"new_{h}h_km2_mcd"] = float((yk & (L != 5)).sum() * 0.25)
                ls = L[yk]
                row[f"new_{h}h_src_af"] = float((ls <= 2).mean()) if len(ls) else np.nan        # доля целей с временем по термоточкам
                row[f"new_{h}h_src_day"] = float((ls == 4).mean()) if len(ls) else np.nan       # доля целей только с датой MCD64A1
            W.add(X.astype(np.float16), Yk, V, L, row)
            if big:
                n2 = 2 * N
                R0 = int(np.clip(cr - n2 // 2, 0, GH - n2)) // 2 * 2; C0 = int(np.clip(ccn - n2 // 2, 0, GW - n2)) // 2 * 2
                X2, Y2, V2, L2 = scene(y, e, t, R0, C0, n2, ctx)
                cnt = [CH.index(c) for c in ("af_count_12h", "af_frp_12h", "af_count_24h", "af_count_all")]
                Xc = coarsen(X2, "nanmean"); Xc[cnt] = coarsen(X2[cnt], "sum"); Xc[CH.index("fbfm40")] = coarsen(X2[CH.index("fbfm40")], "first")
                Yc = (coarsen(Y2.astype(np.float32)) * 4).astype(np.uint8)       # число ячеек 500 м (0–4) в ячейке 1 км
                Vc = coarsen(V2.astype(np.float32)) >= 0.5
                Lc = coarsen(L2, "first").astype(np.uint8)
                r1 = dict(row, r0=R0, c0=C0, cell_m=1000, valid_frac=float(Vc.mean()))
                for i, h in enumerate(HZ):
                    tot = row[f"new_{h}h_km2_total"]; ins = float(Y2[i].sum() * 0.25)
                    r1[f"new_{h}h_km2"] = ins; r1[f"new_{h}h_in_scene"] = ins / tot if tot else np.nan
                W1.add(Xc.astype(np.float16), Yc, Vc, Lc, r1)
    d = W.close(); idx.append(d); d1 = W1.close() if W1.rows else None
    if d1 is not None:
        idx1.append(d1)
    print(f"{y}: fires {len(E)}, slices {len(d)}, 1 km slices {0 if d1 is None else len(d1)}", flush=True)

si = pd.concat(idx, ignore_index=True); si.to_csv(os.path.join(KZ, "slices_v3_base.csv"), index=False)
if idx1:
    pd.concat(idx1, ignore_index=True).to_csv(os.path.join(KZ, "slices_1km_base.csv"), index=False)
json.dump(dict(channels=CH, targets=[f"new_{h}h" for h in HZ], cell_m=CELL, size=N, crs=CRS, grid=dict(X0=X0, Y1=Y1, GW=GW, GH=GH),
               lsrc={"1": "VIIRS", "2": "MODIS", "3": "interpolated", "4": "MCD64A1 day only", "5": "VIIRS outside MCD64A1"},
               lsrc_uncertainty_h={"1": 6, "2": 12, "3": 12, "4": 24, "5": 6},
               slices_1km="крупные пожары: 96 x 96 ячеек 1 км; Y = число ячеек 500 м (0-4), счётчики термоточек - сумма"),
          open(os.path.join(KZ, "slices_v3_channels.json"), "w"), indent=1, ensure_ascii=False)
print("slices:", len(si), "| fires:", si.fire_id.nunique(), "|", si.groupby("split").agg(slices=("slice", "size"), fires=("fire_id", "nunique")).to_dict("index"))
for h in HZ:
    print(f"{h}h: share with growth {(si[f'new_{h}h_km2'] > 0).mean():.2f}, in-scene area share {si[f'new_{h}h_km2'].sum() / si[f'new_{h}h_km2_total'].sum():.3f}, "
          f"targets (area-weighted) timed by AF {np.nansum(si[f'new_{h}h_src_af'] * si[f'new_{h}h_km2']) / si[f'new_{h}h_km2'].sum():.2f}, "
          f"by day only {np.nansum(si[f'new_{h}h_src_day'] * si[f'new_{h}h_km2']) / si[f'new_{h}h_km2'].sum():.2f}")
