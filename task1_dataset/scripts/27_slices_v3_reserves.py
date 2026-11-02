"""Шаг 27 (датасет v3). Срезы экспертных пожаров заповедников в формате v3 (как шаг 22) - для проверки моделей, обученных на v3.

Метки берутся не из GFA / MCD64A1, а из экспертного контура (куб 30 м, шаг 3) и времени прихода огня внутри него (шаг 7):
- контур и время переносятся на национальную сетку 500 м: доля контура в ячейке, медиана времени прихода, преобладающий источник;
- ячейка «этого пожара» - доля контура ≥ 0,5; ячейки с долей (0; 0,5) - неоднозначные, исключаются из valid;
- valid только внутри окна куба (вне его неизвестно, горело ли): покрытие кубом > 0,9, без зон ignore;
- ячейки GFA, совпадающие с контуром или горевшие в окне события внутри куба, заменяются экспертной разметкой;
  остальные пожары GFA остаются (каналы prior_burn и исключение «сгорело в другом пожаре» - как в шаге 22).
Время прихода - из data/arrival_img/ (уточнено по промежуточным снимкам, шаг 30), если файл есть, иначе data/arrival/ (шаг 7).
Источник метки lsrc в кодах v3: 1 - время по термоточкам FIRMS, 4 - только дата MCD64A1, 3 - экстраполяция или уточнение по снимку;
веса - через v3_loader.label_weights, как у основной выборки.
Каналы (23), сроки t (00/12 UTC, шаг 12 ч), цели 12/24/48/72 ч, крупные пожары > 40 км - дополнительно сцена 1 км: как в шаге 22.
Пересечение с обучением: в индексе gfa_fire_ids / gfa_split - пожары GFA, совпадающие с контуром; для честного теста их
нужно исключать из обучающих срезов v3 (большинство пожаров заповедников 2014–2021 попадают в train GFA).
Выход: data/ext_kz/slices_v3_reserves/<event>.npz (+ slices_1km_reserves/), data/ext_kz/slices_v3_reserves_base.csv
"""
import os, sys, json
import numpy as np, pandas as pd
from rasterio.warp import reproject, Resampling, transform as warp_transform, transform_bounds
from rasterio.transform import Affine, from_origin
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OD = os.path.join(KZ, "slices_v3_reserves"); OD1 = os.path.join(KZ, "slices_1km_reserves")
os.makedirs(OD, exist_ok=True); os.makedirs(OD1, exist_ok=True)
META = json.load(open(os.path.join(KZ, "slices_v3_channels.json"), encoding="utf8"))
CH = META["channels"]; HZ = (12, 24, 48, 72); BIG = 80; MAX_SL = 80; EXP_ID = -1
SRC_MAP = {1: 1, 2: 4, 3: 3, 6: 3}   # источник времени шага 7 (6 - уточнено по снимку, шаг 30) -> код lsrc v3
LONG = {"AltynDala_2016_07_07", "AltynDala_2017_06_06", "AltynDala_2017_06_08", "Korgalzhyn_2014_08_08", "Korgalzhyn_2017_06_30"}
lc = np.load(os.path.join(KZ, "lc.npz")); FR = lc["frac"]; FB = lc["fbfm40"]; COV = lc["covered"]
dem = np.load(os.path.join(KZ, "dem.npz")); ELEV = dem["elev"]; SLOPE = dem["slope"]
osm = np.load(os.path.join(KZ, "osm.npz")); DSET = osm["dist_settlement_km"]; DROAD = osm["dist_road_km"]; RDEN = osm["road_density"]
cl = np.load(os.path.join(KZ, "ndvi_clim.npz")); CLIM = np.pad(cl["ndvi"], ((0, 0), (0, 2), (0, 2)), constant_values=-32768); CLIM_DOY = list(cl["doy"])
gsplit = pd.read_csv(os.path.join(KZ, "events_kz.csv")).set_index("fire_id").split.to_dict()


def up2(a, r0, c0, n):
    s = a[r0 // 2:(r0 + n) // 2 + 1, c0 // 2:(c0 + n) // 2 + 1].astype(np.float32)
    s = np.repeat(np.repeat(s, 2, 0), 2, 1)[(r0 % 2):(r0 % 2) + n, (c0 % 2):(c0 % 2) + n]
    return np.where(s < -20000, np.nan, s * 1e-4)


def coarsen(a, how="mean"):
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


def scene(y, t, t_from, r0, c0, n, ctx):
    """Как scene() шага 22, но «этот пожар» - экспертный (FID == EXP_ID), valid дополнительно ограничен окном куба (OK)."""
    AR, FID, SRC, OK, fa, p1, p2, LB, NDV, NPK, kn, NDOY, ign_rc = ctx
    w = (slice(r0, r0 + n), slice(c0, c0 + n))
    A = AR[w]; F = FID[w]; mine = F == EXP_ID
    S = mine & (A <= t)
    Yk = np.stack([mine & (A > t) & (A <= t + h) for h in HZ]).astype(np.uint8)
    other = (F != 0) & ~mine
    other_new = other & (A > t) & (A <= t + 72)
    V = COV[w] & OK[w] & ~S & ~other_new
    L = np.where(mine, SRC[w], 0).astype(np.uint8)
    age = np.where(S, np.minimum(t - A, 240), 0).astype(np.float32)
    prior = np.where(p1[w] | (other & (A <= t)), 1.0, np.where(p2[w], 0.5, 0.0))
    lb = LB[w]; ysb = np.where(lb > 0, y - lb, 13).astype(np.float32)
    q = fa[(fa.gr >= r0) & (fa.gr < r0 + n) & (fa.gc >= c0) & (fa.gc < c0 + n) & (fa.t_h <= t) & (fa.t_h > t_from)]
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
    dist = np.hypot(gr - ign_rc[0], gc - ign_rc[1]) * CELL / 1000
    Lr = dict(burned_frac=S.astype(np.float32), burn_age_h=age, af_count_12h=af12, af_frp_12h=fp12, af_count_24h=af24, af_count_all=afa,
              prior_burn=prior, years_since_burn=ysb, fbfm40=FB[w].astype(np.float32), grass_frac=FR[0][w] / 100, crop_frac=FR[1][w] / 100,
              shrub_tree_frac=FR[2][w] / 100, wetland_frac=FR[3][w] / 100, nonburnable_frac=FR[4][w] / 100,
              elev=ELEV[w].astype(np.float32), slope_deg=SLOPE[w].astype(np.float32), ndvi=nv, ndvi_anom=an, ndvi_peak_prev=pk,
              dist_ign_km=dist, dist_settlement_km=DSET[w].astype(np.float32), dist_road_km=DROAD[w].astype(np.float32),
              road_density=RDEN[w].astype(np.float32))
    return np.stack([Lr[c] for c in CH]).astype(np.float32), Yk, V, L


def save(od, name, X, Y, V, L):
    np.savez_compressed(os.path.join(od, name + ".npz"), X=np.stack(X), Y=np.stack(Y), valid=np.stack(V), lsrc=np.stack(L))


ev = load_events(); idx, idx1 = [], []
OK = np.zeros((GH, GW), bool)
for y, E in ev.groupby(ev.ign_time_utc.dt.year):
    a = np.load(os.path.join(KZ, "arrival", f"arr_{y}.npz"))
    AR = np.full((GH, GW), np.nan, np.float32); FID = np.zeros((GH, GW), np.int32); SRC = np.zeros((GH, GW), np.uint8)
    AR[a["r"], a["c"]] = a["arr_h"]; FID[a["r"], a["c"]] = a["fire_id"]; SRC[a["r"], a["c"]] = a["src"]
    p1 = np.asarray(load_doy(y - 1)) > 0
    p2 = np.asarray(load_doy(y - 2)) > 0
    LB = np.load(os.path.join(KZ, f"last_burn_{y}.npy"))
    nd = np.load(os.path.join(KZ, f"ndvi_{y}.npz")); NDV = np.pad(nd["ndvi"], ((0, 0), (0, 2), (0, 2)), constant_values=-32768); NDOY = nd["doy"]
    fpk = os.path.join(KZ, f"ndvi_peak_{y - 1}.npy"); NPK = np.pad(np.load(fpk), ((0, 2), (0, 2)), constant_values=-32768) if os.path.exists(fpk) else None
    fa = pd.read_csv(os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz"), parse_dates=["time_utc"])
    fa = fa[fa["type"].fillna(0) == 0]
    px, py = warp_transform("EPSG:4326", CRS, fa.lon.tolist(), fa.lat.tolist())
    fa["gr"] = ((Y1 - np.array(py)) // CELL).astype(int); fa["gc"] = ((np.array(px) - X0) // CELL).astype(int)
    fa["t_h"] = (fa.time_utc - pd.Timestamp(year=y, month=1, day=1)).dt.total_seconds() / 3600
    fa = fa[(fa.gr >= 0) & (fa.gr < GH) & (fa.gc >= 0) & (fa.gc < GW)][["gr", "gc", "t_h", "frp"]].reset_index(drop=True)
    for e in E.itertuples():
        meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json"))); H, W = meta["H"], meta["W"]; tr = Affine(*meta["transform"])
        cb = np.load(os.path.join(DATA, "cubes", e.event + ".npz")); lab = cb["label"].astype(bool)
        fi = os.path.join(DATA, "arrival_img", e.event + ".npz")       # время, уточнённое по снимкам (шаг 30), если есть
        ar = np.load(fi if os.path.exists(fi) else os.path.join(DATA, "arrival", e.event + ".npz"))
        if np.isfinite(ar["arrival_h"][lab]).sum() <= 1:        # как в шаге 8: динамика неизвестна
            print(f"{e.event:24s} SKIP: arrival time unknown", flush=True); continue
        t_ign = (e.ign_time_utc - pd.Timestamp(year=y, month=1, day=1)).total_seconds() / 3600
        # окно национальной сетки, покрывающее куб
        bx = transform_bounds(meta["crs"], CRS, tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f, densify_pts=21)
        c0 = int((bx[0] - X0) // CELL); c1 = int((bx[2] - X0) // CELL) + 1; r0 = int((Y1 - bx[3]) // CELL); r1 = int((Y1 - bx[1]) // CELL) + 1
        gt = from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL); sh = (r1 - r0, c1 - c0)

        def G(arr, rs, nod=None):
            out = np.full(sh, np.nan if nod is not None else 0, np.float32)
            reproject(arr.astype(np.float32), out, src_transform=tr, src_crs=meta["crs"], dst_transform=gt, dst_crs=CRS, resampling=rs,
                      src_nodata=nod, dst_nodata=nod)
            return out
        fr = G(lab, Resampling.average); cov = G(np.ones((H, W)), Resampling.average); ig = G(cb["ignore"] > 0, Resampling.average)
        ta = G(np.where(lab, ar["arrival_h"], np.nan), Resampling.med, np.nan) + t_ign
        sm = G(np.where(lab, ar["arrival_src"], 0), Resampling.mode, 0)
        mine = (fr >= 0.5) & np.isfinite(ta)
        amb = (fr > 0) & ~mine
        t_end = np.nanmax(ta[mine])
        win = (slice(r0, r1), slice(c0, c1))
        saved = (AR[win].copy(), FID[win].copy(), SRC[win].copy())
        # пожары GFA, совпадающие с контуром (для учёта пересечения с обучением), затем замена разметки внутри куба
        gf = FID[win]; ov = pd.Series(gf[(fr > 0) & (gf > 0)]).value_counts()
        gids = [int(k) for k, v in ov.items() if v >= 2]
        in_cube = cov > 0.9
        drop = (fr > 0) | (in_cube & (FID[win] > 0) & (AR[win] >= t_ign - 48) & (AR[win] <= t_end + 48))
        AR[win][drop] = np.nan; FID[win][drop] = 0; SRC[win][drop] = 0
        AR[win][mine] = ta[mine]; FID[win][mine] = EXP_ID
        SRC[win][mine] = np.vectorize(lambda s: SRC_MAP.get(int(s), 3))(sm[mine]).astype(np.uint8) if mine.any() else 0
        OK[win] = in_cube & (ig <= 0.5) & ~amb
        ign_rc = rc_of_lonlat(e.ign_lon, e.ign_lat); ign_rc = (int(ign_rc[0][0]), int(ign_rc[1][0]))
        doy_ign = e.ign_time_utc.dayofyear
        kn = np.nonzero(NDOY + 16 <= doy_ign)[0]
        rr, cc = np.nonzero(mine); cells = pd.DataFrame(dict(r=rr + r0, c=cc + c0, t=ta[mine]))
        ctx = (AR, FID, SRC, OK, fa[(fa.gr >= r0 - 150) & (fa.gr <= r1 + 150) & (fa.gc >= c0 - 150) & (fa.gc <= c1 + 150)],
               p1, p2, LB, NDV, NPK, kn, NDOY, ign_rc)
        t0, t1 = cells.t.min(), cells.t.max()
        ts = np.arange(np.floor(t0 / 12) * 12 + 12, t1 + 1e-3, 12.0)
        if len(ts) > MAX_SL:
            ts = ts[np.unique(np.linspace(0, len(ts) - 1, MAX_SL).round().astype(int))]
        big = max(rr.max() - rr.min(), cc.max() - cc.min()) > BIG
        B, B1 = ([], [], [], []), ([], [], [], [])
        for t in ts:
            st = cells[cells.t <= t]
            if st.empty:
                continue
            cr, ccn = int(st.r.mean()), int(st.c.mean())
            s0 = int(np.clip(cr - N // 2, 0, GH - N)); q0 = int(np.clip(ccn - N // 2, 0, GW - N))
            X, Yk, V, L = scene(y, t, t_ign - 48, s0, q0, N, ctx)
            fut = cells[cells.t > t]
            tt = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(hours=float(t))
            row = dict(event=e.event, site=e.site, year=y, split_time=e.split_time, fire_id=EXP_ID, t_utc=tt, t_h=float(t),
                       t_since_ign_h=float(t - t_ign), hour_utc=tt.hour, k_12h=int((t - ts[0]) // 12), r0=s0, c0=q0,
                       lat=e.ign_lat, lon=e.ign_lon, fire_area_km2=e.area_ha / 100, fire_area_ext_km2=len(cells) * 0.25,
                       state_km2=float(X[0].sum() * 0.25), af_12h=int(X[2].sum()), valid_frac=float(V.mean()), big_fire=bool(big),
                       flag_long=e.event in LONG, duration_ok=bool(e.duration_ok),
                       gfa_fire_ids=" ".join(map(str, gids)), gfa_split=" ".join(sorted({gsplit.get(g, "?") for g in gids})),
                       file=e.event + ".npz", slice=len(B[0]),
                       state_src_af=float((L[X[0] > 0] == 1).mean()) if X[0].any() else np.nan)
            for i, h in enumerate(HZ):
                yk = Yk[i].astype(bool); tot = int((fut.t <= t + h).sum())
                row[f"new_{h}h_km2"] = float(yk.sum() * 0.25); row[f"new_{h}h_km2_total"] = tot * 0.25
                row[f"new_{h}h_in_scene"] = float(yk.sum() / tot) if tot else np.nan
                row[f"new_{h}h_km2_mcd"] = row[f"new_{h}h_km2"]
                ls = L[yk]
                row[f"new_{h}h_src_af"] = float((ls == 1).mean()) if len(ls) else np.nan
                row[f"new_{h}h_src_day"] = float((ls == 4).mean()) if len(ls) else np.nan
            for b, v in zip(B, (X.astype(np.float16), Yk, V, L)):
                b.append(v)
            idx.append(row)
            if big:
                n2 = 2 * N
                R0 = int(np.clip(cr - n2 // 2, 0, GH - n2)) // 2 * 2; C0 = int(np.clip(ccn - n2 // 2, 0, GW - n2)) // 2 * 2
                X2, Y2, V2, L2 = scene(y, t, t_ign - 48, R0, C0, n2, ctx)
                cnt = [CH.index(c) for c in ("af_count_12h", "af_frp_12h", "af_count_24h", "af_count_all")]
                Xc = coarsen(X2, "nanmean"); Xc[cnt] = coarsen(X2[cnt], "sum"); Xc[CH.index("fbfm40")] = coarsen(X2[CH.index("fbfm40")], "first")
                r1d = dict(row, r0=R0, c0=C0, cell_m=1000, valid_frac=float((coarsen(V2.astype(np.float32)) >= 0.5).mean()), slice=len(B1[0]))
                for i, h in enumerate(HZ):
                    tot = row[f"new_{h}h_km2_total"]; ins = float(Y2[i].sum() * 0.25)
                    r1d[f"new_{h}h_km2"] = ins; r1d[f"new_{h}h_in_scene"] = ins / tot if tot else np.nan
                for b, v in zip(B1, (Xc.astype(np.float16), (coarsen(Y2.astype(np.float32)) * 4).astype(np.uint8),
                                     coarsen(V2.astype(np.float32)) >= 0.5, coarsen(L2, "first").astype(np.uint8))):
                    b.append(v)
                idx1.append(r1d)
        AR[win], FID[win], SRC[win] = saved; OK[win] = False
        if B[0]:
            save(OD, e.event, *B)
        if B1[0]:
            save(OD1, e.event, *B1)
        sub = [r for r in idx if r["event"] == e.event]
        print(f"{e.event:24s} cells {mine.sum():5d} amb {amb.sum():4d} slices {len(B[0]):3d} (1 km {len(B1[0]):2d}) "
              f"t_end {t_end - t_ign:6.1f} h  gfa {gids[:4]} {sub[0]['gfa_split'] if sub else ''}  "
              f"new24 km2 {[round(r['new_24h_km2'], 1) for r in sub][:6]}", flush=True)

si = pd.DataFrame(idx); si.to_csv(os.path.join(KZ, "slices_v3_reserves_base.csv"), index=False)
if idx1:
    pd.DataFrame(idx1).to_csv(os.path.join(KZ, "slices_1km_reserves_base.csv"), index=False)
print("\nslices:", len(si), "| events:", si.event.nunique(), "| 1 km slices:", len(idx1))
print(si.groupby("split_time").agg(slices=("slice", "size"), events=("event", "nunique")).to_string())
print("events overlapping GFA train fires:", si[si.gfa_split.str.contains("train", na=False)].event.nunique(),
      "| no GFA match:", si[si.gfa_fire_ids.fillna("") == ""].event.nunique())
for h in HZ:
    q = si[si[f"new_{h}h_km2_total"] > 0]
    print(f"{h}h: share with growth {(si[f'new_{h}h_km2'] > 0).mean():.2f}, in-scene area share {q[f'new_{h}h_km2'].sum() / q[f'new_{h}h_km2_total'].sum():.3f}, "
          f"targets timed by FIRMS {np.nansum(si[f'new_{h}h_src_af'] * si[f'new_{h}h_km2']) / si[f'new_{h}h_km2'].sum():.2f}, "
          f"by MCD64A1 day {np.nansum(si[f'new_{h}h_src_day'] * si[f'new_{h}h_km2']) / si[f'new_{h}h_km2'].sum():.2f}")
