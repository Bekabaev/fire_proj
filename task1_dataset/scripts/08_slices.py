"""Шаг 8. Срезы «состояние пожара на момент t -> новое выгорание через 24/48/72 ч» (формат, как в FireSpread_Report_2:
сетка 500 м в EPSG:6933, сцена 96 × 96 ячеек = 48 × 48 км, ячейка 0.25 км²; погода по срезу; FBFM40-топливо).

t_k = возгорание + 24·k ч (k = 0, 1, ...), пока t_k < последнего времени прихода огня.
Состояние: arrival ≤ t (доля ячейки) + термоточки FIRMS за (t − 24 ч, t]. Цель: t < arrival ≤ t + h, h = 24/48/72.
Центр сцены - центроид состояния на t (без утечки будущего), привязан к глобальной сетке 500 м.
valid: ячейка внутри окна куба (там известно «сгорело / не сгорело в этом пожаре»), без зон ignore.
Топливо FBFM40 (Scott & Burgan 2005) - приближённое сопоставление из ESA WorldCover (LANDFIRE для Казахстана нет).
Погода по срезу: ERA5 (все события) и архив GFS (Open-Meteo Historical Forecast, gfs_seamless; доступен с 2021 г.).
Выход: data/slices/<event>.npz (X [S,C,96,96] float16, Y [S,3,96,96] float16, valid [S,96,96]),
       data/slices_full/<event>.npz (те же каналы на всей сетке события 500 м + transform; сцена = окно [r0:r0+96, c0:c0+96]),
       data/slices_index.csv (по срезу: время, площади состояния/прироста, погода), data/slices_channels.json
"""
import os, sys, json, time
import numpy as np, pandas as pd, requests
from rasterio.warp import reproject, Resampling, transform as warp_transform
from rasterio.transform import Affine, from_origin
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from fwi import run as fwi_run

OD = os.path.join(DATA, "slices"); os.makedirs(OD, exist_ok=True)
FD = os.path.join(DATA, "slices_full"); os.makedirs(FD, exist_ok=True)   # те же срезы на полной сетке события (крупные пожары > 48 км)
GD = os.path.join(DATA, "weather_gfs"); os.makedirs(GD, exist_ok=True)
CELL, N = 500.0, 96
DST_CRS = "EPSG:6933"
FBFM40 = {10: 161, 20: 121, 30: 102, 40: 101, 50: 91, 60: 99, 70: 92, 80: 98, 90: 104, 95: 98, 100: 99}  # WorldCover -> FBFM40
BURNABLE = {10, 20, 30, 40, 90}
CH = ["burned_frac", "af_count_24h", "af_frp_24h", "af_count_all", "prior_burn_frac", "fbfm40", "grass_frac", "crop_frac",
      "shrub_tree_frac", "wetland_frac", "nonburnable_frac", "elev", "slope_deg", "pre_NDVI", "pre_NDMI", "dist_ign_km"]
HZ = (24, 48, 72)


def gfs(e):
    f = os.path.join(GD, e.event + ".csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["time"])
    if e.ign_time_utc < pd.Timestamp("2021-01-01"):
        return None
    t0 = e.ign_time_utc.normalize(); t1 = t0 + pd.Timedelta(days=45)
    q = dict(latitude=e.ign_lat, longitude=e.ign_lon, start_date=str(t0.date()), end_date=str(t1.date()), models="gfs_seamless",
             hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,vapour_pressure_deficit",
             wind_speed_unit="ms", timezone="GMT")
    j = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params=q, timeout=120).json()
    h = pd.DataFrame(j["hourly"]); h["time"] = pd.to_datetime(h["time"])
    if h.drop(columns="time").notna().mean().mean() < 0.5:
        return None
    h.to_csv(f, index=False); time.sleep(1)
    return h


def wfeat(h, t, pref):
    o = {}
    if h is None:
        return o
    h = h.set_index("time")
    for H in HZ:
        w = h.loc[t:t + pd.Timedelta(hours=H - 1)]
        if len(w) < H * 0.8 or w.wind_speed_10m.isna().mean() > 0.2:
            continue
        u = (w.wind_speed_10m * np.sin(np.deg2rad(w.wind_direction_10m))).mean(); v = (w.wind_speed_10m * np.cos(np.deg2rad(w.wind_direction_10m))).mean()
        o.update({f"{pref}T_mean_{H}h": w.temperature_2m.mean(), f"{pref}RH_min_{H}h": w.relative_humidity_2m.min(),
                  f"{pref}VPD_mean_{H}h": w.vapour_pressure_deficit.mean(), f"{pref}ws_mean_{H}h": w.wind_speed_10m.mean(),
                  f"{pref}gust_max_{H}h": w.wind_gusts_10m.max(), f"{pref}wind_u_{H}h": u, f"{pref}wind_v_{H}h": v,
                  f"{pref}P_sum_{H}h": w.precipitation.sum()})
    return o


def to_grid(a, src_tr, src_crs, dst_tr, shape, rs=Resampling.average):
    out = np.full(shape, np.nan, np.float32)
    reproject(a.astype(np.float32), out, src_transform=src_tr, src_crs=src_crs, dst_transform=dst_tr, dst_crs=DST_CRS,
              resampling=rs, src_nodata=np.nan, dst_nodata=np.nan)
    return out


ev = load_events(); idx = []
for e in ev.itertuples():
    meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json"))); CC = meta["channels"]
    d = np.load(os.path.join(DATA, "cubes", e.event + ".npz")); X30 = d["x"].astype(np.float32)
    ar = np.load(os.path.join(DATA, "arrival", e.event + ".npz")); arr = ar["arrival_h"]
    lab = d["label"].astype(bool); ign = d["ignore"]; lc = d["landcover"]; H30, W30 = lab.shape
    if np.isfinite(arr[lab]).sum() <= 1:   # нет ни FIRMS, ни MCD64A1 (известен только пиксель возгорания) -> динамика неизвестна
        stale = os.path.join(OD, e.event + ".npz")
        if os.path.exists(stale):
            os.remove(stale)
        print(f"{e.event:24s} SKIP: arrival time unknown", flush=True); continue
    tr30 = Affine(*meta["transform"]); crs30 = meta["crs"]
    # общая сетка события в EPSG:6933, 500 м, с запасом 48 км (чтобы любую сцену 96x96 можно было вырезать)
    t = meta["transform"]; xs = [t[2], t[2] + W30 * t[0]]; ys = [t[5], t[5] + H30 * t[4]]
    gx, gy = warp_transform(crs30, DST_CRS, [xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    pad = N * CELL
    X0 = np.floor((min(gx) - pad) / CELL) * CELL; Y1 = np.ceil((max(gy) + pad) / CELL) * CELL
    GW = int(np.ceil((max(gx) + pad - X0) / CELL)); GH = int(np.ceil((Y1 - (min(gy) - pad)) / CELL))
    gtr = from_origin(X0, Y1, CELL, CELL); gs = (GH, GW)
    G = lambda a, rs=Resampling.average: to_grid(a, tr30, crs30, gtr, gs, rs)
    inwin = G(np.ones((H30, W30), np.float32))                         # доля ячейки внутри окна куба
    ign_frac = G(ign.astype(np.float32))
    valid_g = (inwin > 0.5) & ~(ign_frac > 0.5)
    static = {}
    lc_f = lambda ks: G(np.isin(lc, list(ks)).astype(np.float32))
    static["grass_frac"] = lc_f({30}); static["crop_frac"] = lc_f({40}); static["shrub_tree_frac"] = lc_f({10, 20})
    static["wetland_frac"] = lc_f({90}); static["nonburnable_frac"] = lc_f(set(FBFM40) - BURNABLE)
    fb = np.vectorize(lambda k: FBFM40.get(int(k), 0))(lc).astype(np.float32); fb[lc == 0] = np.nan
    static["fbfm40"] = G(fb, Resampling.mode)
    for k in ("elev", "slope_deg", "pre_NDVI", "pre_NDMI", "dist_ign_km"):
        static[k] = G(X30[CC.index(k)])
    static["prior_burn_frac"] = G(np.isfinite(X30[CC.index("prior_burn_days")]).astype(np.float32))
    # термоточки -> ячейки сетки
    fa = read_firms(e.event)
    if len(fa):
        px, py = warp_transform("EPSG:4326", DST_CRS, fa.longitude.tolist(), fa.latitude.tolist())
        fa["gc"] = ((np.array(px) - X0) // CELL).astype(int); fa["gr"] = ((Y1 - np.array(py)) // CELL).astype(int)
        fa = fa[(fa.gc >= 0) & (fa.gc < GW) & (fa.gr >= 0) & (fa.gr < GH)]
    era = pd.read_csv(os.path.join(DATA, "weather_hourly", e.event + ".csv"), parse_dates=["time"])
    dfw = pd.read_csv(os.path.join(DATA, "weather_daily", e.event + ".csv"), parse_dates=["date"])
    gf = gfs(e)
    tmax = np.nanmax(arr[lab]) if np.isfinite(arr[lab]).any() else 0
    Xs, Ys, Vs, XF, YF = [], [], [], [], []
    k = 0
    while 24 * k <= max(tmax, 0) and k <= 40:
        th = 24.0 * k; T = e.ign_time_utc + pd.Timedelta(hours=th)
        st = lab & (arr <= th)
        if st.sum() == 0:
            st = np.zeros_like(lab); ir, ic = meta["ign_rc"]; st[min(max(ir, 0), H30 - 1), min(max(ic, 0), W30 - 1)] = True
        bf = G(st.astype(np.float32))
        ys_ = [G((lab & (arr > th) & (arr <= th + h)).astype(np.float32)) for h in HZ]
        # центр сцены: центроид состояния
        rr, cc = np.nonzero(np.nan_to_num(bf) > 0)
        if len(rr) == 0:
            k += 1; continue
        r0 = int(np.clip(np.average(rr, weights=bf[rr, cc]) - N // 2, 0, GH - N)); c0 = int(np.clip(np.average(cc, weights=bf[rr, cc]) - N // 2, 0, GW - N))
        win = (slice(r0, r0 + N), slice(c0, c0 + N))
        afc = np.zeros(gs, np.float32); afp = np.zeros(gs, np.float32); afa = np.zeros(gs, np.float32)
        if len(fa):
            q = fa[(fa.time_utc > T - pd.Timedelta(hours=24)) & (fa.time_utc <= T)]
            np.add.at(afc, (q.gr.values, q.gc.values), 1); np.add.at(afp, (q.gr.values, q.gc.values), q.frp.fillna(0).values)
            q = fa[fa.time_utc <= T]; np.add.at(afa, (q.gr.values, q.gc.values), 1)
        layers = dict(burned_frac=np.nan_to_num(bf), af_count_24h=afc, af_frp_24h=afp, af_count_all=afa, **static)
        XF.append(np.stack([layers[c] for c in CH]).astype(np.float16)); YF.append(np.stack([np.nan_to_num(y) for y in ys_]).astype(np.float16))
        Xs.append(np.stack([layers[c][win] for c in CH])); Ys.append(np.stack([np.nan_to_num(y[win]) for y in ys_])); Vs.append(valid_g[win])
        row = dict(event=e.event, site=e.site, year=e.year, split_time=e.split_time, slice=len(Xs) - 1, k_day=k, t_h=th,
                   t_utc=T, r0=r0, c0=c0, grid_x0=X0 + c0 * CELL, grid_y1=Y1 - r0 * CELL,
                   state_km2=float(np.nansum(bf) * 0.25), af_24h=int(afc.sum()),
                   **{f"new_{h}h_km2": float(np.nansum(ys_[i]) * 0.25) for i, h in enumerate(HZ)},
                   **{f"new_{h}h_km2_in_scene": float(Ys[-1][i].sum() * 0.25) for i, h in enumerate(HZ)},
                   valid_frac=float(valid_g[win].mean()))
        dd = dfw[dfw.date == (T + pd.Timedelta(hours=5)).normalize()]
        for kk in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI"):
            row[f"{kk}_day"] = float(dd[kk].iloc[0]) if len(dd) else np.nan
        row.update(wfeat(era, T.floor("h"), "era5_")); row.update(wfeat(gf, T.floor("h"), "gfs_"))
        idx.append(row); k += 1
    if Xs:
        np.savez_compressed(os.path.join(OD, e.event + ".npz"), X=np.stack(Xs).astype(np.float16), Y=np.stack(Ys).astype(np.float16), valid=np.stack(Vs))
        np.savez_compressed(os.path.join(FD, e.event + ".npz"), X=np.stack(XF), Y=np.stack(YF), valid=valid_g,
                            transform=np.array(list(gtr)[:6]), crs=DST_CRS)
    sub = [r for r in idx if r["event"] == e.event]
    print(f"{e.event:24s} slices={len(Xs):3d} tmax={tmax:7.1f}h gfs={'yes' if gf is not None else 'no '} "
          f"new24 km2: {[round(r['new_24h_km2'], 1) for r in sub][:8]}", flush=True)
si = pd.DataFrame(idx); si.to_csv(os.path.join(DATA, "slices_index.csv"), index=False)
json.dump(dict(channels=CH, targets=[f"new_{h}h" for h in HZ], cell_m=CELL, size=N, crs=DST_CRS,
               fbfm40_from_worldcover=FBFM40), open(os.path.join(DATA, "slices_channels.json"), "w"), indent=1)
print("slices:", len(si), "| events:", si.event.nunique(), "| with growth in 24 h:", int((si.new_24h_km2 > 0).sum()),
      "| with GFS:", int(si.filter(like="gfs_").notna().any(axis=1).sum()))
for h in HZ:
    q = si[si[f"new_{h}h_km2"] > 0]; fr = q[f"new_{h}h_km2_in_scene"] / q[f"new_{h}h_km2"]
    print(f"new burn inside 96x96 scene / total ({h}h): median {fr.median():.3f}, area-weighted {q[f'new_{h}h_km2_in_scene'].sum() / q[f'new_{h}h_km2'].sum():.3f}")
print(si.groupby("split_time").agg(slices=("slice", "size"), events=("event", "nunique")).to_string())
