"""Задание 2, шаг 4. Проверка обученных моделей на экспертных контурах заповедников (датасет задания 1, 43 пожара, 235 срезов).

Срезы задания 1 (время прихода огня по FIRMS/MCD64A1 внутри экспертного контура 30 м) переводятся в формат казахстанских
срезов: состояние, термоточки, цели и valid - из задания 1; топливо, рельеф, NDVI, прошлые гари - из национальных слоёв
(сетки совпадают: 500 м EPSG:6933, кратны 500 м); погода - ERA5 задания 1 теми же окнами.
Модели и пороги - из шагов 1–3 без дообучения. Это независимая разметка (эксперт), но годы 2014–2020 совпадают с годами
обучения U-Net, а 2021–2022 - с годами обучения поправки (там мог быть тот же пожар в версии GFA) -> результаты по группам лет.
Выход: out/reserves_comparison.csv
"""
import os, sys, json, pickle, importlib
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
sys.path.insert(0, os.path.join(ROOT, "task1_dataset", "scripts"))
import kz_common as K
U = importlib.import_module("01_unet"); PH = importlib.import_module("02_physics"); HY = importlib.import_module("03_hybrid")

si1 = pd.read_csv(os.path.join(T1, "slices_index.csv"), parse_dates=["t_utc"])
ch1 = json.load(open(os.path.join(T1, "slices_channels.json")))["channels"]
ev1 = pd.read_csv(os.path.join(T1, "events_spread.csv"), parse_dates=["ign_time_utc"]).set_index("event")
lc = np.load(os.path.join(KZ, "lc.npz")); dem = np.load(os.path.join(KZ, "dem.npz"))


def weather_rows(event, t):
    h = pd.read_csv(os.path.join(T1, "weather_hourly", event + ".csv"), parse_dates=["time"]).set_index("time")
    o = {}
    for H in HZ:
        w = h.loc[t:t + pd.Timedelta(hours=H - 1)]
        if len(w) < H * 0.8:
            continue
        u = (w.wind_speed_10m * np.sin(np.deg2rad(w.wind_direction_10m))).mean(); v = (w.wind_speed_10m * np.cos(np.deg2rad(w.wind_direction_10m))).mean()
        o.update({f"T_mean_{H}h": w.temperature_2m.mean(), f"T_max_{H}h": w.temperature_2m.max(), f"RH_min_{H}h": w.relative_humidity_2m.min(),
                  f"VPD_mean_{H}h": w.vapour_pressure_deficit.mean(), f"ws_mean_{H}h": w.wind_speed_10m.mean(), f"gust_max_{H}h": w.wind_gusts_10m.max(),
                  f"wind_u_{H}h": u, f"wind_v_{H}h": v, f"P_sum_{H}h": w.precipitation.sum()})
    a = h.loc[t - pd.Timedelta(days=7):t]
    o.update(P_sum_prev7d=a.precipitation.sum(), VPD_mean_prev7d=a.vapour_pressure_deficit.mean(),
             soilm_t=a.soil_moisture_0_to_7cm.iloc[-1], RH_min_prev24h=a.relative_humidity_2m.iloc[-24:].min())
    return o


def build():
    Xs, Ys, Vs, rows = [], [], [], []
    ndvi_cache = {}
    for ev, s in si1.groupby("event"):
        d = np.load(os.path.join(T1, "slices", ev + ".npz")); e = ev1.loc[ev]; y = int(e.year)
        if y not in ndvi_cache:
            nd = np.load(os.path.join(KZ, f"ndvi_{y}.npz")); ndvi_cache = {y: (nd["ndvi"], nd["doy"])}
        NDV, NDOY = ndvi_cache[y]
        p1 = np.asarray(K.load_doy(y - 1)) > 0 if os.path.exists(K.doy_path(y - 1)) else None
        p2 = np.asarray(K.load_doy(y - 2)) > 0 if os.path.exists(K.doy_path(y - 2)) else None
        ign_doy = e.ign_time_utc.dayofyear; kn = np.nonzero(NDOY + 16 <= ign_doy)[0]
        for r in s.itertuples():
            R0 = int(round((K.Y1 - r.grid_y1) / K.CELL)); C0 = int(round((r.grid_x0 - K.X0) / K.CELL))
            w = (slice(R0, R0 + 96), slice(C0, C0 + 96))
            X1 = d["X"][r.slice].astype(np.float32); g1 = lambda k: X1[ch1.index(k)]
            prior = np.maximum(g1("prior_burn_frac"), 0)
            if p1 is not None:
                prior = np.maximum(prior, np.where(p1[w], 1.0, 0.0))
            if p2 is not None:
                prior = np.maximum(prior, np.where(p2[w], 0.5, 0.0))
            if len(kn):
                nv = NDV[kn[-1], R0 // 2:(R0 + 96) // 2 + 1, C0 // 2:(C0 + 96) // 2 + 1].astype(np.float32)
                nv = np.repeat(np.repeat(nv, 2, 0), 2, 1)[(R0 % 2):(R0 % 2) + 96, (C0 % 2):(C0 % 2) + 96]
                nv = np.where(nv < -20000, np.nan, nv * 1e-4)
            else:
                nv = np.full((96, 96), np.nan, np.float32)
            L = dict(burned_frac=(g1("burned_frac") > 0.5).astype(np.float32), af_count_24h=g1("af_count_24h"), af_frp_24h=g1("af_frp_24h"),
                     af_count_all=g1("af_count_all"), prior_burn=prior, fbfm40=lc["fbfm40"][w].astype(np.float32),
                     grass_frac=lc["frac"][0][w] / 100, crop_frac=lc["frac"][1][w] / 100, shrub_tree_frac=lc["frac"][2][w] / 100,
                     wetland_frac=lc["frac"][3][w] / 100, nonburnable_frac=lc["frac"][4][w] / 100,
                     elev=dem["elev"][w].astype(np.float32), slope_deg=dem["slope"][w].astype(np.float32), ndvi=nv, dist_ign_km=g1("dist_ign_km"))
            Xs.append(np.stack([L[c] for c in CH]).astype(np.float16))
            Y = (d["Y"][r.slice].astype(np.float32) > 0.5).astype(np.uint8); V = d["valid"][r.slice] & ~(L["burned_frac"] > 0.5)
            Ys.append(Y); Vs.append(V)
            row = dict(event=ev, site=e.site, year=y, slice=len(Xs) - 1, fire_id=ev, doy=r.t_utc.dayofyear, k_day=r.k_day,
                       state_km2=float(L["burned_frac"].sum() * 0.25), af_24h=float(L["af_count_24h"].sum()), lat=e.ign_lat, lon=e.ign_lon,
                       t_utc=r.t_utc, **{f"new_{h}h_km2": float(Y[i].sum() * 0.25) for i, h in enumerate(HZ)})
            row.update(weather_rows(ev, r.t_utc.floor("h"))); rows.append(row)
    return np.stack(Xs), np.stack(Ys), np.stack(Vs), pd.DataFrame(rows)


if __name__ == "__main__":
    X, Y, V, rs = build()
    print("reserve slices:", len(rs), "events:", rs.event.nunique(), "| new burn share of valid (24h):", round(float(Y[:, 0].sum() / V.sum()), 4), flush=True)
    ck = torch.load(os.path.join(OUT, "unet.pt")); net = U.CompactFront(ck["cin"], ck["ns"]).to(U.dev); net.load_state_dict(ck["state"])
    up = U.predict(net, U.nn_maps(X, V) if hasattr(U, "nn_maps") else nn_maps(X, V), scalars_unet(rs))
    pt = np.stack([PH.slice_physics(X[i], r._asdict()) for i, r in enumerate(rs.itertuples())])
    cmpj = json.load(open(os.path.join(OUT, "comparison.json"))); ths = cmpj["thresholds"]; m = cmpj["physics_multiplier"]
    P = {"unet": up, "phys": (pt / m <= np.array(HZ)[None, :, None, None]).astype(np.float32)}
    from scipy import ndimage
    P["persist"] = np.stack([np.repeat((ndimage.binary_dilation((X[i, CH.index("burned_frac")] > 0.5) | (X[i, CH.index("af_count_24h")] > 0),
                                                                  np.ones((3, 3), bool)) & ~(X[i, CH.index("burned_frac")] > 0.5))[None], 3, 0)
                             for i in range(len(X))]).astype(np.float32)
    for vn in ("ml_coords", "ml_nocoords", "hybrid_nocoords"):
        G = pickle.load(open(os.path.join(OUT, f"gbm_{vn}.pkl"), "rb")); out = []
        for i, r in enumerate(rs.itertuples()):
            p = np.stack([G["models"][hi].predict_proba(HY.cell_features(X[i], r._asdict(), up[i], pt[i], m, hi, G["coords"], G["phys"])[0])[:, 1].reshape(96, 96)
                          for hi in range(3)])
            out.append(np.maximum.accumulate(0.5 * up[i] + 0.5 * p, 0))
        P[vn] = np.stack(out)
    rows = []
    for grp, mk in [("all 2014-2023", np.ones(len(rs), bool)), ("2014-2020 (годы U-Net)", rs.year.values <= 2020), ("2021-2023", rs.year.values >= 2021)]:
        for k, p in P.items():
            a, b, c = HY.counts(p[mk] >= ths.get(k, 0.5), Y[mk], V[mk])
            for i, h in enumerate(HZ):
                lo, hi_ = f1_boot(a[:, i], b[:, i], c[:, i], rs.event.values[mk])
                rows.append(dict(group=grp, model=k, horizon_h=h, **{kk: round(vv, 4) for kk, vv in prf(a[:, i].sum(), b[:, i].sum(), c[:, i].sum()).items()},
                                 F1_lo=round(lo, 4), F1_hi=round(hi_, 4), slices=int(mk.sum()), events=int(rs.event[mk].nunique())))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(OUT, "reserves_comparison.csv"), index=False)
    print(out.pivot_table(index=["group", "model"], columns="horizon_h", values="F1").round(3).to_string())
