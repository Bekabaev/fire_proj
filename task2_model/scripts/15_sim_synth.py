"""Задание 2, шаг 15. Симуляции: синтетические пожары, проверка физичности моделей, вероятность выгорания.

Три уровня (принятая практика проверки ML-эмуляторов пожаров):
1) синтетические сцены 96 × 96 (500 м): ровная степь при ветре 0/3/6/10 м/с, смена ветра, река-барьер, полоса пашни,
   мозаика топлива, холм. Эталон - физика (sim_physics: FBP O-1b + эллипс, k_ros с шага 14). Модели (wx, swxh) получают те же
   карты и погоду в формате обучающих срезов. Метрики: Sørensen (= F1) модели к физике на 12/24/48/72 ч, площадь, вытянутость L/B
   по ветру (у физики по FBP LB = 1 + 8,729·(1 − e^(−0,030·WS))^2,155), прохождение барьера;
2) синтетические возгорания на реальных ландшафтах теста (карты и погода ERA5 срезов 2023–24, случайная точка возгорания) -
   согласие модели с физикой по 300 случаям; карта вероятности выгорания за 72 ч (100 возгораний на одной сцене, как FSim / Burn-P3);
3) повтор реальных пожаров теста: прогноз модели против факта (кадры для презентации).
Порог модели - по F1 на val 2022 (как в шаге 12). Выход: out_v3/sim_*.csv, figures/sim/*.png|gif, out_v3/sim_web.json.
"""
import os, sys, json, math
import numpy as np, pandas as pd, torch
import importlib.util
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *
import sim_physics as sp

_s = importlib.util.spec_from_file_location("ev12", os.path.join(os.path.dirname(__file__), "12_eval_all.py"))
ev12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev12)
FIGS = os.path.join(FIG, "sim"); os.makedirs(FIGS, exist_ok=True)
N = 96; RUNS = ("wx", "swxh")
stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
pb = pd.read_csv(os.path.join(OUT, "phys_baseline.csv")) if os.path.exists(os.path.join(OUT, "phys_baseline.csv")) else None
# k_ros для симуляций - площадь как в факте (отношение площадей на 24–48 ч ближе всего к 1 на val), а не максимум F1 (F1 выгодно завышать площадь)
if pb is not None:
    v = pb[(pb.part == "val600") & pb.horizon_h.isin([24, 48])].groupby("k_ros").area_ratio.apply(lambda a: abs(np.log(a)).mean())
    K_ROS = float(v.idxmin())
else:
    K_ROS = 0.018


# ---------- синтетический ландшафт и обучающий формат ----------
def smooth_noise(rng, s):
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(rng.standard_normal((N, N)), s); return (a - a.min()) / (a.max() - a.min() + 1e-9)


def landscape(kind, rng):
    L = dict(grass=np.ones((N, N)), crop=np.zeros((N, N)), shrub=np.zeros((N, N)), wet=np.zeros((N, N)), nonb=np.zeros((N, N)),
             elev=np.full((N, N), 300.0), ndvi=np.full((N, N), 0.30), road=np.zeros((N, N), bool))
    if kind == "river":
        L["nonb"][:, 58:60] = 1; L["grass"][:, 58:60] = 0
    if kind == "cropbreak":
        L["crop"][:, 56:64] = 1; L["grass"][:, 56:64] = 0; L["ndvi"][:, 56:64] = 0.55       # зелёная пашня
    if kind == "road":
        L["road"][:, 58] = True; L["nonb"][:, 58] = 0.6; L["grass"][:, 58] = 0.4
    if kind == "mosaic":
        n = smooth_noise(rng, 4); n2 = smooth_noise(rng, 6)
        L["crop"] = (n > 0.6).astype(float); L["shrub"] = ((n2 > 0.7) & (n <= 0.6)).astype(float); L["grass"] = 1 - L["crop"] - L["shrub"]
        L["ndvi"] = 0.3 + 0.25 * L["crop"] + 0.15 * L["shrub"]
    if kind == "hill":
        yy, xx = np.mgrid[0:N, 0:N]; L["elev"] = 300 + 600 * np.exp(-((xx - 66) ** 2 + (yy - 48) ** 2) / (2 * 10 ** 2))
    return L


def to_slice(L, burned, winds, ffmc=92.0, T=30.0, RH=20.0, age_h=6.0):
    """Сырые 23 канала, valid и строка индекса (погода ERA5-формата) как у обучающего среза. winds - куда дует; в строку пишется вектор «откуда», как в датасете."""
    gy, gx = np.gradient(L["elev"], 500.0); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    edge = burned & ~__import__("scipy.ndimage", fromlist=["binary_erosion"]).binary_erosion(burned)
    ys, xs = np.nonzero(burned); ign = (ys.mean(), xs.mean())
    yy, xx = np.mgrid[0:N, 0:N]
    if L["road"].any():
        from scipy.ndimage import distance_transform_edt
        droad = distance_transform_edt(~L["road"]) * 0.5
    else:
        droad = np.full((N, N), 20.0)
    raw = dict(burned_frac=burned.astype(float), burn_age_h=np.where(burned, age_h, 0), af_count_12h=edge * 1.0, af_frp_12h=edge * 20.0,
               af_count_24h=edge * 1.0, af_count_all=edge * 1.0, prior_burn=np.zeros((N, N)), years_since_burn=np.full((N, N), 13.0),
               fbfm40=np.full((N, N), 102.0), grass_frac=L["grass"], crop_frac=L["crop"], shrub_tree_frac=L["shrub"], wetland_frac=L["wet"],
               nonburnable_frac=L["nonb"], elev=L["elev"], slope_deg=slope, ndvi=L["ndvi"], ndvi_anom=np.zeros((N, N)),
               ndvi_peak_prev=L["ndvi"] + 0.2, dist_ign_km=np.hypot(yy - ign[0], xx - ign[1]) * 0.5, dist_settlement_km=np.full((N, N), 25.0),
               dist_road_km=droad, road_density=L["road"].astype(float))
    X = np.stack([raw[c] for c in CH]).astype(np.float32)[None]
    valid = (~burned)[None]
    row = dict(t_utc="2023-08-15 12:00:00", hour_utc=12, k_12h=1, state_km2=burned.sum() * 0.25, af_12h=int(edge.sum()))
    ws_ = [math.hypot(*w) for w in winds]
    for h, w, s in zip(HZ, winds, ws_):
        row.update({f"era5_T_max_{h}h": T, f"era5_RH_min_{h}h": RH, f"era5_ws_mean_{h}h": s, f"era5_gust_max_{h}h": 1.8 * s + 2,
                    f"era5_wind_u_{h}h": -w[0], f"era5_wind_v_{h}h": -w[1], f"era5_P_sum_{h}h": 0.0})
    isi_ = float(sp.isi(ws_[1] * 3.6, ffmc)); bui = 90.0
    row.update(era5_P_sum_prev7d=0.0, era5_VPD_mean_prev7d=1.8, era5_soilm_t=0.18, era5_RH_min_prev24h=RH, era5_ws_mean_prev24h=ws_[0],
               gfwed_FFMC=ffmc, gfwed_DMC=60.0, gfwed_DC=400.0, gfwed_ISI=isi_, gfwed_BUI=bui, gfwed_FWI=float(sp.fwi_from(isi_, bui)),
               gfwed_FWI_max3d=float(sp.fwi_from(isi_, bui)))
    return X, valid, row


def model_inputs(X, valid, rows):
    M = nn_maps(X, valid).astype(np.float32)
    si = pd.DataFrame(rows)
    Z = np.c_[base_scalars(si), weather_vec(si, stats, "era5")].astype(np.float32)
    return M, Z


def physics(L, burned, winds, ffmc=92.0):
    fuel = sp.fuel_factor(L["grass"], L["crop"], L["shrub"], L["wet"], L["nonb"])
    return sp.arrival(burned, fuel, L["ndvi"], L["elev"], winds, ffmc, k_ros=K_ROS)


def disk(cy, cx, r=2):
    yy, xx = np.mgrid[0:N, 0:N]; return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def sorensen(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def lb_ratio(mask, wind):
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return np.nan
    u, v = wind; n = math.hypot(u, v)
    d = np.array([u, v]) / n if n > 0.1 else np.array([1.0, 0.0])
    pe = xs * d[0] - ys * d[1]; pc = xs * (-d[1]) - ys * (-d[0])
    return (np.ptp(pe) + 1) / (np.ptp(pc) + 1)


# ---------- модели ----------
def load_models():
    out = {}
    si, M, Y, L = ev12.load_part("val")
    for run in RUNS:
        model, ck = ev12.load_model(run)
        Zv = np.c_[base_scalars(si), weather_vec(si, stats, "era5")].astype(np.float32)
        Pv = ev12.predict(model, M, Zv)
        cal = ev12.fit_calib(Pv, Y, L, None)
        out[run] = (model, [c["level"] for c in cal], [c["lut"] for c in cal])
        print(run, "val levels", out[run][1], flush=True)
    return out


def predict(models, M, Z):
    res = {}
    for run, (model, lev, lut) in models.items():
        P = ev12.predict(model, M, Z)                                       # uint8 0..255
        res[run] = (P, np.array(lev))
    return res


# ---------- уровень 1: синтетические сценарии ----------
SCEN = [("flat_w0", "flat", [(0, 0)] * 4), ("flat_w3", "flat", [(3, 0)] * 4), ("flat_w6", "flat", [(6, 0)] * 4),
        ("flat_w10", "flat", [(10, 0)] * 4), ("shift_W_to_S", "flat", [(6, 0), (6, 0), (0, -6), (0, -6)]),
        ("river_w6", "river", [(6, 0)] * 4), ("cropbreak_w6", "cropbreak", [(6, 0)] * 4), ("road_w6", "road", [(6, 0)] * 4),
        ("mosaic_w6", "mosaic", [(6, 0)] * 4), ("hill_w3", "hill", [(3, 0)] * 4)]


def level1(models):
    rng = np.random.default_rng(1); rows = []; web = []
    for name, kind, winds in SCEN:
        L = landscape(kind, rng); b = disk(48, 36, 8)                       # ~50 км² - типичный размер пожара в срезах
        T = physics(L, b, winds)
        X, V, row = to_slice(L, b, winds); M, Z = model_inputs(X, V, [row]); pr = predict(models, M, Z)
        for run, (P, lev) in pr.items():
            for h, H_ in enumerate(HZ):
                pm = (P[0, h] >= lev[h]) & V[0]; ph = (T <= H_) & ~b
                rows.append(dict(scenario=name, model=run, horizon_h=H_, sorensen=sorensen(pm, ph), area_model_km2=pm.sum() * 0.25,
                                 area_phys_km2=ph.sum() * 0.25, LB_model=lb_ratio(pm | b, winds[min(h, 3)]), LB_phys=lb_ratio(ph | b, winds[min(h, 3)]),
                                 cross_barrier_model=float(pm[:, 60:].sum() * 0.25), cross_barrier_phys=float(ph[:, 60:].sum() * 0.25)))
        web.append(dict(name=name, kind=kind, winds=winds, landscape={k: np.round(np.asarray(v, float), 2).tolist() for k, v in L.items() if k != "road"},
                        road=L["road"].astype(int).tolist(), burned=b.astype(int).tolist(), phys_T=np.round(np.minimum(T, 999), 1).tolist(),
                        model={run: dict(P=pr[run][0][0].tolist(), level=pr[run][1].tolist()) for run in pr}))
        print(name, [(r["model"], r["horizon_h"], round(r["sorensen"], 2), round(r["LB_model"], 2), round(r["LB_phys"], 2)) for r in rows if r["scenario"] == name and r["horizon_h"] == 72], flush=True)
        panel(name, L, b, T, pr, winds)
    pd.DataFrame(rows).round(3).to_csv(os.path.join(OUT, "sim_synth.csv"), index=False)
    return web


def panel(name, L, b, T, pr, winds, truth=None, title=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    runs = list(pr); fig, ax = plt.subplots(1, 1 + len(runs), figsize=(4.2 * (1 + len(runs)), 4.2))
    bg = np.dstack([0.85 - 0.3 * L["grass"] * 0 + 0.0 * L["crop"], 0.85 + 0.1 * L["crop"], 0.7 - 0.2 * L["crop"]])
    bg[L["nonb"] > 0.5] = (0.55, 0.7, 0.9)
    cols = ["#f4a261", "#e76f51", "#c1121f", "#6a040f"]
    for a, (lab, data) in zip(ax, [("физика FBP" if truth is None else "факт", None)] + [(r, r) for r in runs]):
        a.imshow(bg, origin="upper"); a.contour(L["elev"], levels=6, colors="#999", linewidths=0.4) if np.ptp(L["elev"]) > 1 else None
        for h, H_ in enumerate(HZ):
            m = (T <= H_) if data is None else (pr[data][0][0, h] >= pr[data][1][h]) | b
            if truth is not None and data is None:
                m = truth[h] | b
            a.contour(m.astype(float), levels=[0.5], colors=cols[h], linewidths=1.6)
        a.imshow(np.ma.masked_where(~b, b), cmap="Greys", vmin=0, vmax=1.2, alpha=0.9)
        u, v = winds[0]
        if math.hypot(u, v) > 0.1:
            a.arrow(8, 8, 6 * u / math.hypot(u, v), -6 * v / math.hypot(u, v), width=0.8, color="k")
        a.set_title(lab, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.suptitle((title or name) + "   контуры: 12 / 24 / 48 / 72 ч", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=110); plt.close(fig)
    # GIF: физика по часам и прогноз модели на ближайший горизонт
    if truth is None:
        frames = []
        for t in range(0, 73, 3):
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
            for a, lab in zip(ax, ("физика", runs[0])):
                a.imshow(bg, origin="upper")
                if lab == "физика":
                    m = T <= t
                else:
                    h = int(np.searchsorted(HZ, max(t, 1)))
                    h = min(h, 3); m = ((pr[runs[0]][0][0, h] >= pr[runs[0]][1][h]) | b) if t > 0 else b
                a.imshow(np.ma.masked_where(~m, m), cmap="Reds", vmin=0, vmax=1.3, alpha=0.85)
                a.set_title(f"{lab}, {t} ч", fontsize=10); a.set_xticks([]); a.set_yticks([])
            fig.tight_layout(); fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])); plt.close(fig)
        frames[0].save(os.path.join(FIGS, f"{name}.gif"), save_all=True, append_images=frames[1:], duration=180, loop=0)


# ---------- уровень 2: синтетические возгорания на реальных ландшафтах ----------
def real_landscape(m):
    g = lambda k: m[MAPS.index(k)].astype(np.float32)
    elev = np.where(g("has_elev") > 0, g("elev") * 300 + 300, 300.0)
    return dict(grass=g("grass_frac"), crop=g("crop_frac"), shrub=g("shrub_tree_frac"), wet=g("wetland_frac"), nonb=g("nonburnable_frac"),
                elev=elev, ndvi=np.where(g("has_ndvi") > 0, g("ndvi") / 3, 0.3), road=np.zeros((N, N), bool),
                dist_road=g("dist_road_km") * 10, dist_set=g("dist_settlement_km") * 30, ndvi_anom=g("ndvi_anom") / 5, peak=g("ndvi_peak_prev") / 3)


def real_slice(m, row, b):
    L = real_landscape(m)
    winds = [(-np.nan_to_num(row.get(f"era5_wind_u_{h}h", 0.0)), -np.nan_to_num(row.get(f"era5_wind_v_{h}h", 0.0))) for h in HZ]
    X, V, r2 = to_slice(L, b, winds, ffmc=float(np.nan_to_num(row.get("gfwed_FFMC", 90), nan=90)))
    X[0, CH.index("dist_road_km")] = L["dist_road"]; X[0, CH.index("dist_settlement_km")] = L["dist_set"]
    X[0, CH.index("ndvi_anom")] = L["ndvi_anom"]; X[0, CH.index("ndvi_peak_prev")] = L["peak"]
    for k in row:                                                         # реальная погода среза
        if k.startswith(("era5_", "gfwed_")):
            r2[k] = row[k]
    r2["t_utc"] = row["t_utc"]; r2["hour_utc"] = row["hour_utc"]
    return L, winds, X, V, r2


def level2(models, n=300):
    si, M, Y, Lb = ev12.load_part("test"); rng = np.random.default_rng(2); rows = []
    for j, i in enumerate(rng.choice(len(si), n, replace=False)):
        m = np.asarray(M[i], np.float32); row = si.iloc[i].to_dict(); L = real_landscape(m)
        fuel = sp.fuel_factor(L["grass"], L["crop"], L["shrub"], L["wet"], L["nonb"])
        cand = np.argwhere(fuel[20:76, 20:76] > 0.5) + 20
        if len(cand) == 0:
            continue
        cy, cx = cand[rng.integers(len(cand))]; b = disk(cy, cx, 6) & (fuel > 0)
        L, winds, X, V, r2 = real_slice(m, row, b)
        T = sp.arrival(b, fuel, L["ndvi"], L["elev"], winds, r2["gfwed_FFMC"], k_ros=K_ROS)
        Mm, Z = model_inputs(X, V, [r2]); pr = predict(models, Mm, Z)
        for run, (P, lev) in pr.items():
            for h, H_ in enumerate(HZ):
                pm = (P[0, h] >= lev[h]) & V[0]; ph = (T <= H_) & ~b
                rows.append(dict(case=j, model=run, horizon_h=H_, sorensen=sorensen(pm, ph), area_model=pm.sum() * 0.25, area_phys=ph.sum() * 0.25,
                                 ws=math.hypot(*winds[1])))
    if rows:
        d = pd.DataFrame(rows); d.round(3).to_csv(os.path.join(OUT, "sim_real_ignitions.csv"), index=False)
        print(d.groupby(["model", "horizon_h"])[["sorensen", "area_model", "area_phys"]].median().round(3), flush=True)
    # карта вероятности выгорания за 72 ч: 100 возгораний на одной сцене
    i = int(si.index[(si.valid_frac > 0.95)][0]) if "valid_frac" in si else 0
    m = np.asarray(M[i], np.float32); row = si.iloc[i].to_dict(); L0 = real_landscape(m)
    fuel = sp.fuel_factor(L0["grass"], L0["crop"], L0["shrub"], L0["wet"], L0["nonb"])
    cand = np.argwhere(fuel[10:86, 10:86] > 0.5) + 10; BPp = np.zeros((N, N)); BPm = {r: np.zeros((N, N)) for r in RUNS}
    for k in range(100):
        cy, cx = cand[rng.integers(len(cand))]; b = disk(cy, cx, 6) & (fuel > 0)
        L, winds, X, V, r2 = real_slice(m, row, b)
        BPp += sp.arrival(b, fuel, L["ndvi"], L["elev"], winds, r2["gfwed_FFMC"], k_ros=K_ROS) <= 72
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.4)); ax = [ax]
    for a, (lab, A) in zip(ax, [("физика FBP", BPp)]):
        im = a.imshow(A / 100, cmap="inferno", vmin=0, vmax=max(0.05, (BPp / 100).max())); a.set_title(f"вероятность выгорания за 72 ч: {lab}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8); fig.savefig(os.path.join(FIGS, "burn_probability.png"), dpi=110); plt.close(fig)
    np.savez_compressed(os.path.join(OUT, "sim_burnprob.npz"), phys=BPp, **BPm)
    return dict(burnprob=dict(phys=(BPp / 100).round(2).tolist()),
                landscape={k: np.round(v, 2).tolist() for k, v in L0.items() if k in ("grass", "crop", "nonb")})


# ---------- уровень 1б: чувствительность к ветру на реальных срезах (подмена погоды) ----------
def level1b(models, n=60):
    """Реальные срезы теста с ростом ≥ 5 км² за 72 ч: всё как в срезе, кроме ветра (8 м/с на С/В/Ю/З и штиль во всех окнах).
    Метрики: cos угла между ветром и смещением центра новой гари от центра гари на t (1 = строго по ветру), площадь."""
    si, M, Y, Lb = ev12.load_part("test"); rng = np.random.default_rng(3)
    idx = rng.choice(np.nonzero((si.new_72h_km2 >= 5).values)[0], n, replace=False); rows = []
    WIND = {"calm": (0.0, 0.0), "to_E": (8.0, 0.0), "to_N": (0.0, 8.0), "to_W": (-8.0, 0.0), "to_S": (0.0, -8.0)}
    iu = [list(MAPS)]; Zcols = None
    for j, i in enumerate(idx):
        m = np.asarray(M[i], np.float32); row = si.iloc[[i]].copy(); b = m[MAPS.index("burned_frac")] > 0; V = Lb[i] != INVALID
        L = real_landscape(m); fuel = sp.fuel_factor(L["grass"], L["crop"], L["shrub"], L["wet"], L["nonb"])
        ffmc = float(np.nan_to_num(row.gfwed_FFMC.iloc[0], nan=90))
        cy, cx = np.argwhere(b).mean(0) if b.any() else (48, 48)
        for wn, (u, v) in WIND.items():
            r2 = row.copy()
            for h in HZ:                                                  # в датасете - вектор «откуда»
                r2[f"era5_wind_u_{h}h"] = -u; r2[f"era5_wind_v_{h}h"] = -v; r2[f"era5_ws_mean_{h}h"] = math.hypot(u, v)
                r2[f"era5_gust_max_{h}h"] = 1.8 * math.hypot(u, v) + 2
            Z = np.c_[base_scalars(r2), weather_vec(r2, stats, "era5")].astype(np.float32)
            pr = predict(models, m[None], Z)
            T = sp.arrival(b, fuel, L["ndvi"], L["elev"], [(u, v)] * 4, ffmc, k_ros=K_ROS)
            outs = {r: (P[0, 3] >= lev[3]) & V for r, (P, lev) in pr.items()}; outs["physics"] = (T <= 72) & ~b
            for r, pm in outs.items():
                if pm.sum() == 0:
                    rows.append(dict(case=j, wind=wn, model=r, area_km2=0.0, cos=np.nan)); continue
                py, px = np.argwhere(pm).mean(0); d = np.array([px - cx, -(py - cy)])
                cos = float(d @ np.array([u, v]) / (np.linalg.norm(d) * math.hypot(u, v) + 1e-9)) if wn != "calm" else np.nan
                rows.append(dict(case=j, wind=wn, model=r, area_km2=pm.sum() * 0.25, cos=cos))
    d = pd.DataFrame(rows); d.round(3).to_csv(os.path.join(OUT, "sim_wind_sensitivity.csv"), index=False)
    s = d.groupby("model").apply(lambda g: pd.Series(dict(cos_mean=g[g.wind != "calm"].cos.mean(),
                                                          share_downwind=(g[g.wind != "calm"].cos > 0.5).mean(),
                                                          area_calm=g[g.wind == "calm"].area_km2.median(), area_wind8=g[g.wind != "calm"].area_km2.median())))
    print("wind sensitivity\n", s.round(3).to_string(), flush=True)
    return s.round(3).reset_index().to_dict("records")


# ---------- уровень 3: реальные пожары теста ----------
def level3(models, k=4):
    si, M, Y, Lb = ev12.load_part("test")
    big = si.index[(si.new_72h_km2 >= 15) & (si.new_72h_src_af > 0.5)].tolist()[:k]
    stats_ = []
    for i in big:
        m = np.asarray(M[i], np.float32); row = si.iloc[i].to_dict(); L = real_landscape(m); b = m[MAPS.index("burned_frac")] > 0
        Z = np.c_[base_scalars(si.iloc[[i]]), weather_vec(si.iloc[[i]], stats, "era5")].astype(np.float32)
        pr = predict(models, m[None], Z)
        winds = [(-np.nan_to_num(row.get(f"era5_wind_u_{h}h", 0.0)), -np.nan_to_num(row.get(f"era5_wind_v_{h}h", 0.0))) for h in HZ]
        fuel = sp.fuel_factor(L["grass"], L["crop"], L["shrub"], L["wet"], L["nonb"])
        T = sp.arrival(b, fuel, L["ndvi"], L["elev"], winds, float(np.nan_to_num(row.get("gfwed_FFMC", 90), nan=90)), k_ros=K_ROS)
        truth = [(Y[i, h] > 0) for h in range(4)]
        prp = dict(pr); prp["физика"] = (np.stack([(T <= H_) for H_ in HZ])[None].astype(np.uint8) * 255, np.array([128] * 4))
        panel(f"real_{int(row['fire_id'])}_{int(row['t_h'])}", L, b, T, prp, winds, truth=truth,
              title=f"пожар {int(row['fire_id'])}, {str(row['t_utc'])[:13]} UTC")
        stats_.append(dict(fire_id=row["fire_id"], t_utc=row["t_utc"]))
    return stats_


if __name__ == "__main__":
    models = load_models()
    web = dict(k_ros=K_ROS, horizons=list(HZ), scenarios=level1(models))
    web["wind_sensitivity"] = level1b(models)
    web.update(level2(models, n=0)); web["real"] = level3(models)
    web["metrics"] = pd.read_csv(os.path.join(OUT, "sim_synth.csv")).to_dict("records")
    json.dump(web, open(os.path.join(OUT, "sim_web.json"), "w"))
    print("done", flush=True)
