"""Задание 2, шаг 20 (прод). Гибрид «U-Net + физика» для сайта: стекинг по ячейкам, отдельно на каждом горизонте.

logit p = a·logit(p_unet) + b·log(T/h) + c·[T недостижимо] + d·log(T/h)·ws_h + e,
T - время прихода огня по физике (FBP O-1b + эллипс по ветру, шаг 14/19), h - горизонт, ws_h - ветер окна, м/с.
Физика задаёт направление (по ветру) и порядок выгорания, U-Net - где и сколько сгорит по данным. log(T/h) не зависит от масштаба
k_ros (сдвиг уходит в свободный член). Веса - логистическая регрессия на val 2022 (метки GFA + VIIRS); вариант «res» - на экспертных
пожарах заповедников с исключением одного пожара (leave-one-fire-out, честная оценка на тех же заповедниках).
Порог и изотоническая калибровка - по val, как в шаге 12. Плюс метрики по ветру: cos(смещение центра прогноза, ветер «куда»).
Выход: out_v3/prod_hybrid.json (веса), out_v3/prod_eval.csv, out_v3/prod_wind.csv.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("ev12", os.path.join(os.path.dirname(__file__), "12_eval_all.py"))
ev12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev12)
TMAX = 1000.0; NEG_RATE = 0.03
yy, xx = np.mgrid[0:96, 0:96]


def feats(P, T, ws, h):
    """P [n,96,96] u8, T [n,96,96] f16 (ч), ws [n] -> [n,96,96,4] f32."""
    p = np.clip(P.astype(np.float32) / 255, 0.5 / 255, 254.5 / 255); lp = np.log(p / (1 - p))
    t = T.astype(np.float32); far = (~np.isfinite(t)) | (t >= TMAX); lt = np.log(np.clip(np.where(far, TMAX, t), 0.5, TMAX) / HZ[h])
    return np.stack([lp, lt, far.astype(np.float32), lt * ws[:, None, None]], -1)


def wind_speed(si, src, h):
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    return np.nan_to_num(np.hypot(si[f"{pre}wind_u_{HZ[h]}h"].values, si[f"{pre}wind_v_{HZ[h]}h"].values)).astype(np.float32)


def sample(P, T, Y, L, si, src, idx, rng):
    """Обучающие ячейки: все положительные + NEG_RATE отрицательных (вес 1/NEG_RATE), по горизонтам."""
    out = []
    for h in range(len(HZ)):
        Xs, ys, ws_ = [], [], []
        ws = wind_speed(si, src, h)
        for c in np.array_split(idx, max(1, len(idx) // 256)):
            F_ = feats(P[c, h], T[c], ws[c], h); m = ev12.eval_mask(Y[c], L[c], h); y = (Y[c, h] > 0) & m
            keep = y | (m & (rng.random(m.shape) < NEG_RATE))
            Xs.append(F_[keep]); ys.append(y[keep]); ws_.append(np.where(y[keep], 1.0, 1 / NEG_RATE))
        out.append((np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws_)))
    return out


def fit(samples):
    return [LogisticRegression(C=1.0, max_iter=500).fit(X, y, sample_weight=w) for X, y, w in samples]


def apply(models, P, T, si, src):
    """-> P_hyb [n,4,96,96] u8."""
    out = np.zeros(P.shape, np.uint8)
    for h, lr in enumerate(models):
        ws = wind_speed(si, src, h); w, b = lr.coef_[0], lr.intercept_[0]
        for c in np.array_split(np.arange(len(P)), max(1, len(P) // 256)):
            z = feats(P[c, h], T[c], ws[c], h) @ w + b
            out[c, h] = np.round(255 / (1 + np.exp(-z))).astype(np.uint8)
    return out


def wind_cos(P, Y, M, si, src, h, min_new=8, min_ws=4):
    """Средний cos между смещением центра прогноза (вероятность как вес) от центра текущей гари и ветром «куда»."""
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    u, v = -np.nan_to_num(si[f"{pre}wind_u_{HZ[h]}h"].values), -np.nan_to_num(si[f"{pre}wind_v_{HZ[h]}h"].values); ws = np.hypot(u, v)
    newa = Y[:, h].reshape(len(Y), -1).sum(1); cs = []
    for i in np.where((newa >= min_new) & (ws >= min_ws))[0]:
        b = np.asarray(M[i, 0], np.float32); a = P[i, h].astype(np.float32) * (b == 0)
        if b.sum() == 0 or a.sum() == 0:
            continue
        d = np.array([(a * xx).sum() / a.sum() - (b * xx).sum() / b.sum(), -((a * yy).sum() / a.sum() - (b * yy).sum() / b.sum())])
        cs.append(d @ np.array([u[i], v[i]]) / (np.linalg.norm(d) * ws[i] + 1e-9))
    cs = np.array(cs)
    return dict(n=len(cs), cos=float(cs.mean()) if len(cs) else np.nan, downwind=float((cs > 0.707).mean()) if len(cs) else np.nan)


def ld(name):
    return np.load(os.path.join(OUT, name), mmap_mode="r")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    parts = {p: ev12.load_part(p) for p in ("val", "test", "reserves")}
    inp = {("val", "era5"): (ld("probs_wx_era5_val.npy"), ld("phys_T_val_era5.npy")),
           ("val", "gfsa"): (ld("probs_wx_gfsa_val.npy"), ld("phys_T_val_gfsa.npy")),
           ("test", "era5"): (ld("probs_wx_era5_test.npy"), ld("phys_T_test_era5.npy")),
           ("test", "gfsa"): (ld("probs_wx_gfsa_test.npy"), ld("phys_T_test_gfsa.npy")),
           ("reserves", "era5"): (ld("probs_wx_era5_reserves.npy"), ld("phys_T_reserves_era5.npy"))}
    si_v, M_v, Y_v, L_v = parts["val"]
    # веса гибрида - по val той же погоды, что на входе (ERA5 - «идеальный прогноз»; GFS - как в проде)
    MODELS_ = {src: fit(sample(*inp[("val", src)], Y_v, L_v, si_v, src, np.arange(len(si_v)), rng)) for src in ("era5", "gfsa")}
    models = MODELS_["era5"]
    Wall = {src: {f"{h}h": dict(coef=lr.coef_[0].round(4).tolist(), intercept=round(float(lr.intercept_[0]), 4)) for h, lr in zip(HZ, ms)}
            for src, ms in MODELS_.items()}
    W = Wall["era5"]
    print("weights [logit_unet, log(T/h), far, log(T/h)*ws]:", json.dumps(Wall), flush=True)

    rows, wrows = [], []

    def run(name, src, P_of):
        """P_of(part) -> u8 [n,4,96,96]; калибровка/порог по val той же погоды."""
        Pv = P_of("val", src); cal = ev12.fit_calib(Pv, Y_v, L_v, rng)
        for part in ("test", "reserves"):
            if part == "reserves" and src == "gfsa":
                continue
            si, M, Y, L = parts[part]; P = P_of(part, src)
            rr, _ = ev12.score(P, Y, L, si, cal, None)
            rows.extend(dict(model=name, src=src, part=part, **r) for r in rr)
            for h in (1, 3):
                wrows.append(dict(model=name, src=src, part=part, horizon_h=HZ[h], **wind_cos(P, Y, M, si, src, h)))
            print(f"{name:10s} {src} {part:8s} " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}[{r['lo']:.3f},{r['hi']:.3f}]/{r['F1_tol1']:.3f} "
                                                          f"BSS {r['BSS']:.3f} A {r['area_ratio']:.2f}" for r in rr), flush=True)
        pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "prod_eval.csv"), index=False)
        pd.DataFrame(wrows).round(3).to_csv(os.path.join(OUT, "prod_wind.csv"), index=False)

    cache = {}

    def unet(part, src):
        return np.asarray(inp[(part, src)][0])

    def physics(part, src):                                  # только физика: p = доля по log(T/h), порог по val
        T = np.asarray(inp[(part, src)][1], np.float32)
        return np.stack([np.round(255 / (1 + np.exp(3 * np.log(np.clip(T, 0.5, TMAX) / h)))).astype(np.uint8) for h in HZ], 1)

    def hybrid(part, src):
        if (part, src) not in cache:
            cache[(part, src)] = apply(MODELS_[src], *inp[(part, src)], parts[part][0], src)
        return cache[(part, src)]

    for src in ("era5", "gfsa"):
        run("unet", src, unet); run("physics", src, physics); run("hybrid", src, hybrid)

    # leave-one-fire-out на экспертных пожарах заповедников: веса гибрида по 34 пожарам, прогноз на 35-м
    si_r, M_r, Y_r, L_r = parts["reserves"]; Pr, Tr = inp[("reserves", "era5")]; g = si_r.event.values if "event" in si_r else si_r.fire_id.values
    Pl = np.zeros(Pr.shape, np.uint8); Wr = []; Br = []
    for f in np.unique(g):
        tr = np.where(g != f)[0]; te = np.where(g == f)[0]
        ms = fit(sample(Pr, Tr, Y_r, L_r, si_r, "era5", tr, rng)); Wr.append([lr.coef_[0] for lr in ms]); Br.append([lr.intercept_[0] for lr in ms])
        Pl[te] = apply(ms, np.asarray(Pr[te]), np.asarray(Tr[te]), si_r.iloc[te].reset_index(drop=True), "era5")
    Wr = np.array(Wr).mean(0)
    print("reserves LOFO mean weights:", Wr.round(3).tolist(), flush=True)
    # порог для LOFO: по самим заповедникам нельзя - берём уровень, дающий ту же долю площади, что и гибрид по val (без подгонки к экспертам)
    cal_v = ev12.fit_calib(hybrid("val", "era5"), Y_v, L_v, rng)
    rr, _ = ev12.score(Pl, Y_r, L_r, si_r, cal_v, None)
    rows.extend(dict(model="hybrid_res_lofo", src="era5", part="reserves", **r) for r in rr)
    for h in (1, 3):
        wrows.append(dict(model="hybrid_res_lofo", src="era5", part="reserves", horizon_h=HZ[h], **wind_cos(Pl, Y_r, M_r, si_r, "era5", h)))
    print("hybrid_res_lofo reserves " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}[{r['lo']:.3f},{r['hi']:.3f}] A {r['area_ratio']:.2f}" for r in rr), flush=True)
    for h in (1, 3):
        wrows.append(dict(model="labels", src="era5", part="reserves", horizon_h=HZ[h], **wind_cos(np.asarray(Y_r) * 255, Y_r, M_r, si_r, "era5", h)))
        wrows.append(dict(model="labels", src="era5", part="test", horizon_h=HZ[h], **wind_cos(np.asarray(parts["test"][2]) * 255, parts["test"][2], parts["test"][1], parts["test"][0], "era5", h)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "prod_eval.csv"), index=False)
    pd.DataFrame(wrows).round(3).to_csv(os.path.join(OUT, "prod_wind.csv"), index=False)
    json.dump(dict(features=["logit_unet", "log_T_over_h", "unreachable", "log_T_over_h_x_ws"], tmax=TMAX, k_ros=0.05, unet="wx",
                   val=W, val_gfsa=Wall["gfsa"], reserves_lofo_mean={f"{h}h": Wr[i].round(4).tolist() for i, h in enumerate(HZ)},
                   reserves_lofo_intercept={f"{h}h": round(float(np.mean(Br, 0)[i]), 4) for i, h in enumerate(HZ)},
                   levels_val={f"{h}h": c["level"] for h, c in zip(HZ, cal_v)}), open(os.path.join(OUT, "prod_hybrid.json"), "w"), indent=1)
    print(pd.DataFrame(wrows).to_string(), flush=True)
    print("done", flush=True)
