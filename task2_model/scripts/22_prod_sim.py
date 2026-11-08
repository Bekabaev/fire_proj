"""Задание 2, шаг 22 (прод). Режим симулятора «пожар не гаснет»: калиброванная физика по растущим пожарам.

Почему отдельно: U-Net (и гибрид через логит U-Net) для небольшой гари почти не продолжает фронт наружу - в метках GFA большинство
мелких пожаров гаснет за сутки (на тесте растут ≥ 1 км² за 24 ч лишь 29 % первых срезов), и синтетическое возгорание сеть относит
к затухающим (проверено абляцией 04.11: max p вне гари < 0,03 при любом ветре, термоточках, радиусе). Для сценария «что если»
нужен ответ на вопрос «куда пойдёт огонь, если не погаснет» - это условная задача, и её честно решает физика (ветер, топливо, склон,
барьеры), откалиброванная на срезах, где пожар реально рос.

Модель «sim»: logit p = b·log(T/h) + c·[недостижимо] + d·log(T/h)·ws + e (как гибрид, но без U-Net), веса - на срезах val 2022
с ростом ≥ 1 км² за горизонт (прогноз GFS), изотоническая калибровка и порог по F1 там же.
Сравнение на растущих срезах теста 2023–24 (GFS) и заповедников (ERA5; веса sim для ERA5 подобраны на val-ERA5 так же):
U-Net (порог по всему val), гибрид (deploy), гибрид, подобранный на растущих (hybrid_active), sim.
Выход: блок "sim" в out_v3/prod_hybrid.json, out_v3/prod_sim_eval.csv.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("h20", os.path.join(os.path.dirname(__file__), "20_prod_hybrid.py"))
h20 = importlib.util.module_from_spec(_s); _s.loader.exec_module(h20)
ev12 = h20.ev12
MIN_KM2 = 1.0


class _LR:
    def __init__(s, coef, b):
        s.coef_ = np.array([coef], np.float64); s.intercept_ = np.array([b], np.float64)


def grow_idx(si, h):
    return np.where(si[f"new_{h}h_km2"].fillna(0).values >= MIN_KM2)[0]


def fit_masked(P, T, Y, L, si, src, rng, drop_unet):
    """Веса по горизонтам на растущих срезах; drop_unet - без признака U-Net (коэффициент 0)."""
    out = []
    for k, h in enumerate(HZ):
        a = grow_idx(si, h)
        X, y, w = h20.sample(np.asarray(P[a]), np.asarray(T[a]), np.asarray(Y[a]), np.asarray(L[a]), si.iloc[a].reset_index(drop=True), src, np.arange(len(a)), rng)[k]
        from sklearn.linear_model import LogisticRegression
        cols = [1, 2, 3] if drop_unet else [0, 1, 2, 3]
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X[:, cols], y, sample_weight=w)
        coef = np.zeros(4); coef[cols] = lr.coef_[0]
        out.append(_LR(coef, lr.intercept_[0]))
    return out


def score_grow(P, Y, L, si, cal, name, part, src):
    rows = []
    for k, h in enumerate(HZ):
        a = grow_idx(si, h)
        rr, _ = ev12.score(np.asarray(P[a]), np.asarray(Y[a]), np.asarray(L[a]), si.iloc[a].reset_index(drop=True), cal, None)
        rows.append(dict(model=name, part=part, src=src, n_slices=len(a), **rr[k]))
    print(f"{name:14s} {src} {part:8s} " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}[{r['lo']:.3f},{r['hi']:.3f}] A {r['area_ratio']:.2f} R {r['recall']:.2f}"
                                                   for r in rows), flush=True)
    return rows


def calib_grow(P, Y, L, si):
    """Калибровка и порог для каждого горизонта по растущим на этом горизонте срезам val."""
    cal = []
    for k, h in enumerate(HZ):
        a = grow_idx(si, h); cal.append(ev12.fit_calib(np.asarray(P[a]), np.asarray(Y[a]), np.asarray(L[a]), None)[k])
    return cal


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfg = json.load(open(os.path.join(OUT, "prod_hybrid.json")))
    parts = {p: ev12.load_part(p) for p in ("val", "test", "reserves")}
    ld = h20.ld
    inp = {("val", "era5"): (ld("probs_wx_era5_val.npy"), ld("phys_T_val_era5.npy")), ("val", "gfsa"): (ld("probs_wx_gfsa_val.npy"), ld("phys_T_val_gfsa.npy")),
           ("test", "gfsa"): (ld("probs_wx_gfsa_test.npy"), ld("phys_T_test_gfsa.npy")),
           ("reserves", "era5"): (ld("probs_wx_era5_reserves.npy"), ld("phys_T_reserves_era5.npy"))}
    si_v, M_v, Y_v, L_v = parts["val"]
    rows = []; blocks = {}
    for src, tparts in (("gfsa", ("test",)), ("era5", ("reserves",))):
        Pv, Tv = inp[("val", src)]
        hyb_all = [_LR(cfg["val_gfsa" if src == "gfsa" else "val"][f"{h}h"]["coef"], cfg["val_gfsa" if src == "gfsa" else "val"][f"{h}h"]["intercept"]) for h in HZ]
        models = {"hybrid": hyb_all,
                  "hybrid_active": fit_masked(Pv, Tv, Y_v, L_v, si_v, src, rng, drop_unet=False),
                  "sim": fit_masked(Pv, Tv, Y_v, L_v, si_v, src, rng, drop_unet=True)}
        for name, ms in models.items():
            print(name, src, "weights", [np.round(m.coef_[0], 3).tolist() + [round(float(m.intercept_[0]), 3)] for m in ms], flush=True)
        for name in ("unet",) + tuple(models):
            Pv_ = np.asarray(Pv) if name == "unet" else h20.apply(models[name], Pv, Tv, si_v, src)
            cal = ev12.fit_calib(Pv_, Y_v, L_v, None) if name in ("unet", "hybrid") else calib_grow(Pv_, Y_v, L_v, si_v)
            for part in tparts:
                si, M, Y, L = parts[part]; Pp, Tp = inp[(part, src)]
                P = np.asarray(Pp) if name == "unet" else h20.apply(models[name], Pp, Tp, si, src)
                rows += score_grow(P, Y, L, si, cal, name, part, src)
            if name == "sim" and src == "gfsa":
                blocks["sim"] = dict(weather="gfs", min_growth_km2=MIN_KM2,
                                     weights={f"{h}h": dict(coef=list(map(float, m.coef_[0])), intercept=float(m.intercept_[0])) for h, m in zip(HZ, models["sim"])},
                                     lut={f"{h}h": np.round(c["lut"], 5).tolist() for h, c in zip(HZ, cal)},
                                     level={f"{h}h": int(c["level"]) for h, c in zip(HZ, cal)})
            pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "prod_sim_eval.csv"), index=False)
    cfg.update(blocks); json.dump(cfg, open(os.path.join(OUT, "prod_hybrid.json"), "w"), indent=1)
    print("done", flush=True)
