"""Задание 2, шаг 24 (прод). Симулятор от точки возгорания: физика с калиброванной скоростью по МАЛЫМ начинающимся растущим пожарам.

Шаги 22–23 калибровали физику по всем растущим срезам - там преобладают большие пожары с длинной кромкой, и для возгорания из точки
симулятор получался слишком медленным (за 72 ч - единицы ячеек). Сценарий сайта - «огонь возник здесь», поэтому калибровка - на срезах
val 2022 (GFS) с k_12h = 0, текущей гарью ≤ 5 км² и ростом ≥ 1 км² за горизонт: множитель f(h) такой, что площадь {T ≤ f·h} = факту.
Время прихода T обратно пропорционально k_ros, поэтому f - это поправка скорости без пересчёта физики.
Вероятность: p = σ(−S·log(T / (f·h))), S = 3, затем изотоническая калибровка по тем же срезам; порог - p ≥ 0,5 по сырой шкале (T ≤ f·h).
Проверка - те же срезы теста (GFS) и заповедников (ERA5, f подобран так же на val-ERA5), против гибрида и U-Net.
Выход: блок "sim" в out_v3/prod_hybrid.json (прежний -> "sim_logit"), out_v3/prod_sim_speed.csv.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("s22", os.path.join(os.path.dirname(__file__), "22_prod_sim.py"))
s22 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s22)
h20, ev12 = s22.h20, s22.ev12
S = 3.0; FGRID = np.exp(np.linspace(np.log(0.2), np.log(20), 80))


def small_idx(si, h):
    return np.where(((si[f"new_{h}h_km2"].fillna(0) >= 1.0) & (si.k_12h == 0) & (si.state_km2 <= 5)).values)[0]


def speed_factor(T, Y, L, si):
    f = []
    for k, h in enumerate(HZ):
        pos = 0; cnt = np.zeros(len(FGRID))
        for i in small_idx(si, h):
            m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], k)[0]; t = np.sort(T[i].astype(np.float32)[m]); t = t[t > 0]
            pos += (Y[i, k][m] > 0).sum(); cnt += np.searchsorted(t, h * FGRID, side="right")
        f.append(float(FGRID[int(np.argmin(np.abs(cnt - pos)))]))
    return f


class _LR:
    def __init__(s, coef, b):
        s.coef_ = np.array([coef], np.float64); s.intercept_ = np.array([b], np.float64)


def sim_models(f):                                             # σ(−S·(log(T/h) − log f)); недостижимо -> −10
    return [_LR([0.0, -S, -10.0, 0.0], S * float(np.log(fh))) for fh in f]


def score_on(P, Y, L, si, cal, name, part, src, idx_fn):
    rows = []
    for k, h in enumerate(HZ):
        a = idx_fn(si, h)
        rr, _ = ev12.score(np.asarray(P[a]), np.asarray(Y[a]), np.asarray(L[a]), si.iloc[a].reset_index(drop=True), cal, None)
        rows.append(dict(model=name, part=part, src=src, subset=idx_fn.__name__, n_slices=len(a), **rr[k]))
    print(f"{name:8s} {src} {part:8s} {idx_fn.__name__:10s} " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}[{r['lo']:.3f},{r['hi']:.3f}] A {r['area_ratio']:.2f}" for r in rows), flush=True)
    return rows


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(OUT, "prod_hybrid.json")))
    parts = {p: ev12.load_part(p) for p in ("val", "test", "reserves")}; ld = h20.ld
    si_v, _, Y_v, L_v = parts["val"]; rows = []
    for src, part in (("gfsa", "test"), ("era5", "reserves")):
        Pv, Tv = ld(f"probs_wx_{src}_val.npy"), ld(f"phys_T_val_{src}.npy")
        f = speed_factor(Tv, Y_v, L_v, si_v); print(src, "speed factor f(h)", np.round(f, 2).tolist(), flush=True)
        ms = sim_models(f)
        Pv_ = h20.apply(ms, Pv, Tv, si_v, src)
        cal = []
        for k, h in enumerate(HZ):                              # калибровка по малым растущим val; порог - сырой 0,5 (T = f·h)
            a = small_idx(si_v, h); c = ev12.fit_calib(np.asarray(Pv_[a]), np.asarray(Y_v[a]), np.asarray(L_v[a]), None)[k]
            c["level"] = 128; cal.append(c)
        hyb = [s22._LR(cfg["val_gfsa" if src == "gfsa" else "val"][f"{h}h"]["coef"], cfg["val_gfsa" if src == "gfsa" else "val"][f"{h}h"]["intercept"]) for h in HZ]
        cal_h = ev12.fit_calib(h20.apply(hyb, Pv, Tv, si_v, src), Y_v, L_v, None); cal_u = ev12.fit_calib(np.asarray(Pv), Y_v, L_v, None)
        si, M, Y, L = parts[part]; Pp, Tp = ld(f"probs_wx_{src}_{part}.npy"), ld(f"phys_T_{part}_{src}.npy")
        for fn in (small_idx, s22.grow_idx):
            rows += score_on(h20.apply(ms, Pp, Tp, si, src), Y, L, si, cal, "sim", part, src, fn)
            rows += score_on(h20.apply(hyb, Pp, Tp, si, src), Y, L, si, cal_h, "hybrid", part, src, fn)
            rows += score_on(np.asarray(Pp), Y, L, si, cal_u, "unet", part, src, fn)
        if src == "gfsa":
            if cfg.get("sim", {}).get("kind") != "physics_speed":   # логистический sim шага 22 сохраняется для сравнения
                cfg["sim_logit"] = cfg.get("sim")
            cfg["sim"] = dict(weather="gfs", kind="physics_speed", speed_factor={f"{h}h": fh for h, fh in zip(HZ, f)}, sharpness=S,
                              weights={f"{h}h": dict(coef=list(map(float, m.coef_[0])), intercept=float(m.intercept_[0])) for h, m in zip(HZ, ms)},
                              lut={f"{h}h": np.round(c["lut"], 5).tolist() for h, c in zip(HZ, cal)}, level={f"{h}h": 128 for h in HZ})
        pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "prod_sim_speed.csv"), index=False)
    json.dump(cfg, open(os.path.join(OUT, "prod_hybrid.json"), "w"), indent=1)
    print("done", flush=True)
