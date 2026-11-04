"""Задание 2, шаг 23 (прод). Порог симулятора по площади: прогноз/факт ≈ 1 на растущих срезах val 2022 (GFS).

Порог по F1 (шаг 22) завышает площадь в 2–3,5 раза (F1 выгодно расширять прогноз), а пользователю симулятора нужен реалистичный
размер пожара. Здесь для каждого горизонта - уровень вероятности, при котором суммарная прогнозная площадь = фактической на растущих
срезах val; F1 и площадь при нём - на растущих срезах теста (GFS) и заповедников (ERA5, веса sim для ERA5 - как в шаге 22).
Выход: sim.level (заменяется), sim.level_f1 (порог шага 22) в out_v3/prod_hybrid.json; out_v3/prod_sim_area.csv.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("s22", os.path.join(os.path.dirname(__file__), "22_prod_sim.py"))
s22 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s22)
h20, ev12 = s22.h20, s22.ev12


def area_level(P, Y, L, si, k, h):
    a = s22.grow_idx(si, h); n = np.zeros(256); pos = 0
    for i in a:
        m = ev12.eval_mask(Y[i:i + 1], L[i:i + 1], k)[0]
        n += np.bincount(P[i, k][m], minlength=256); pos += (Y[i, k][m] > 0).sum()
    cum = n[::-1].cumsum()[::-1]                               # число ячеек с p >= уровня
    return int(np.argmin(np.abs(cum[1:] - pos)) + 1)


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(OUT, "prod_hybrid.json"))); rng = np.random.default_rng(0)
    parts = {p: ev12.load_part(p) for p in ("val", "test", "reserves")}; ld = h20.ld
    si_v, _, Y_v, L_v = parts["val"]; rows = []
    for src, part in (("gfsa", "test"), ("era5", "reserves")):
        Pv, Tv = ld(f"probs_wx_{src}_val.npy"), ld(f"phys_T_val_{src}.npy")
        if src == "gfsa":                                      # веса, экспортированные шагом 22 (их использует движок)
            ms = [s22._LR(cfg["sim"]["weights"][f"{h}h"]["coef"], cfg["sim"]["weights"][f"{h}h"]["intercept"]) for h in HZ]
        else:
            ms = s22.fit_masked(Pv, Tv, Y_v, L_v, si_v, src, rng, drop_unet=True)
        Pv_ = h20.apply(ms, Pv, Tv, si_v, src)
        cal = s22.calib_grow(Pv_, Y_v, L_v, si_v)
        for k, h in enumerate(HZ):
            cal[k]["level_f1"] = cal[k]["level"]; cal[k]["level"] = area_level(Pv_, Y_v, L_v, si_v, k, h)
        si, M, Y, L = parts[part]; P = h20.apply(ms, ld(f"probs_wx_{src}_{part}.npy"), ld(f"phys_T_{part}_{src}.npy"), si, src)
        rows += s22.score_grow(P, Y, L, si, cal, "sim_area", part, src)
        if src == "gfsa":
            cfg["sim"].setdefault("level_f1", cfg["sim"]["level"])
            cfg["sim"]["level"] = {f"{h}h": int(c["level"]) for h, c in zip(HZ, cal)}
            print("levels area / f1", cfg["sim"]["level"], cfg["sim"]["level_f1"], flush=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "prod_sim_area.csv"), index=False)
    json.dump(cfg, open(os.path.join(OUT, "prod_hybrid.json"), "w"), indent=1)
    print("done", flush=True)
