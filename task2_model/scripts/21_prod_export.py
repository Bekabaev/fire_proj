"""Задание 2, шаг 21 (прод). Экспорт гибрида для движка prod/: веса выбранного варианта, изотоническая калибровка и порог.

VARIANT=val (веса по val 2022) | reserves (средние веса leave-one-fire-out по экспертным пожарам заповедников)
| iter (итерации 05.11, ITERATIONS.md: NAME - вариант шага 26, UNET - префиксы вероятностей ансамбля, MODELS - чекпойнты, TTA=1;
  веса гибрида - out_v3/iter_hybrid_<NAME>_gfsa.json, движок усредняет MODELS с TTA - блок deploy.unet).
Калибровка и порог - по val 2022 на прогнозе GFS (как в проде: на входе прогноз, а не реанализ).
Выход: блок "deploy" в out_v3/prod_hybrid.json.
"""
import os, sys, json, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("h20", os.path.join(os.path.dirname(__file__), "20_prod_hybrid.py"))
h20 = importlib.util.module_from_spec(_s); _s.loader.exec_module(h20)
ev12 = h20.ev12


class _LR:
    def __init__(s, coef, b):
        s.coef_ = np.array([coef], np.float64); s.intercept_ = np.array([b], np.float64)


if __name__ == "__main__":
    var = os.environ.get("VARIANT", "val")
    f = os.path.join(OUT, "prod_hybrid.json"); cfg = json.load(open(f))
    if var == "val":                                            # веса, подобранные на val с прогнозом GFS (вход прода)
        W = {h: (cfg["val_gfsa"][h]["coef"], cfg["val_gfsa"][h]["intercept"]) for h in cfg["val_gfsa"]}
    elif var == "iter":
        Wi = json.load(open(os.path.join(OUT, f"iter_hybrid_{os.environ['NAME']}_gfsa.json")))
        W = {h: (Wi[h]["coef"], Wi[h]["intercept"]) for h in Wi}
    else:                                                       # свободный член LOFO не сохранён - берём из val, наклоны - из заповедников
        W = {h: (cfg["reserves_lofo_mean"][h], cfg["reserves_lofo_intercept"][h]) for h in cfg["reserves_lofo_mean"]}
    models = [_LR(*W[f"{h}h"]) for h in HZ]
    si, M, Y, L = ev12.load_part("val")
    U = [u for u in os.environ.get("UNET", "wx").split(",")]
    Pu = np.round(np.mean([np.asarray(h20.ld(f"probs_{u}_gfsa_val.npy"), np.float32) for u in U], 0)).astype(np.uint8)
    P = h20.apply(models, Pu, h20.ld("phys_T_val_gfsa.npy"), si, "gfsa")
    cal = ev12.fit_calib(P, Y, L, np.random.default_rng(0))
    # режим «активный пожар» (симулятор): порог по F1 только на срезах, где пожар реально рос (≥ 1 км² за горизонт) -
    # прогноз «куда пойдёт огонь, если не погаснет»; обычный порог - ожидаемый исход с учётом затухания мелких пожаров
    lev_act = {}
    for k, h in enumerate(HZ):
        a = np.where(si[f"new_{h}h_km2"].fillna(0).values >= 1.0)[0]
        lev_act[f"{h}h"] = int(ev12.fit_calib(P[a], Y[a], L[a], np.random.default_rng(0))[k]["level"])
    print("active levels", lev_act, "share of val slices active 24h", round(float((si.new_24h_km2.fillna(0) >= 1).mean()), 3), flush=True)
    cfg["deploy"] = dict(variant=var, weather="gfs",
                         weights={f"{h}h": dict(coef=list(map(float, W[f"{h}h"][0])), intercept=float(W[f"{h}h"][1])) for h in HZ},
                         lut={f"{h}h": np.round(c["lut"], 5).tolist() for h, c in zip(HZ, cal)},
                         level={f"{h}h": int(c["level"]) for h, c in zip(HZ, cal)}, level_active=lev_act,
                         unet=dict(models=os.environ.get("MODELS", "wx").split(","), tta=os.environ.get("TTA") == "1", probs=U))
    json.dump(cfg, open(f, "w"), indent=1)
    print(var, "levels", [c["level"] for c in cal], flush=True)
