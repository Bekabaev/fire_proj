"""Задание 2, шаг 14. Физический бейзлайн (FBP O-1b + эллипс + кратчайшее время прихода, sim_physics.py) на реальных срезах.

Входы - те же карты кэша (трава / пашня / кусты / болото / негорючее, NDVI, высота) и погода ERA5 или прогноз GFS по окнам 12/24/48/72 ч
(ветер u, v; в датасете - вектор «откуда», здесь разворачивается), FFMC из GFWED. Единственный свободный параметр k_ros (множитель скорости) подбирается по F1 на val 2022 (600 срезов),
затем считается тест 2023–24 и заповедники - строгий F1, F1 с допуском 1 ячейка, отношение площадей. Выход: out_v3/phys_baseline.csv,
out_v3/phys_T_<часть>_<src>.npy (время прихода, ч, f16).
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import CACHE, OUT, HZ, MAPS, INVALID
import sim_physics as sp

ST1 = np.ones((3, 3), bool)
IX = {k: MAPS.index(k) for k in ("burned_frac", "grass_frac", "crop_frac", "shrub_tree_frac", "wetland_frac", "nonburnable_frac", "ndvi", "elev", "has_elev")}


def load(part):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    return si, *(np.load(os.path.join(CACHE, f"{part}_{k}.npy"), mmap_mode="r") for k in "MYL")


def inputs(m, row, src):
    g = lambda k: m[IX[k]].astype(np.float32)
    fuel = sp.fuel_factor(g("grass_frac"), g("crop_frac"), g("shrub_tree_frac"), g("wetland_frac"), g("nonburnable_frac"))
    elev = np.where(g("has_elev") > 0, g("elev") * 300 + 300, np.nan); elev = np.where(np.isfinite(elev), elev, np.nanmean(elev) if np.isfinite(elev).any() else 0)
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    winds = [(-np.nan_to_num(row.get(f"{pre}wind_u_{h}h", 0.0)), -np.nan_to_num(row.get(f"{pre}wind_v_{h}h", 0.0))) for h in HZ]   # в датасете вектор «откуда»
    ffmc = row.get("gfwed_FFMC", np.nan); ffmc = 88.0 if not np.isfinite(ffmc) else float(ffmc)
    return g("burned_frac") > 0, fuel, g("ndvi") / 3, elev, winds, ffmc, row.get("lat", np.nan)


def run_one(a):
    m, row, src, k = a
    b, fuel, ndvi, elev, winds, ffmc, lat = inputs(m, row, src)
    return sp.arrival(b, fuel, ndvi, elev, winds, ffmc, k_ros=k, lat=lat).astype(np.float16)    # геометрия ячейки по широте (шаг 19, 04.11)


def simulate(M, si, idx, src, k, pool):
    return np.stack(pool.map(run_one, [(np.asarray(M[i]), si.iloc[i].to_dict(), src, k) for i in idx], chunksize=8))


def score(T, Y, L, idx):
    out = []
    for h, H_ in enumerate(HZ):
        tp = fp = fn = tpp = tpr = 0
        for j, i in enumerate(idx):
            m = L[i] != INVALID
            if h == 0:
                m = m & ~((Y[i, 0] > 0) & (L[i] == 4))
            y = (Y[i, h] > 0) & m; p = (T[j] <= H_) & m
            tp += (p & y).sum(); fp += (p & ~y).sum(); fn += (~p & y).sum()
            tpp += (p & binary_dilation(y, ST1)).sum(); tpr += (y & binary_dilation(p, ST1)).sum()
        P = tpp / max(tp + fp, 1); R = tpr / max(tp + fn, 1)
        out.append(dict(horizon_h=H_, F1=2 * tp / max(2 * tp + fp + fn, 1), F1_tol1=2 * P * R / max(P + R, 1e-9),
                        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), area_ratio=(tp + fp) / max(tp + fn, 1)))
    return out


if __name__ == "__main__":
    rows = []
    with Pool(int(os.environ.get("PROCS", 6))) as pool:
        si, M, Y, L = load("val"); rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(si), 600, replace=False))
        best = None
        for k in (0.01, 0.018, 0.03, 0.05, 0.08):
            T = simulate(M, si, idx, "era5", k, pool); s = score(T, Y, L, idx); f = np.mean([r["F1"] for r in s])
            print(f"k_ros {k}: val F1 {[round(r['F1'], 3) for r in s]} area {[round(r['area_ratio'], 2) for r in s]}", flush=True)
            rows += [dict(part="val600", src="era5", k_ros=k, **r) for r in s]
            if best is None or f > best[1]:
                best = (k, f)
        k = best[0]; print("best k_ros", k, flush=True)
        for part, srcs in (("test", ("era5", "gfsa")), ("reserves", ("era5",))):
            si, M, Y, L = load(part); idx = np.arange(len(si))
            for src in srcs:
                T = simulate(M, si, idx, src, k, pool); np.save(os.path.join(OUT, f"phys_T_{part}_{src}.npy"), T)
                s = score(T, Y, L, idx); rows += [dict(part=part, src=src, k_ros=k, **r) for r in s]
                print(part, src, [round(r["F1"], 3) for r in s], "tol1", [round(r["F1_tol1"], 3) for r in s], "area", [round(r["area_ratio"], 2) for r in s], flush=True)
                pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "phys_baseline.csv"), index=False)
