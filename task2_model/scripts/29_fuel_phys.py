"""Задание 2, шаг 29. Физика с картой топлива FBFM40 (шаг 38 датасета, data/ext_kz/fuel_s2.npz) против текущей FBP.

Варианты (VAR):
  fbp     - как шаг 14: FBP O-1b, топливо = доли WorldCover (трава 1, пашня 0,6, кусты 0,4, болото 0,2);
  fbp_s2  - FBP O-1b, топливо ячейки = сумма долей кодов FBFM40 × относительная скорость кода по Ротермелу (GR2 = 1, сухо, 2 м/с),
            пашня 0,6; негорючее > 50 % - не горит;
  roth_s2 - Ротермел по каждому коду с погодой окна (T, RH -> влажность мёртвого топлива, NDVI -> влажность травы, ветер),
            скорость ячейки = сумма долей × скорость кода; форма эллипса и решатель - те же, что у FBP.
k (общий множитель скорости) подбирается по F1 на тех же 600 срезах val 2022, что в шаге 14; затем тест 2023–24 (GFS) и заповедники (ERA5).
Выход: out_v3/fuel_phys.csv, out_v3/phys_T_<часть>_<src>_<VAR>.npy.
"""
import os, sys, math, importlib.util
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import OUT
import sim_physics as sp
import physics as ph

_s = importlib.util.spec_from_file_location("ph14", os.path.join(os.path.dirname(__file__), "14_phys_baseline.py"))
ph14 = importlib.util.module_from_spec(_s); sys.modules["ph14"] = ph14; _s.loader.exec_module(ph14)

FUEL = os.path.join(os.path.dirname(__file__), "..", "..", "task1_dataset", "data", "ext_kz", "fuel_s2.npz")
CROP, NBC = 250, 99
WAF = 0.4                                     # ветер 10 м -> средняя высота пламени (трава, без полога)


def rel_speeds():
    """Относительная скорость кода (GR2 = 1) при сухой траве и ветре 2 м/с."""
    r = {c: ph.ros_rothermel(c, 0.06, 0.07, 0.08, 0.6, 0.9, 2.0) for c in ph.FM}
    return {c: v / r[102] for c, v in r.items()}


REL = rel_speeds()


def load_fuel():
    z = np.load(FUEL); return z["frac"], [int(c) for c in z["codes"]]


def window(F, r0, c0, n=96):
    """Доли [n_codes, n, n] (0–1) окна национальной сетки; за краем сетки - нули."""
    out = np.zeros((F.shape[0], n, n), np.float32)
    a = F[:, max(r0, 0):r0 + n, max(c0, 0):c0 + n].astype(np.float32) / 100
    out[:, max(-r0, 0):max(-r0, 0) + a.shape[1], max(-c0, 0):max(-c0, 0) + a.shape[2]] = a
    s = out.sum(0)                     # доли считаются от пикселей с данными (край страны, облака) -> нормируем к 1
    return np.where(s > 0.2, out / np.maximum(s, 1e-6), 0.0).astype(np.float32)


def fuel_s2(fr, codes):
    f = sum(fr[i] * (0.6 if c == CROP else REL.get(c, 0.0)) for i, c in enumerate(codes))
    nb = fr[codes.index(NBC)]
    return np.where(nb > 0.5, 0.0, f)


def roth_heads(fr, codes, ndvi, row, src, fuel_wc):
    """Скорость головы по Ротермелу, км/ч [4 окна, 96, 96] (до множителя k)."""
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    nd = float(np.nanmedian(ndvi)) if np.isfinite(ndvi).any() else 0.3
    mherb = float(np.clip(0.3 + (nd - 0.15) / 0.45 * 1.7, 0.3, 2.0))
    nb = fr[codes.index(NBC)] > 0.5; out = []
    for h in ph14.HZ:
        T = row.get(f"{pre}T_max_{h}h", np.nan); RH = row.get(f"{pre}RH_min_{h}h", np.nan)
        u = row.get(f"{pre}wind_u_{h}h", 0.0); v = row.get(f"{pre}wind_v_{h}h", 0.0)
        T = 20.0 if not np.isfinite(T) else T; RH = 40.0 if not np.isfinite(RH) else RH
        m1 = float(ph.emc(T, RH)); U = WAF * math.hypot(np.nan_to_num(u), np.nan_to_num(v))
        r = {c: ph.ros_rothermel(c, m1, m1 + 0.01, m1 + 0.02, mherb, 1.0, U) for c in codes if c in ph.FM}
        r2 = ph.ros_rothermel(102, m1, m1 + 0.01, m1 + 0.02, mherb, 1.0, U)
        head = sum(fr[i] * (0.6 * r2 if c == CROP else r.get(c, 0.0)) for i, c in enumerate(codes))
        head = np.where(fr.sum(0) > 0, head, fuel_wc * r2)          # нет карты -> WorldCover × GR2
        out.append(np.where(nb, 0.0, head) * 0.06)           # м/мин -> км/ч
    return out


def arrival_roth(burned, heads, elev, winds, ffmc, k, lat):
    """Решатель sim_physics.arrival, но с готовой скоростью головы по окнам (форма эллипса - FBP от ветра)."""
    orig = sp.speeds
    it = iter(heads)
    def sp_roth(fuel, ndvi, wind_uv, ffmc_, k_ros, ev=sp.EV):
        hd = next(it)
        u, v = wind_uv; ws = math.hypot(u, v) * 3.6
        h0 = float(sp.ros_o1b(sp.isi(ws, ffmc_))); b0 = float(sp.ros_o1b(sp.isi(ws, ffmc_, back=True)))
        lb = 1 + 8.729 * (1 - math.exp(-0.030 * ws)) ** 2.155
        d = np.array([u, v]) / (math.hypot(u, v) + 1e-9) if ws > 0.1 else np.array([1.0, 0.0])
        rel = np.array([sp.ellipse_speed(1.0, b0 / max(h0, 1e-9), lb, c) for c in ev @ d])
        return rel[:, None, None] * (hd * k_ros)[None]
    sp.speeds = sp_roth
    try:
        return sp.arrival(burned, np.ones_like(elev), np.zeros_like(elev), elev, winds, ffmc, k_ros=k, lat=lat)
    finally:
        sp.speeds = orig


def run_one(a):
    m, row, src, k, var, fr, codes = a
    b, fuel, ndvi, elev, winds, ffmc, lat = ph14.inputs(m, row, src)
    if var == "fbp":
        T = sp.arrival(b, fuel, ndvi, elev, winds, ffmc, k_ros=k, lat=lat)
    elif var == "fbp_s2":
        T = sp.arrival(b, np.where(fr.sum(0) > 0, fuel_s2(fr, codes), fuel), ndvi, elev, winds, ffmc, k_ros=k, lat=lat)   # нет карты -> WorldCover
    else:
        T = arrival_roth(b, roth_heads(fr, codes, ndvi, row, src, fuel), elev, winds, ffmc, k, lat)
    return T.astype(np.float16)


def simulate(M, si, idx, src, k, var, F, codes, pool):
    args = [(np.asarray(M[i]), si.iloc[i].to_dict(), src, k, var,
             window(F, int(si.r0.iloc[i]), int(si.c0.iloc[i])) if var != "fbp" else None, codes) for i in idx]
    return np.stack(pool.map(run_one, args, chunksize=8))


if __name__ == "__main__":
    var = os.environ.get("VAR", "fbp_s2")
    KS = {"fbp": (0.01, 0.018, 0.03, 0.05, 0.08), "fbp_s2": (0.01, 0.018, 0.03, 0.05, 0.08, 0.13),
          "roth_s2": (0.03, 0.06, 0.1, 0.18, 0.3, 0.5, 1.0)}[var]
    F, codes = load_fuel() if var != "fbp" else (None, None)
    f_csv = os.path.join(OUT, "fuel_phys.csv")
    rows = pd.read_csv(f_csv).to_dict("records") if os.path.exists(f_csv) else []
    rows = [r for r in rows if r["var"] != var]
    with Pool(int(os.environ.get("PROCS", 6))) as pool:
        si, M, Y, L = ph14.load("val"); idx = np.sort(np.random.default_rng(0).choice(len(si), 600, replace=False))
        best = None
        for k in KS:
            T = simulate(M, si, idx, "era5", k, var, F, codes, pool); s = ph14.score(T, Y, L, idx); f = np.mean([r["F1"] for r in s])
            print(f"{var} k {k}: val F1 {[round(r['F1'], 3) for r in s]} area {[round(r['area_ratio'], 2) for r in s]}", flush=True)
            rows += [dict(var=var, part="val600", src="era5", k=k, **r) for r in s]
            if best is None or f > best[1]:
                best = (k, f)
        k = best[0]; print(var, "best k", k, flush=True)
        for part, srcs in (("test", ("gfsa",)), ("reserves", ("era5",))):
            si, M, Y, L = ph14.load(part); idx = np.arange(len(si))
            for src in srcs:
                T = simulate(M, si, idx, src, k, var, F, codes, pool)
                np.save(os.path.join(OUT, f"phys_T_{part}_{src}_{var}.npy"), T)
                s = ph14.score(T, Y, L, idx); rows += [dict(var=var, part=part, src=src, k=k, **r) for r in s]
                print(var, part, src, "F1", [round(r["F1"], 3) for r in s], "tol1", [round(r["F1_tol1"], 3) for r in s],
                      "area", [round(r["area_ratio"], 2) for r in s], flush=True)
                pd.DataFrame(rows).round(4).to_csv(f_csv, index=False)
