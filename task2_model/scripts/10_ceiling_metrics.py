"""Задание 2 v3, шаг 10. Потолок разметки и метрики с допуском.

1) Согласие двух независимых разметок одного пожара: экспертные контуры заповедников против меток GFA + VIIRS для тех же пожаров
   (gfa_fire_ids) в тот же момент t. F1 «одна разметка предсказывает другую» - оценка потолка: модель, обученная на GFA, не может
   систематически совпадать с экспертами лучше, чем совпадают сами метки.
2) Метрики с допуском для сохранённых прогнозов (out_v3/probs_*): F1 с буфером 1 и 2 ячейки, ошибка положения фронта (км, симметричное
   среднее расстояние между новыми выгоревшими ячейками прогноза и факта), ошибка площади, Brier и надёжность вероятностей.
Пороги - из comparison_v3.csv (подобраны на val 2022). Выход: out_v3/ceiling.csv, out_v3/tolerant_metrics.csv, out_v3/reliability.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *
from v3_loader import DIRS, load_index

KM = 0.5




# ---------- 1. потолок: эксперты против GFA ----------
def ceiling():
    rs = load_index(kind="reserves"); rs = rs[rs.gfa_fire_ids.notna()].sort_values("year")
    si = load_index(); si["key"] = si.fire_id.astype(str) + "|" + si.t_utc.astype(str)
    gidx = si.drop_duplicates("key").set_index("key")[["file", "slice", "r0", "c0"]]
    rows = []; cache = {}
    for _, r in rs.iterrows():
        cand = [k for k in (f"{int(x)}|{r.t_utc}" for x in str(r.gfa_fire_ids).split()) if k in gidx.index]
        if not cand:
            continue
        dr = np.load(os.path.join(KZ, DIRS["reserves"], r.file)); Yr = dr["Y"][r.slice] > 0; Vr = dr["valid"][r.slice]
        Yg = np.zeros_like(Yr); Vg = np.zeros_like(Vr); Lg = np.full(Vr.shape, INVALID, np.uint8)
        for k in cand:                                          # несколько пожаров GFA на один экспертный - объединяем
            g = gidx.loc[k]
            if g.file not in cache:
                cache.clear(); z = np.load(os.path.join(KZ, DIRS["500m"], g.file)); cache[g.file] = {k: z[k] for k in ("Y", "valid", "lsrc")}
            d = cache[g.file]; dy, dx = int(g.r0 - r.r0), int(g.c0 - r.c0)
            if abs(dy) >= 96 or abs(dx) >= 96:
                continue
            ys, xs = slice(max(dy, 0), min(96 + dy, 96)), slice(max(dx, 0), min(96 + dx, 96))
            yg, xg = slice(max(-dy, 0), min(96 - dy, 96)), slice(max(-dx, 0), min(96 - dx, 96))
            Yg[:, ys, xs] |= d["Y"][g.slice][:, yg, xg] > 0; Vg[ys, xs] |= d["valid"][g.slice][yg, xg]
            l = d["lsrc"][g.slice][yg, xg]; Lg[ys, xs] = np.where(l > 0, l, Lg[ys, xs])
        m = Vr & Vg
        for h, H_ in enumerate(HZ):
            a, b = Yr[h] & m, Yg[h] & m
            for tol in (0, 1, 2):
                if tol:
                    st = np.ones((2 * tol + 1,) * 2, bool)
                    tp_p = (b & binary_dilation(a, st)).sum(); tp_r = (a & binary_dilation(b, st)).sum()
                else:
                    tp_p = tp_r = (a & b).sum()
                rows.append(dict(event=r.event, t_utc=r.t_utc, horizon_h=H_, tol=tol, n_true=int(a.sum()), n_gfa=int(b.sum()),
                                 tp_p=int(tp_p), tp_r=int(tp_r)))
    t = pd.DataFrame(rows); out = []
    for (H_, tol), g in t.groupby(["horizon_h", "tol"]):
        P = g.tp_p.sum() / max(g.n_gfa.sum(), 1); R = g.tp_r.sum() / max(g.n_true.sum(), 1)
        out.append(dict(horizon_h=H_, tol_cells=tol, n_slices=len(g), n_events=g.event.nunique(), precision=P, recall=R,
                        F1=2 * P * R / max(P + R, 1e-9), area_ratio_gfa_to_expert=g.n_gfa.sum() / max(g.n_true.sum(), 1)))
    c = pd.DataFrame(out).round(4); c.to_csv(os.path.join(OUT, "ceiling.csv"), index=False)
    print("Согласие разметок (эксперты = факт, GFA+VIIRS = «прогноз»):\n", c.to_string(index=False), flush=True)
    return c


# ---------- 2. метрики с допуском ----------
def tolerant(part, variants, th, ring_r):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    M = np.load(os.path.join(CACHE, f"{part}_M.npy"), mmap_mode="r"); Y = np.load(os.path.join(CACHE, f"{part}_Y.npy"), mmap_mode="r")
    L = np.load(os.path.join(CACHE, f"{part}_L.npy"), mmap_mode="r")
    P = {v: np.load(os.path.join(OUT, f"probs_{v}_{part}.npy"), mmap_mode="r") for v in variants}
    ib = MAPS.index("burned_frac"); rows = []; rel = {v: np.zeros((len(HZ), 3, 20)) for v in variants}
    st = {t: np.ones((2 * t + 1,) * 2, bool) for t in (1, 2)}
    for i in range(len(si)):
        b = M[i, ib] > 0; l = L[i]; y = Y[i] > 0
        preds = {"ring": [binary_dilation(b, np.ones((2 * ring_r[h] + 1,) * 2, bool)) & ~b for h in range(len(HZ))]}
        for v in variants:
            pv = P[v][i].astype(np.float32) / 255
            preds[v] = [pv[h] >= th[v][h] for h in range(len(HZ))]
        for h, H_ in enumerate(HZ):
            m = (l != INVALID)
            if h == 0:
                m = m & ~(y[0] & (l == 4))
            yt = y[h] & m
            for v in variants:
                pv = P[v][i, h].astype(np.float32)[m] / 255; yy = yt[m]
                k = np.minimum((pv * 20).astype(int), 19)
                rel[v][h, 0] += np.bincount(k, minlength=20); rel[v][h, 1] += np.bincount(k, pv, 20); rel[v][h, 2] += np.bincount(k, yy, 20)
            dt_true = distance_transform_edt(~(yt | b)) if yt.any() else None
            for v, pr in preds.items():
                pp = pr[h] & m
                r = dict(i=i, fire=si.fire_id.iat[i] if "fire_id" in si else si.event.iat[i], variant=v, horizon_h=H_,
                         n_true=int(yt.sum()), n_pred=int(pp.sum()), tp=int((pp & yt).sum()))
                for t in (1, 2):
                    r[f"tp_p{t}"] = int((pp & binary_dilation(yt, st[t])).sum()); r[f"tp_r{t}"] = int((yt & binary_dilation(pp, st[t])).sum())
                if yt.any() and pp.any():                       # ошибка фронта: среднее расстояние новых ячеек до ближайших ячеек другой карты
                    dt_pred = distance_transform_edt(~(pp | b))
                    r["front_km"] = 0.5 * KM * (dt_pred[yt].mean() + dt_true[pp].mean())
                rows.append(r)
        if i % 1000 == 0:
            print(f"  {part} {i}/{len(si)}", flush=True)
    t = pd.DataFrame(rows); out = []
    for (v, H_), g in t.groupby(["variant", "horizon_h"]):
        r = dict(part=part, variant=v, horizon_h=H_)
        for tol in (0, 1, 2):
            tpp, tpr = (g.tp, g.tp) if tol == 0 else (g[f"tp_p{tol}"], g[f"tp_r{tol}"])
            Pp = tpp.sum() / max(g.n_pred.sum(), 1); R = tpr.sum() / max(g.n_true.sum(), 1)
            r[f"F1_tol{tol}"] = 2 * Pp * R / max(Pp + R, 1e-9)
        has = g[g.n_true > 0]
        r["front_km_median"] = has.front_km.median()
        r["missed_slices"] = float(((has.n_pred == 0)).mean())                    # рост был, прогноз пуст
        r["area_err_km2_median"] = float(np.median(np.abs(g.n_pred - g.n_true)) * 0.25)
        r["area_ratio_total"] = g.n_pred.sum() / max(g.n_true.sum(), 1)
        big = g[g.n_true * 0.25 >= 10]                                             # срезы с ростом ≥ 10 км²
        Pp = big.tp.sum() / max(big.n_pred.sum(), 1); R = big.tp.sum() / max(big.n_true.sum(), 1)
        r["F1_big10"] = 2 * Pp * R / max(Pp + R, 1e-9); r["recall_big10"] = R; r["n_big10"] = len(big)
        out.append(r)
    rel_rows = []
    for v in variants:
        for h, H_ in enumerate(HZ):
            n, sp, sy = rel[v][h]
            brier_parts = None
            for k in range(20):
                if n[k]:
                    rel_rows.append(dict(part=part, variant=v, horizon_h=H_, bin=k, n=int(n[k]), p_mean=sp[k] / n[k], obs=sy[k] / n[k]))
            ece = np.nansum(n * np.abs(sp / np.maximum(n, 1) - sy / np.maximum(n, 1))) / n.sum()
            for r in out:
                if r["variant"] == v and r["horizon_h"] == H_:
                    r["ECE"] = ece
    return pd.DataFrame(out), pd.DataFrame(rel_rows), t


if __name__ == "__main__":
    c = ceiling()
    cmp_ = pd.read_csv(os.path.join(OUT, "comparison_v3.csv")); cmp_ = cmp_[(cmp_.subset == "all")]
    th = {v: [float(cmp_[(cmp_.variant == v) & (cmp_.part == "test") & (cmp_.horizon_h == H_)].threshold.iloc[0]) for H_ in HZ]
          for v in ("nowx", "wx_era5", "wx_gfsa")}
    ring_r = [int(cmp_[(cmp_.variant == "ring") & (cmp_.part == "test") & (cmp_.horizon_h == H_)].threshold.iloc[0]) for H_ in HZ]
    A, R = [], []
    for part, vs in (("test", ["nowx", "wx_era5", "wx_gfsa"]), ("reserves", ["nowx", "wx_era5"])):
        a, r, t = tolerant(part, vs, th, ring_r); A.append(a); R.append(r)
        t.to_csv(os.path.join(OUT, f"tolerant_slices_{part}.csv.gz"), index=False)
    A = pd.concat(A).round(4); A.to_csv(os.path.join(OUT, "tolerant_metrics.csv"), index=False)
    pd.concat(R).round(4).to_csv(os.path.join(OUT, "reliability.csv"), index=False)
    pd.set_option("display.width", 250)
    print(A.to_string(index=False))
