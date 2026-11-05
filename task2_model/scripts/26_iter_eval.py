"""Задание 2, итерации улучшения (ITERATIONS.md), шаг 26. Единая оценка варианта: U-Net (или ансамбль) -> голова -> метрики.

ITER=1 NAME=tta UNET=wx_tta HEAD=unet,hybrid,gbm python 26_iter_eval.py
  UNET - префиксы файлов вероятностей out_v3/probs_<префикс>_<src>_<часть>.npy через запятую (несколько - среднее, ансамбль);
  HEAD - unet: изотоническая калибровка и порог по val 2022 (как шаг 12);
         hybrid: логистический стекинг U-Net + время прихода по физике (как прод, шаг 20), веса по val;
         gbm: градиентный бустинг по ячейкам (U-Net, физика, расстояния до гари и термоточек, положение вдоль/поперёк ветра,
              топливо, погода окна) - аналог погодной поправки из отчёта Jafar, плюс физика. Обучение на val 2022 по 2 фолдам
              пожаров (cross-fit): прогноз вне фолда -> калибровка и порог, на тест - среднее двух моделей.
Оценка: тест 2023–24 с прогнозом GFS (как на сайте), заповедники - ERA5; порог всегда с val (тест не участвует в выборе).
Выход: дописывает строки в out_v3/iter_eval.csv (ключ iter+name+head заменяется), вероятности гибридов
out_v3/probs_<name>_<head>_<src>_<часть>.npy для следующих итераций.
"""
import os, sys, json, time, importlib.util
import numpy as np, pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *


def _mod(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f)); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


ev12 = _mod("ev12", "12_eval_all.py"); ev20 = _mod("ev20", "20_prod_hybrid.py")
ITER = os.environ.get("ITER", "0"); NAME = os.environ.get("NAME", "base")
UNET = os.environ.get("UNET", "wx").split(","); HEADS = os.environ.get("HEAD", "unet").split(",")
JOBS = {"test": "gfsa", "reserves": "era5"}
FCSV = os.path.join(OUT, "iter_eval.csv")
R_CAND = (12, 12, 12, 12)                                       # кандидаты: U-Net > 0 или ≤ 12 ячеек от гари/термоточек (вне - 0)
NEG = 0.04
LIMIT = int(os.environ.get("SMOKE", 0)) or None              # SMOKE=300 - быстрая проверка кода на первых срезах
STATIC = ["af_count_all", "prior_burn", "grass_frac", "crop_frac", "shrub_tree_frac", "nonburnable_frac", "slope_deg",
          "ndvi", "ndvi_anom", "dist_road_km", "road_density", "dist_settlement_km"]
GB_NAMES = (["logit_unet", "p_unet_5x5", "p_unet_max5", "log_T_h", "unreach", "dist_burn", "dist_af", "burn_dens7", "along", "cross"]
            + STATIC + ["ws", "T_max", "RH_min", "P_sum", "FFMC", "log_state", "k_12h", "log_af12", "hour12", "doy_s", "doy_c"])


def probs(prefix, src, part):
    return np.load(os.path.join(OUT, f"probs_{prefix}_{src}_{part}.npy"), mmap_mode="r")[:LIMIT]


def unet_probs(src, part):
    if len(UNET) == 1:
        return np.asarray(probs(UNET[0], src, part))
    acc = None
    for u in UNET:
        a = np.asarray(probs(u, src, part), np.uint16); acc = a if acc is None else acc + a
    return np.round(acc / len(UNET)).astype(np.uint8)


def physT(part, src):
    return np.load(os.path.join(OUT, f"phys_T_{part}_{src}.npy"), mmap_mode="r")[:LIMIT]


# ---------- признаки бустинга ----------
def slice_geo(M):
    """M [n,28,96,96] -> расстояние до гари и до термоточек (ячейки), плотность гари 7×7, центр гари."""
    b = np.asarray(M[:, MAPS.index("burned_frac")], np.float32) > 0
    a = np.asarray(M[:, MAPS.index("af_count_24h")], np.float32) > 0
    n = len(M); db = np.full((n, 96, 96), 60, np.float32); da = np.full((n, 96, 96), 60, np.float32); cen = np.full((n, 2), 47.5)
    for i in range(n):
        src = b[i] | a[i]
        if b[i].any():
            db[i] = np.minimum(distance_transform_edt(~b[i]), 60)
        if a[i].any():
            da[i] = np.minimum(distance_transform_edt(~a[i]), 60)
        if src.any():
            yy, xx = np.nonzero(src); cen[i] = (yy.mean(), xx.mean())
    dens = uniform_filter(b.astype(np.float32), (1, 7, 7))
    return db, da, dens, cen


def wind_to(si, src, h):
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    u = -np.nan_to_num(si[f"{pre}wind_u_{HZ[h]}h"].values); v = -np.nan_to_num(si[f"{pre}wind_v_{HZ[h]}h"].values)
    return u, v


def scal(si, src, h):
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]; g = lambda c: si.get(c, pd.Series(np.nan, index=si.index)).values.astype(np.float32)
    d = pd.to_datetime(si.t_utc).dt.dayofyear.values
    u, v = wind_to(si, src, h)
    return np.c_[np.hypot(u, v), g(f"{pre}T_max_{HZ[h]}h"), g(f"{pre}RH_min_{HZ[h]}h"), np.log1p(np.maximum(g(f"{pre}P_sum_{HZ[h]}h"), 0)),
                 g("gfwed_FFMC"), np.log1p(si.state_km2.values), np.minimum(si.k_12h.values, 30), np.log1p(si.af_12h.values),
                 (si.hour_utc.values == 12), np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)].astype(np.float32)


YY, XX = np.mgrid[0:96, 0:96].astype(np.float32)


def gb_chunk(P, T, M, si, src, h, geo):
    """-> признаки [n,96,96,F] f32 и маска кандидатов [n,96,96]."""
    db, da, dens, cen = geo; n = len(P)
    p = np.clip(np.asarray(P[:, h], np.float32) / 255, 0.5 / 255, 254.5 / 255); lp = np.log(p / (1 - p))
    p5 = uniform_filter(p, (1, 5, 5)); from scipy.ndimage import maximum_filter; pm = maximum_filter(p, (1, 5, 5))
    t = np.asarray(T, np.float32); far = (~np.isfinite(t)) | (t >= ev20.TMAX); lt = np.log(np.clip(np.where(far, ev20.TMAX, t), 0.5, ev20.TMAX) / HZ[h])
    u, v = wind_to(si, src, h); ws = np.hypot(u, v) + 1e-6; ux, uy = u / ws, v / ws          # восток, север
    dx = XX[None] - cen[:, 1, None, None]; dy = -(YY[None] - cen[:, 0, None, None])
    along = dx * ux[:, None, None] + dy * uy[:, None, None]; cross = np.abs(-dx * uy[:, None, None] + dy * ux[:, None, None])
    st = [np.asarray(M[:, MAPS.index(k)], np.float32) for k in STATIC]
    sc = scal(si, src, h)
    F_ = np.stack([lp, p5, pm, lt, far.astype(np.float32), db, da, dens, along, cross] + st
                  + [np.broadcast_to(sc[:, j, None, None], (n, 96, 96)) for j in range(sc.shape[1])], -1)
    cand = (np.minimum(db, da) <= R_CAND[h]) | (np.asarray(P[:, h]) >= 1)        # U-Net > 0 или рядом с гарью/термоточками
    return F_, cand


def gb_iter(P, T, M, si, src, idx, fn):
    """Проход по срезам кусками по 192: fn(h, c, F, cand)."""
    for c in np.array_split(idx, max(1, len(idx) // 192)):
        c = np.sort(c); sic = si.iloc[c].reset_index(drop=True); Mc = np.asarray(M[c]); geo = slice_geo(Mc)
        for h in range(len(HZ)):
            F_, cand = gb_chunk(P[c], T[c], Mc, sic, src, h, geo); fn(h, c, F_, cand)


def gb_fit(P, T, M, Y, L, si, src, idx, rng):
    Xs = [[] for _ in HZ]; ys = [[] for _ in HZ]; ws = [[] for _ in HZ]; cov = np.zeros((len(HZ), 2))

    def take(h, c, F_, cand):
        m = ev12.eval_mask(Y[c], L[c], h); y = (Y[c, h] > 0) & m
        cov[h] += (y & cand).sum(), y.sum()
        keep = (y & cand) | (m & cand & ~y & (rng.random(m.shape) < NEG))
        Xs[h].append(F_[keep]); ys[h].append(y[keep]); ws[h].append(np.where(y[keep], 1.0, 1 / NEG))
    gb_iter(P, T, M, si, src, idx, take)
    print("  candidate area keeps positives:", (cov[:, 0] / np.maximum(cov[:, 1], 1)).round(3).tolist(), flush=True)
    out = []
    for h in range(len(HZ)):
        X = np.concatenate(Xs[h]); y = np.concatenate(ys[h]); w = np.concatenate(ws[h])
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=10,
                                           early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0)
        out.append(m.fit(X, y, sample_weight=w)); print(f"  gbm {HZ[h]}h: {len(y)} cells, {int(y.sum())} pos, {m.n_iter_} iter", flush=True)
    return out


def gb_apply(models_list, P, T, M, si, src, idx=None):
    idx = np.arange(len(P)) if idx is None else idx; out = np.zeros((len(idx), len(HZ), 96, 96), np.uint8); pos = {v: k for k, v in enumerate(idx)}

    def put(h, c, F_, cand):
        q = np.zeros(cand.shape, np.float32); X = F_[cand]
        q[cand] = np.mean([ms[h].predict_proba(X)[:, 1] for ms in models_list], 0)
        out[[pos[i] for i in c], h] = np.round(q * 255).astype(np.uint8)
    gb_iter(P, T, M, si, src, idx, put)
    return out


# ---------- оценка и запись ----------
def score_rows(head, src, part, P, cal, si, M, Y, L):
    rr, _ = ev12.score(P, Y, L, si, cal, None); rows = []
    wc = {h: ev20.wind_cos(P, Y, M, si, src, h) for h in (1, 3)}
    for r in rr:
        h = HZ.index(r["horizon_h"]); w = wc.get(h, {})
        rows.append(dict(iter=ITER, name=NAME, head=head, unet="+".join(UNET), src=src, part=part, **{k: round(float(x), 4) for k, x in r.items()},
                         downwind=round(w.get("downwind", np.nan), 3), wind_cos=round(w.get("cos", np.nan), 3)))
    print(f"[{ITER} {NAME} {head}] {src} {part:8s} " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}[{r['lo']:.3f},{r['hi']:.3f}] A {r['area_ratio']:.2f}" for r in rr)
          + f" | downwind 24h {wc[1]['downwind']:.2f} 72h {wc[3]['downwind']:.2f}", flush=True)
    return rows


def save_rows(rows):
    if LIMIT:
        return
    t = pd.read_csv(FCSV) if os.path.exists(FCSV) else pd.DataFrame()
    if len(t):
        t = t[~((t["iter"].astype(str) == str(ITER)) & (t["name"] == NAME) & (t["head"].isin({r["head"] for r in rows})))]
    pd.concat([t, pd.DataFrame(rows)], ignore_index=True).to_csv(FCSV, index=False)


if __name__ == "__main__":
    rng = np.random.default_rng(0); t0 = time.time()
    parts = {p: tuple(a.iloc[:LIMIT].reset_index(drop=True) if k == 0 else a[:LIMIT] for k, a in enumerate(ev12.load_part(p)))
             for p in ("val", "test", "reserves")}
    si_v, M_v, Y_v, L_v = parts["val"]
    fit_src = {s: (s if all(os.path.exists(os.path.join(OUT, f"probs_{u}_{s}_val.npy")) for u in UNET) else "gfsa") for s in ("gfsa", "era5")}
    for head in HEADS:
        rows = []
        if head == "unet":
            cal = {s: ev12.fit_calib(unet_probs(fit_src[s], "val"), Y_v, L_v, rng) for s in set(JOBS.values())}
            rows += score_rows(head, "gfsa", "val", unet_probs("gfsa", "val"), cal["gfsa"], si_v, M_v, Y_v, L_v)   # val: порог по нему же (для выбора)
            for part, src in JOBS.items():
                si, M, Y, L = parts[part]; rows += score_rows(head, src, part, unet_probs(src, part), cal[src], si, M, Y, L)
        elif head == "hybrid":
            for part, src in JOBS.items():
                fs = fit_src[src]; Pv = unet_probs(fs, "val"); Tv = physT("val", fs)
                ms = ev20.fit(ev20.sample(Pv, Tv, Y_v, L_v, si_v, fs, np.arange(len(si_v)), rng))
                cal = ev12.fit_calib(ev20.apply(ms, Pv, Tv, si_v, fs), Y_v, L_v, rng)
                si, M, Y, L = parts[part]; P = ev20.apply(ms, unet_probs(src, part), physT(part, src), si, src)
                np.save(os.path.join(OUT, f"probs_{NAME}_hybrid_{src}_{part}.npy"), P)
                rows += score_rows(head, src, part, P, cal, si, M, Y, L)
                json.dump({f"{h}h": dict(coef=m.coef_[0].round(4).tolist(), intercept=round(float(m.intercept_[0]), 4)) for h, m in zip(HZ, ms)},
                          open(os.path.join(OUT, f"iter_hybrid_{NAME}_{fs}.json"), "w"))
        elif head == "gbm":
            done = {}
            for part, src in JOBS.items():
                fs = fit_src[src]
                if fs not in done:
                    Pv = unet_probs(fs, "val"); Tv = physT("val", fs)
                    fid = pd.Series(si_v.fire_id.values).astype("category").cat.codes.values; fold = ((fid.astype(np.int64) * 7919) % 97) % 2
                    oof = np.zeros(Pv.shape, np.uint8); fm = []
                    for k in (0, 1):
                        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
                        print(f" gbm fold {k} ({fs}): fit {len(tr)} slices", flush=True)
                        ms = gb_fit(Pv, Tv, M_v, Y_v, L_v, si_v, fs, tr, rng); fm.append(ms)
                        oof[te] = gb_apply([ms], Pv, Tv, M_v, si_v, fs, te)
                    cal = ev12.fit_calib(oof, Y_v, L_v, rng); done[fs] = (fm, cal)
                    rows += score_rows("gbm_val_oof", fs, "val", oof, cal, si_v, M_v, Y_v, L_v)
                    imp = {}
                    print(f" gbm ({fs}) ready, {time.time() - t0:.0f} s", flush=True)
                fm, cal = done[fs]; si, M, Y, L = parts[part]
                P = gb_apply(fm, unet_probs(src, part), physT(part, src), M, si, src)
                LIMIT or np.save(os.path.join(OUT, f"probs_{NAME}_gbm_{src}_{part}.npy"), P)
                rows += score_rows(head, src, part, P, cal, si, M, Y, L)
        save_rows(rows); print(f"{head} done, {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)
