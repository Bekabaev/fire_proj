"""Задание 2, шаг 3. Погодная поправка (GBM) поверх U-Net, гибрид с физикой и сравнение моделей как в FireSpread_Report_2.

Признаки ячейки: вероятности U-Net (24/48/72), локальные карты (термоточки, топливо, NDVI, прошлые гари, рельеф),
геометрия относительно текущего пожара (расстояние до кромки, смещение вдоль/поперёк ветра), погода ERA5 и сезон;
вариант «с координатами» добавляет широту/долготу; гибрид добавляет физику (время прихода огня по Ротермелу при
калиброванном множителе, достижимость). GBM: HistGradientBoosting, 140 итераций, до 15 листьев (как в отчёте).
Обучение поправки - calib_fit (2021–2022, 70 % пожаров), выбор порога и множителя физики - calib_sel, тест - 2023–2024.
Итоговая оценка = 0.5·U-Net + 0.5·поправка, накопление по времени (p48 ≥ p24, p72 ≥ p48), единый порог.
Выход: out/comparison.csv, out/comparison.json, out/gbm_*.pkl, out/test_preds_<model>.npz
"""
import os, sys, json, pickle, time
import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from common import *

MULTS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
WEATHER = ["T_mean_{h}h", "T_max_{h}h", "RH_min_{h}h", "VPD_mean_{h}h", "ws_mean_{h}h", "gust_max_{h}h", "P_sum_{h}h"]
SC_COMMON = ["P_sum_prev7d", "VPD_mean_prev7d", "soilm_t", "RH_min_prev24h"]


def cell_features(X, row, up, phys_T, mult, h_i, coords, phys):
    """Признаки всех ячеек одного среза для горизонта h_i -> [9216, F]."""
    h = HZ[h_i]; g = lambda k: X[CH.index(k)].astype(np.float32)
    state = g("burned_frac") > 0.5; af = g("af_count_24h") > 0
    d_state, ind = ndimage.distance_transform_edt(~state, return_indices=True) if state.any() else (np.full((96, 96), 99.0), None)
    d_af = ndimage.distance_transform_edt(~af) if af.any() else np.full((96, 96), 99.0)
    rr, cc = np.mgrid[0:96, 0:96]
    u, v = row.get(f"wind_u_{h}h", np.nan), row.get(f"wind_v_{h}h", np.nan)
    if ind is not None and np.isfinite(u):
        dx = (cc - ind[1]).astype(np.float32); dy = -(rr - ind[0]).astype(np.float32)     # восток / север, ячейки
        ws = np.hypot(u, v) + 1e-6; along = (dx * u + dy * v) / ws; cross = np.abs(dx * v - dy * u) / ws
    else:
        along = cross = np.full((96, 96), np.nan, np.float32)
    F = [up[0], up[1], up[2], up[h_i], np.minimum(d_state, 99), np.minimum(d_af, 99), along, cross,
         np.log1p(g("af_count_24h")), np.log1p(g("af_count_all")), g("prior_burn"), g("grass_frac"), g("crop_frac"),
         g("shrub_tree_frac"), g("wetland_frac"), g("nonburnable_frac"), g("ndvi"), g("slope_deg"), g("dist_ign_km"),
         ndimage.uniform_filter(state.astype(np.float32), 5), ndimage.uniform_filter(g("af_count_24h"), 5)]
    names = ["unet24", "unet48", "unet72", "unet_h", "d_state", "d_af", "along_wind", "cross_wind", "af24", "af_all", "prior_burn",
             "grass", "crop", "shrub_tree", "wetland", "nonburnable", "ndvi", "slope", "dist_ign", "state_5x5", "af_5x5"]
    sc = {k.format(h=h): row.get(k.format(h=h), np.nan) for k in WEATHER}
    sc.update({k: row.get(k, np.nan) for k in SC_COMMON})
    sc.update(doy_sin=np.sin(2 * np.pi * row["doy"] / 365), doy_cos=np.cos(2 * np.pi * row["doy"] / 365), k_day=row["k_day"],
              log_state=np.log1p(row["state_km2"]), log_af=np.log1p(row["af_24h"]))
    if coords:
        sc.update(lat=row["lat"], lon=row["lon"])
    if phys:
        T = phys_T[h_i].astype(np.float32) / mult
        F += [np.log1p(np.minimum(T, 1e4)), (T <= h).astype(np.float32), np.isfinite(T).astype(np.float32)]
        names += ["phys_logT", "phys_reach_h", "phys_finite"]
    A = np.stack([f.ravel() for f in F] + [np.full(96 * 96, float(vv), np.float32) for vv in sc.values()], 1).astype(np.float32)
    return A, names + [k.replace(f"_{h}h", "_h") for k in sc]


class Data:
    def __init__(s):
        s.si = load_index(); s.cache = {}

    def year(s, y):
        if y not in s.cache:
            X, Y, V = load_year(y)
            up = np.load(os.path.join(OUT, f"unet_probs_{y}.npy"))
            pf = os.path.join(OUT, f"phys_T_{y}.npy")
            pt = np.load(pf) if os.path.exists(pf) else None
            s.cache = {y: (X, Y, V, up, pt)}
        return s.cache[y]

    def rows(s, part):
        return s.si[s.si.part == part].sort_values(["year", "slice"])


def counts(pred, Y, V):
    yy = (Y > 0) & V[:, None]; pr = pred & V[:, None]
    return (pr & yy).sum((2, 3)), (pr & ~yy).sum((2, 3)), (~pr & yy).sum((2, 3))   # [S,3]


def main():
    D = Data(); t00 = time.time()
    weather_ok = D.si.weather_ok.fillna(False).astype(bool) if "weather_ok" in D.si else pd.Series(True, index=D.si.index)
    D.si = D.si[weather_ok | (D.si.part == "unet_train")]
    # ---------- 1) калибровка множителя физики на calib_sel (только физика, как ELMFIRE в отчёте)
    sel = D.rows("calib_sel"); res = {}
    for m in MULTS:
        tp = fp = fn = 0
        for y, s in sel.groupby("year"):
            X, Y, V, up, pt = D.year(y)
            k = s.slice.values; P = (pt[k].astype(np.float32) / m <= np.array(HZ)[None, :, None, None])
            a, b, c = counts(P, Y[k], V[k]); tp += a[:, 2].sum(); fp += b[:, 2].sum(); fn += c[:, 2].sum()
        res[m] = 2 * tp / max(2 * tp + fp + fn, 1)
    m_best = max(res, key=res.get); print("physics multiplier F1@72h on calib_sel:", {k: round(v, 4) for k, v in res.items()}, "->", m_best, flush=True)
    # ---------- 2) обучение GBM-поправок
    rng = np.random.default_rng(0)
    variants = {"ml_coords": (True, False), "ml_nocoords": (False, False), "hybrid_nocoords": (False, True)}
    models = {}
    fit = D.rows("calib_fit")
    for vn, (coords, phys) in variants.items():
        models[vn] = []
        for hi in range(3):
            A_, y_, w_ = [], [], []
            for y, s in fit.groupby("year"):
                X, Y, V, up, pt = D.year(y)
                for r in s.itertuples():
                    A, names = cell_features(X[r.slice], r._asdict(), up[r.slice].astype(np.float32) / 255, pt[r.slice] if pt is not None else None,
                                             m_best, hi, coords, phys)
                    t = (Y[r.slice, hi] > 0).ravel(); v = V[r.slice].ravel()
                    near = A[:, 4] <= 30                                      # до 15 км от кромки
                    pk = np.where(t, 1.0, np.where(near, 0.25, 0.02)) * v
                    take = rng.random(len(pk)) < pk
                    A_.append(A[take]); y_.append(t[take]); w_.append(1 / pk[take])
            A_ = np.concatenate(A_); y_ = np.concatenate(y_); w_ = np.concatenate(w_)
            clf = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, learning_rate=0.1, random_state=0)
            clf.fit(A_, y_, sample_weight=w_ / w_.mean())
            models[vn].append(clf)
            print(f"  {vn} h={HZ[hi]}: rows {len(y_)}, pos {int(y_.sum())}, features {A_.shape[1]} ({time.time() - t00:.0f} s)", flush=True)
        pickle.dump(dict(models=models[vn], names=names, coords=coords, phys=phys, mult=m_best), open(os.path.join(OUT, f"gbm_{vn}.pkl"), "wb"))

    # ---------- 3) прогноз на calib_sel (порог) и test
    def predict_part(part):
        rows = D.rows(part); out = {k: [] for k in ["unet", "phys", "persist"] + list(variants)}; Ys, Vs = [], []
        for y, s in rows.groupby("year"):
            X, Y, V, up, pt = D.year(y)
            for r in s.itertuples():
                u = up[r.slice].astype(np.float32) / 255; out["unet"].append(u)
                out["phys"].append((pt[r.slice].astype(np.float32) / m_best <= np.array(HZ)[:, None, None]).astype(np.float32))
                st = X[r.slice, CH.index("burned_frac")] > 0.5; af = X[r.slice, CH.index("af_count_24h")] > 0
                pm = ndimage.binary_dilation(st | af, np.ones((3, 3), bool)) & ~st          # начальная расширенная маска
                out["persist"].append(np.repeat(pm[None].astype(np.float32), 3, 0))
                for vn, (coords, phys) in variants.items():
                    p = []
                    for hi in range(3):
                        A, _ = cell_features(X[r.slice], r._asdict(), u, pt[r.slice] if pt is not None else None, m_best, hi, coords, phys)
                        p.append(models[vn][hi].predict_proba(A)[:, 1].reshape(96, 96))
                    p = 0.5 * u + 0.5 * np.stack(p)
                    out[vn].append(np.maximum.accumulate(p, 0))                   # вложенность по времени
                Ys.append(Y[r.slice]); Vs.append(V[r.slice])
        return rows, {k: np.stack(v) for k, v in out.items()}, np.stack(Ys), np.stack(Vs)

    rs, Ps, Ys, Vs = predict_part("calib_sel")
    ths = {}
    for k, P in Ps.items():
        if k in ("phys", "persist"):
            ths[k] = 0.5; continue
        best = (-1, 0.5)
        for t in np.linspace(0.05, 0.9, 69):
            a, b, c = counts(P >= t, Ys, Vs); f = np.mean([2 * a[:, i].sum() / max(2 * a[:, i].sum() + b[:, i].sum() + c[:, i].sum(), 1) for i in range(3)])
            best = max(best, (f, t))
        ths[k] = best[1]
    print("thresholds (calib_sel):", {k: round(v, 4) for k, v in ths.items()}, flush=True)
    rt, Pt, Yt, Vt = predict_part("test")
    rows = []; per_slice = {}
    for k, P in Pt.items():
        a, b, c = counts(P >= ths[k], Yt, Vt); per_slice[k] = (a, b, c)
        for i, h in enumerate(HZ):
            m = prf(a[:, i].sum(), b[:, i].sum(), c[:, i].sum()); lo, hi_ = f1_boot(a[:, i], b[:, i], c[:, i], rt.fire_id.values)
            rows.append(dict(model=k, horizon_h=h, **{kk: round(vv, 4) for kk, vv in m.items()}, F1_lo=round(lo, 4), F1_hi=round(hi_, 4), threshold=ths[k]))
    cmp_ = pd.DataFrame(rows); cmp_.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    # парное сравнение гибрида и ML без координат (72 ч), бутстрэп по пожарам
    g = pd.Series(rt.fire_id.values).astype("category").cat.codes.values; G = g.max() + 1; rng = np.random.default_rng(1); dd = []
    agg = {k: [np.bincount(g, per_slice[k][j][:, 2], G) for j in range(3)] for k in ("hybrid_nocoords", "ml_nocoords")}
    for _ in range(1000):
        kk = rng.integers(0, G, G)
        f = {k: 2 * v[0][kk].sum() / max(2 * v[0][kk].sum() + v[1][kk].sum() + v[2][kk].sum(), 1) for k, v in agg.items()}
        dd.append(f["hybrid_nocoords"] - f["ml_nocoords"])
    summary = dict(test_slices=len(rt), test_fires=int(rt.fire_id.nunique()), physics_multiplier=m_best, physics_sel_F1_72h=res,
                   thresholds=ths, hybrid_minus_ml_nocoords_72h=dict(mean=float(np.mean(dd)), ci95=[float(x) for x in np.percentile(dd, [2.5, 97.5])]),
                   new_burn_share_of_valid=float((Yt[:, 0] > 0).sum() / Vt.sum()),
                   median_growth_cells={h: float(np.median((Yt[:, i] > 0).sum((1, 2)))) for i, h in enumerate(HZ)})
    json.dump(summary, open(os.path.join(OUT, "comparison.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, "test_preds.npz"), **{k: (v * 255).astype(np.uint8) for k, v in Pt.items()}, Y=Yt, V=Vt,
                        fire_id=rt.fire_id.values, year=rt.year.values, slice=rt.slice.values)
    print(cmp_.pivot(index="model", columns="horizon_h", values="F1").round(3).to_string())
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
