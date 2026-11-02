"""Задание 2 v3, шаг 12. Единая оценка моделей: строгий F1, F1 с допуском 1–2 ячейки, калибровка вероятностей.

Модели: nowx, wx (FrontV3, шаг 7), swx / swx_res / snowx (SolverNet, шаг 11) - какие обучены. Погода на тесте: ERA5 и прогноз GFS.
Калибровка: изотоническая регрессия по val 2022 (на каждом горизонте) - вероятность ячейки становится частотой выгорания.
Порог - по F1 на val (после калибровки порядок не меняется, F1 тот же). Надёжность: ECE по ячейкам с p ≥ 0,05, Brier skill
относительно доли выгорания на val. Выход: out_v3/eval_all.csv, out_v3/eval_reliability.csv, out_v3/probs_<вариант>_<часть>.npy.
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from scipy.ndimage import binary_dilation
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *
from common_solver import SolverNet

dev = "cuda" if torch.cuda.is_available() else "cpu"
ST = {t: np.ones((2 * t + 1,) * 2, bool) for t in (1, 2)}


def load_part(part):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    return si, *(np.load(os.path.join(CACHE, f"{part}_{k}.npy"), mmap_mode="r") for k in "MYL")


def load_model(run):
    ck = torch.load(os.path.join(OUT, f"{run}_best.pt"), map_location=dev, weights_only=False)
    if run.startswith("s"):
        m = SolverNet(ck["cin"], ck["ns"], K=ck.get("K", 48), res=ck.get("res", False), wind_pos=ck.get("wind_pos"), hard=ck.get("hard", False))
    else:
        m = FrontV3(ck["cin"], ck["ns"])
    m.to(dev).load_state_dict(ck["state"]); m.eval()
    return m, ck


def predict(model, M, Z, bs=48):
    P = np.zeros((len(M), len(HZ), 96, 96), np.uint8)
    with torch.no_grad():
        for i in range(0, len(M), bs):
            x = torch.from_numpy(M[i:i + bs].astype(np.float32)).to(dev)
            P[i:i + bs] = (torch.sigmoid(model(x, torch.from_numpy(Z[i:i + bs]).to(dev))) * 255).round().byte().cpu().numpy()
    return P


def eval_mask(Y, L, h):
    m = L != INVALID
    if h == 0:
        m = m & ~((Y[:, 0] > 0) & (L == 4))
    return m


def fit_calib(P, Y, L, rng):
    """Изотоническая калибровка и порог по val на каждом горизонте (по гистограмме 256 уровней вероятности)."""
    out = []
    for h in range(len(HZ)):
        m = eval_mask(Y, L, h); p = P[:, h][m]; y = Y[:, h][m] > 0
        n = np.bincount(p, minlength=256); k = np.bincount(p, y, minlength=256)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.arange(256)[n > 0], k[n > 0] / n[n > 0], sample_weight=n[n > 0])
        lut = iso.predict(np.arange(256))
        pos = k[::-1].cumsum()[::-1]; neg = (n - k)[::-1].cumsum()[::-1]       # p >= level
        f1 = 2 * pos / np.maximum(2 * pos + neg + (k.sum() - pos), 1); lev = int(np.argmax(f1[1:]) + 1)
        out.append(dict(lut=lut, level=lev, base=k.sum() / n.sum()))
    return out


def score(P, Y, L, si, cal, ib_maps):
    rows = []; groups = si.event.values if "event" in si else si.fire_id.values
    rel = np.zeros((len(HZ), 3, 20))
    for h in range(len(HZ)):
        lev, lut = cal[h]["level"], cal[h]["lut"]
        tp = np.zeros(len(P)); fp = np.zeros(len(P)); fn = np.zeros(len(P)); tpp = np.zeros((2, len(P))); tpr = np.zeros((2, len(P)))
        bs = 0.0; bs0 = 0.0; ncell = 0
        for i in range(len(P)):
            m = eval_mask(Y[i:i + 1], L[i:i + 1], h)[0]; y = (Y[i, h] > 0) & m; pr = (P[i, h] >= lev) & m
            tp[i] = (pr & y).sum(); fp[i] = (pr & ~y).sum(); fn[i] = (~pr & y).sum()
            for j, t in enumerate((1, 2)):
                tpp[j, i] = (pr & binary_dilation(y, ST[t])).sum(); tpr[j, i] = (y & binary_dilation(pr, ST[t])).sum()
            q = lut[P[i, h][m]]; yy = y[m]
            bs += ((q - yy) ** 2).sum(); bs0 += ((cal[h]["base"] - yy) ** 2).sum(); ncell += m.sum()
            b = np.minimum((q * 20).astype(int), 19)
            rel[h, 0] += np.bincount(b, minlength=20); rel[h, 1] += np.bincount(b, q, 20); rel[h, 2] += np.bincount(b, yy, 20)
        T, Fp, Fn = tp.sum(), fp.sum(), fn.sum(); lo, hi = f1_boot(tp, fp, fn, groups)
        r = dict(horizon_h=HZ[h], F1=2 * T / max(2 * T + Fp + Fn, 1), lo=lo, hi=hi, precision=T / max(T + Fp, 1), recall=T / max(T + Fn, 1))
        for j, t in enumerate((1, 2)):
            Pp = tpp[j].sum() / max(T + Fp, 1); R = tpr[j].sum() / max(T + Fn, 1); r[f"F1_tol{t}"] = 2 * Pp * R / max(Pp + R, 1e-9)
        big = (tp + fn) * 0.25 >= 10
        r["recall_big10"] = tp[big].sum() / max((tp + fn)[big].sum(), 1)
        r["area_ratio"] = (T + Fp) / max(T + Fn, 1)
        r["BSS"] = 1 - bs / max(bs0, 1e-9)
        n, sp, sy = rel[h]; k = n > 0; kk = k & (np.arange(20) >= 1)
        r["ECE_p05"] = float(np.sum(np.abs(sp[kk] / n[kk] - sy[kk] / n[kk]) * n[kk]) / max(n[kk].sum(), 1))
        rows.append(r)
    return rows, rel


if __name__ == "__main__":
    stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
    parts = {p: load_part(p) for p in ("val", "test", "reserves")}
    runs = [r for r in ("nowx", "wx", "wxbig", "snowx", "swx", "swxh", "swx_res") if os.path.exists(os.path.join(OUT, f"{r}_best.pt"))]
    rows, rels = [], []; rng = np.random.default_rng(0)
    KEEP = {}; ENS = [("wx_gfsa", "swx_gfsa"), ("wx_era5", "swx_era5"), ("wxbig_gfsa", "swx_gfsa"), ("wx_gfsa", "swxh_gfsa"), ("wx_era5", "swxh_era5")]
    need = {n for e in ENS for n in e}

    def evaluate(name, Pv, get_part):
        cal = fit_calib(Pv, parts["val"][2], parts["val"][3], rng)
        print(name, "val levels", [c["level"] for c in cal], flush=True)
        for part in ("test", "reserves"):
            si, M, Y, L = parts[part]
            if "gfsa" in name and part == "reserves":
                continue
            P = get_part(part)
            if name in need:
                KEEP[(name, part)] = P
            rr, rel = score(P, Y, L, si, cal, None)
            for r in rr:
                rows.append(dict(model=name, part=part, **r))
            for h in range(len(HZ)):
                for b in range(20):
                    if rel[h, 0, b]:
                        rels.append(dict(model=name, part=part, horizon_h=HZ[h], bin=b, n=int(rel[h, 0, b]),
                                         p=rel[h, 1, b] / rel[h, 0, b], obs=rel[h, 2, b] / rel[h, 0, b]))
            print(f"  {part:8s} " + " ".join(f"{r['horizon_h']}h {r['F1']:.3f}/{r['F1_tol1']:.3f} BSS {r['BSS']:.3f}" for r in rr), flush=True)
        pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "eval_all.csv"), index=False)

    for run in runs:
        model, ck = load_model(run)
        srcs = ["era5", "gfsa"] if ck.get("wx") else [None]
        for src in srcs:
            name = run + ("" if src is None else f"_{src}")
            Zs = {p: (np.c_[base_scalars(v[0]), weather_vec(v[0], stats, src)] if src else base_scalars(v[0])).astype(np.float32)
                  for p, v in parts.items()}
            Pv = predict(model, parts["val"][1], Zs["val"])
            if name in need:
                KEEP[(name, "val")] = Pv

            def get_part(part, name=name, model=model, Zs=Zs, run=run):
                f = os.path.join(OUT, f"probs_{name}_{part}.npy")
                P = predict(model, parts[part][1], Zs[part])        # всегда из текущего чекпойнта - калибровка и тест от одной модели
                np.save(f, P); return P
            evaluate(name, Pv, get_part)
    for a, b in ENS:                                            # ансамбль: среднее вероятностей двух моделей
        if (a, "val") in KEEP and (b, "val") in KEEP:
            mix = lambda part: ((KEEP[(a, part)].astype(np.uint16) + KEEP[(b, part)]) // 2).astype(np.uint8)
            evaluate(f"ens[{a}+{b}]", mix("val"), mix)
    pd.DataFrame(rels).round(4).to_csv(os.path.join(OUT, "eval_reliability.csv"), index=False)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.pivot_table(index=["part", "model"], columns="horizon_h", values=["F1", "F1_tol1", "BSS", "ECE_p05"]).round(3).to_string())
