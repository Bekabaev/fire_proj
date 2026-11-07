"""Оценка ELMFIRE против нашей модели на 40 тестовых пожарах (те же метрики, что на сайте и в офлайн-тесте).

  python3 elmfire_bench/evaluate.py                 (из корня репозитория; нужны numpy, rasterio, scipy)

Модели:
  hybrid  - наш прод-гибрид (ours.npz: hybrid_mask)
  fbp     - наша физика FBP (ours.npz: fbp_arrival_h <= H)
  elmfire - ELMFIRE: case_XX/elmfire_toa.tif = время прихода огня в СЕКУНДАХ от t0 на сетке кейса (96 x 96, 500 м);
            не сгорело / нет данных -> <= 0 или nodata. Нет файла -> кейс пропускается для ELMFIRE.
Цель: новая гарь за 12/24/48/72 ч (без стартовой); на 12 ч ячейки с меткой только по дате MCD64A1 (lsrc 4) не считаются.
Метрики: F1 по всем кейсам вместе (суммы TP/FP/FN, как pooled_F1 сайта), медиана F1 по кейсам, F1 с допуском 1 км
(2 клетки), площадь прогноза / факт. Итог: elmfire_bench/results.csv и results.md.
"""
import csv, glob, json, os
import numpy as np, rasterio
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
HZ = (12, 24, 48, 72)
TOL = np.ones((5, 5), bool)                                     # допуск 2 клетки = 1 км


def score_mask(valid, lsrc, Y12, k):
    return valid & (((lsrc != 4) | (Y12 == 0)) if k == 0 else True)


def counts(p, y, m):
    p, y = p & m, y & m
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    tpt_p = int((p & binary_dilation(y, TOL)).sum())             # точность с допуском: прогноз в 1 км от факта
    tpt_r = int((y & binary_dilation(p, TOL)).sum())             # полнота с допуском
    return np.array([tp, fp, fn, tpt_p, int(p.sum()), tpt_r, int(y.sum())])


def f1(c):
    return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def f1_tol(c):
    pr, rc = c[3] / max(c[4], 1), c[5] / max(c[6], 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


def models(d, start):
    o = np.load(os.path.join(d, "ours.npz"))
    out = {"hybrid": [o["hybrid_mask"][k] & ~start for k in range(4)],
           "fbp": [(o["fbp_arrival_h"] <= h) & ~start for h in HZ]}
    f = os.path.join(d, "elmfire_toa.tif")
    if os.path.exists(f):
        with rasterio.open(f) as r:
            toa = r.read(1).astype(np.float64); nod = r.nodata
        if nod is not None:
            toa[toa == nod] = -1
        burned = np.isfinite(toa) & (toa > 0)
        out["elmfire"] = [burned & (toa <= h * 3600) & ~start for h in HZ]
    return out


rows, pooled = [], {}
for d in sorted(glob.glob(os.path.join(HERE, "cases", "case_*"))):
    t = np.load(os.path.join(d, "truth.npz")); meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    Y, valid, lsrc, start = t["Y"] > 0, t["valid"], t["lsrc"], t["start"]
    for name, P in models(d, start).items():
        for k, h in enumerate(HZ):
            c = counts(P[k], Y[k], score_mask(valid, lsrc, Y[0], k))
            pooled[(name, h)] = pooled.get((name, h), 0) + c
            rows.append(dict(case=meta["case"], title=meta["title"], model=name, h=h, F1=round(f1(c), 3), F1_tol1km=round(f1_tol(c), 3),
                             pred_km2=c[4] * 0.25, true_km2=c[6] * 0.25))

with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
names = [m for m in ("hybrid", "fbp", "elmfire") if (m, 12) in pooled]
n_cases = {m: len({r["case"] for r in rows if r["model"] == m}) for m in names}
lines = ["| Модель | Кейсов | Горизонт | F1 (все вместе) | F1 медиана | F1 с допуском 1 км | Площадь прогноз / факт |", "|---|---|---|---|---|---|---|"]
for m in names:
    for h in HZ:
        c = pooled[(m, h)]; med = np.median([r["F1"] for r in rows if r["model"] == m and r["h"] == h])
        lines.append(f"| {m} | {n_cases[m]} | {h} ч | {f1(c):.3f} | {med:.3f} | {f1_tol(c):.3f} | {c[4] / max(c[6], 1):.2f} |")
open(os.path.join(HERE, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
