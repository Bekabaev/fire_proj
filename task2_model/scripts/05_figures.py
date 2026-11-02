"""Задание 2, шаг 5. Рисунки: F1 моделей по горизонтам (тест 2023–2024 и заповедники), пример прогноза на 72 ч
(цвета как в FireSpread_Report_2: зелёный - совпадение, красный - лишний прогноз, жёлтый - пропуск, серый - уже выгорело,
светло-серый - нет достоверной разметки), обучение U-Net.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(__file__))
from common import *

NAMES = {"persist": "Начальная маска\n(без распространения)", "phys": "Физика (Ротермел)\nс калибровкой", "unet": "U-Net без погоды",
         "ml_coords": "ML с погодой\nи координатами", "ml_nocoords": "ML с погодой\nбез координат", "hybrid_nocoords": "Гибрид\nбез координат"}
ORDER = ["persist", "phys", "unet", "ml_coords", "ml_nocoords", "hybrid_nocoords"]

c = pd.read_csv(os.path.join(OUT, "comparison.csv"))
files = [("test", c)]
if os.path.exists(os.path.join(OUT, "reserves_comparison.csv")):
    r = pd.read_csv(os.path.join(OUT, "reserves_comparison.csv")); files.append(("reserves", r[r.group == r.group.iloc[0]]))
fig, axs = plt.subplots(1, len(files), figsize=(7.5 * len(files), 4.6), squeeze=False)
for ax, (nm, d) in zip(axs[0], files):
    w = 0.13
    for i, m in enumerate(ORDER):
        q = d[d.model == m].sort_values("horizon_h")
        if not len(q):
            continue
        x = np.arange(3) + (i - 2.5) * w
        ax.bar(x, q.F1, w, label=NAMES[m].replace("\n", " "))
        if "F1_lo" in q:
            ax.errorbar(x, q.F1, yerr=[q.F1 - q.F1_lo, q.F1_hi - q.F1], fmt="none", ecolor="k", lw=0.7, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["24 ч", "48 ч", "72 ч"]); ax.set_ylabel("F1 нового выгорания")
    ax.set_title("Тест: пожары Казахстана 2023–2024 (GFA)" if nm == "test" else "Экспертные контуры заповедников (2014–2023)", fontsize=10)
    ax.grid(axis="y", alpha=.3)
h_, l_ = axs[0][0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(os.path.join(FIG, "fig1_f1_by_model.png"), dpi=140); plt.close()

# пример прогноза на 72 ч: срез с наибольшим улучшением гибрида над U-Net среди срезов с приростом ≥ 2 км²
t = np.load(os.path.join(OUT, "test_preds.npz")); J = json.load(open(os.path.join(OUT, "comparison.json"))); th = J["thresholds"]
Y, V = t["Y"], t["V"]
def f1s(P, k):
    pr = (P[:, 2] >= k * 255) & V; yy = (Y[:, 2] > 0) & V
    tp = (pr & yy).sum((1, 2)); fp = (pr & ~yy).sum((1, 2)); fn = (~pr & yy).sum((1, 2))
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
fh, fu, fm = f1s(t["hybrid_nocoords"], th["hybrid_nocoords"]), f1s(t["unet"], th["unet"]), f1s(t["ml_nocoords"], th["ml_nocoords"])
grow = (Y[:, 2] > 0).sum((1, 2)) * 0.25
cand = np.flatnonzero(grow >= 2); i = cand[np.argmax(fh[cand] - fu[cand])] if len(cand) else 0
si = load_index(); row = si[(si.year == t["year"][i]) & (si.slice == t["slice"][i])].iloc[0]
X, _, _ = load_year(int(row.year)); st = X[row.slice, CH.index("burned_frac")] > 0.5
cm = ListedColormap(["white", "#dddddd", "#888888", "#2ca02c", "#d62728", "#ffcc00"])
act = st | (Y[i, 2] > 0) | np.any([t[k][i, 2] >= th[k] * 255 for k in ("unet", "ml_nocoords", "hybrid_nocoords")], 0)
rr_, cc_ = np.nonzero(act); pad = 6
r0_, r1_ = max(rr_.min() - pad, 0), min(rr_.max() + pad + 1, 96); c0_, c1_ = max(cc_.min() - pad, 0), min(cc_.max() + pad + 1, 96)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
for ax, (k, ttl, f) in zip(axs, [("unet", "U-Net без погоды", fu[i]), ("ml_nocoords", "ML с погодой без координат", fm[i]),
                                 ("hybrid_nocoords", "Гибрид без координат", fh[i])]):
    pr = t[k][i, 2] >= th[k] * 255; yy = Y[i, 2] > 0
    img = np.zeros((96, 96), int); img[~V[i]] = 1; img[st] = 2
    img[pr & yy & V[i]] = 3; img[pr & ~yy & V[i]] = 4; img[~pr & yy & V[i]] = 5
    ax.imshow(img[r0_:r1_, c0_:c1_], cmap=cm, vmin=0, vmax=5, interpolation="nearest"); ax.set_title(f"{ttl}: F1 = {f:.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Прогноз на 72 ч: {row.t_utc:%Y-%m-%d}, {row.lat:.3f}° с. ш., {row.lon:.3f}° в. д.; новая площадь по разметке {grow[i]:.2f} км²\n"
             f"фрагмент сцены {r1_ - r0_}×{c1_ - c0_} ячеек по 500 м; зелёный - совпадение, красный - лишний прогноз, жёлтый - пропуск, серый - уже выгорело, светло-серый - нет разметки", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_example_72h.png"), dpi=140); plt.close()

lg = pd.read_csv(os.path.join(OUT, "unet_train_log.csv"))
fig, ax = plt.subplots(figsize=(6, 3.6)); ax.plot(lg.epoch, lg.loss, label="потери (обучение)"); ax2 = ax.twinx()
ax2.plot(lg.epoch, lg.sel_F1, "C1", label="F1 calib_sel"); ax.set_xlabel("эпоха"); ax.set_ylabel("потери"); ax2.set_ylabel("F1")
fig.legend(loc="upper center", fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_unet_training.png"), dpi=140); plt.close()
print("figures saved; example slice", i, "fire", row.fire_id)
