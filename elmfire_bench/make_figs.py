"""Картинки сравнения ELMFIRE / гибрид / факт на 4 кейсах -> elmfire_bench/fig_elmfire_cases.png (тёмная тема слайдов).

  python elmfire_bench/make_figs.py            (после run_elmfire.py и evaluate.py)

Окно 48 x 48 км: старт - белый контур, факт 72 ч - красная заливка, гибрид 72 ч - синий контур,
ELMFIRE - изохроны 12/24/48/72 ч (жёлтый -> оранжевый).
"""
import csv, json, os
import numpy as np, rasterio
from scipy.ndimage import binary_fill_holes
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = (4, 21, 23, 16)
BG, FG, MUTED = "#1B1A18", "#F5F0E8", "#CFC6B8"
ISO = dict(zip((12, 24, 48, 72), ("#FFE08A", "#FDB85C", "#F58F3A", "#E4572E")))
f1 = {(int(r["case"]), r["model"]): r for r in csv.DictReader(open(os.path.join(HERE, "results.csv"), encoding="utf-8")) if r["h"] == "72"}

fig, axs = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=BG)
for ax, c in zip(axs, CASES):
    d = os.path.join(HERE, "cases", f"case_{c:02d}")
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    t, o = np.load(os.path.join(d, "truth.npz")), np.load(os.path.join(d, "ours.npz"))
    with rasterio.open(os.path.join(d, "elmfire_toa.tif")) as r:
        toa = r.read(1) / 3600.0
    with rasterio.open(os.path.join(d, "fbfm40.tif")) as r:
        fb = r.read(1)
    start = t["start"] > 0
    ax.set_facecolor(BG)
    ax.imshow(np.where(fb == 99, 1.0, np.nan), cmap="Greys", vmin=0, vmax=4, alpha=.5)   # негорючее
    ax.imshow(np.where(t["Y"][3] > 0, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1.3, alpha=.85)
    for h, col in ISO.items():                                    # без дырок от одиночных негорючих клеток
        ax.contour(binary_fill_holes((toa > 0) & (toa <= h)).astype(float), [.5], colors=col, linewidths=1.3)
    ax.contour(o["hybrid_mask"][3].astype(float), [.5], colors="#4C9BE8", linewidths=1.8)
    ax.contour(start.astype(float), [.5], colors=FG, linewidths=1.6)
    e, h = f1[(c, "elmfire")], f1[(c, "hybrid")]
    ax.set_title(f"{meta['t0_utc'][:10]}  ·  факт {float(e['true_km2']):.0f} км²", color=FG, fontsize=12)
    ax.text(.5, -.06, f"ELMFIRE {float(e['pred_km2']):.0f} км², F1 {float(e['F1']):.2f}\n"
                      f"гибрид {float(h['pred_km2']):.0f} км², F1 {float(h['F1']):.2f}".replace(".", ","),
            transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3A3833")
leg = [Line2D([], [], color=FG, lw=1.6, label="старт"), Patch(color="#D7263D", alpha=.85, label="факт за 72 ч"),
       Line2D([], [], color="#4C9BE8", lw=1.8, label="гибрид 72 ч")] + \
      [Line2D([], [], color=col, lw=1.3, label=f"ELMFIRE {h} ч") for h, col in ISO.items()]
fig.legend(handles=leg, loc="lower center", ncol=7, frameon=False, labelcolor=MUTED, fontsize=11, bbox_to_anchor=(.5, -.01))
fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.22, wspace=.04)
fig.savefig(os.path.join(HERE, "fig_elmfire_cases.png"), dpi=150, facecolor=BG)
print("готово: fig_elmfire_cases.png")
