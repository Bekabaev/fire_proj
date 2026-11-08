"""Картинки для presentation_final2.pptx (стиль слайдов: фон F7F3EC, Arial).

  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_final2_figs.py      (из project_till_nov)

Выход: presentation/final2_figs/
  ai_vs_phys.png      F1 по горизонтам: Ротермел, FBP, U-Net, гибрид (тест 2023–24) и заповедники
  elm_bars.png        F1 по горизонтам на 40 пожарах: ELMFIRE, ELMFIRE x0,05, FBP, гибрид
  elm_scatter.png     площадь 72 ч прогноз / факт по 40 пожарам (лог), гибрид vs ELMFIRE vs ELMFIRE x0,05
  elm_time.png        два пожара по часам 12/24/48/72: факт, ELMFIRE, ELMFIRE x0,05, гибрид

Источники чисел (все — F1 по всем клеткам вместе, новая гарь):
  тест 2023–24, прогноз GFS, 8 059 срезов: U-Net и гибрид — ITERATIONS.md «Итог итераций 6–8»;
  FBP — CHECKPOINT_task2_model.md §10.8; Ротермел (FBFM40, k по val) — ITERATIONS.md итерация 9 (out_v3/fuel_phys.csv);
  заповедники (эксперты, ERA5): гибрид — итог итераций 6–8, FBP — §10.8, Ротермел — итерация 9;
  40 пожаров сайта: elmfire_bench/results.csv.
"""
import csv, os
import numpy as np, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "presentation", "final2_figs"); os.makedirs(OUT, exist_ok=True)
EB = os.path.join(ROOT, "elmfire_bench")
BG, INK, MUTED, GRID = "#F7F3EC", "#1B1A18", "#4F4A43", "#E4DDD1"
C = dict(roth="#A39A8C", fbp="#E9A37A", unet="#3E7CB1", hyb="#C2501E", elm="#7A6F62", adj="#8A5BC2", fact="#D7263D")
HZ = (12, 24, 48, 72)
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG, "figure.facecolor": BG})
dec = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")


def bars(ax, series, ymax, label_fs=10.5):
    n = len(series); w = 0.8 / n
    for j, (name, col, vals) in enumerate(series):
        xs = np.arange(4) + (j - (n - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.012, dec(v), ha="center", va="bottom", fontsize=label_fs, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} ч" for h in HZ], fontsize=13, color=INK)
    ax.set_ylim(0, ymax); ax.grid(axis="y", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ---------- 1. ИИ против физики ----------
TEST = [("Ротермел (физика США)", C["roth"], [0.034, 0.065, 0.080, 0.083]),
        ("FBP O-1b (физика Канады)", C["fbp"], [0.050, 0.080, 0.106, 0.117]),
        ("U-Net (ИИ)", C["unet"], [0.264, 0.251, 0.295, 0.311]),
        ("Гибрид ИИ + физика", C["hyb"], [0.238, 0.239, 0.276, 0.300])]
RES = [("Ротермел", C["roth"], [0.048, 0.133, 0.173, 0.190]),
       ("FBP O-1b", C["fbp"], [0.113, 0.225, 0.259, 0.276]),
       ("Гибрид", C["hyb"], [0.288, 0.397, 0.436, 0.467])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
bars(a1, TEST, 0.36, 9.5); a1.set_title("Тест 2023–2024 · 8 059 ситуаций · прогноз погоды GFS", loc="left", color=INK, fontsize=14, pad=10)
bars(a2, RES, 0.66, 9.5); a2.set_title("Экспертные пожары заповедников · 393 ситуации", loc="left", color=INK, fontsize=14, pad=10)
for k, c in enumerate((0.48, 0.55, 0.59, 0.58)):                 # потолок разметки: совпадение экспертов со спутниковой гарью
    a2.plot([k - .4, k + .4], [c, c], color=INK, lw=1.4, ls="--", zorder=4)
a2.text(-.4, 0.495, "потолок разметки", fontsize=10, color=INK, va="bottom")
a1.set_ylabel("F1 (по всем клеткам)")
h = [Patch(color=c, label=n) for n, c, _ in TEST]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(.5, -.01))
fig.subplots_adjust(left=.06, right=.99, top=.9, bottom=.2, wspace=.12)
fig.savefig(os.path.join(OUT, "ai_vs_phys.png"), dpi=200); plt.close(fig)

# ---------- 2. ELMFIRE на 40 пожарах ----------
rows = list(csv.DictReader(open(os.path.join(EB, "results.csv"), encoding="utf-8")))
pooled = {}
for l in open(os.path.join(EB, "results.md"), encoding="utf-8").read().splitlines()[2:]:
    c = [x.strip() for x in l.strip("|").split("|")]
    pooled.setdefault(c[0], []).append(float(c[3]))
ELM = [("ELMFIRE как есть", C["elm"], pooled["elmfire"]), ("ELMFIRE ×0,05", C["adj"], pooled["elmfire_adj0.05"]),
       ("наша физика FBP", C["fbp"], pooled["fbp"]), ("Гибрид ИИ + физика", C["hyb"], pooled["hybrid"])]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
bars(ax, ELM, 0.62, 9.5); ax.set_ylabel("F1 (40 пожаров вместе)")
ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in ELM], loc="upper left", frameon=False, fontsize=11.5)
fig.subplots_adjust(left=.1, right=.99, top=.97, bottom=.1)
fig.savefig(os.path.join(OUT, "elm_bars.png"), dpi=200); plt.close(fig)

# площадь прогноз/факт
r72 = {(int(r["case"]), r["model"]): r for r in rows if r["h"] == "72"}
fig, ax = plt.subplots(figsize=(6.2, 4.6))
lo, hi = 0.2, 3000
ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
for m, col, lab, mk in (("elmfire", C["elm"], "ELMFIRE как есть", "o"), ("elmfire_adj0.05", C["adj"], "ELMFIRE ×0,05", "s"),
                        ("hybrid", C["hyb"], "Гибрид", "D")):
    xs = [max(float(r72[(i, "hybrid")]["true_km2"]), lo) for i in range(40) if (i, m) in r72]
    ys = [max(float(r72[(i, m)]["pred_km2"]), lo) for i in range(40) if (i, m) in r72]
    ax.scatter(xs, ys, s=34, color=col, marker=mk, label=lab, zorder=3, edgecolor=BG, linewidth=.6)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("реально сгорело за 72 ч, км²"); ax.set_ylabel("прогноз, км²")
ax.grid(color=GRID, zorder=0); [ax.spines[s].set_visible(False) for s in ("top", "right")]
ax.text(400, 0.5, "идеал —\nна диагонали", color=MUTED, fontsize=10, ha="center")
ax.legend(loc="upper left", frameon=False, fontsize=11)
fig.subplots_adjust(left=.14, right=.98, top=.97, bottom=.14)
fig.savefig(os.path.join(OUT, "elm_scatter.png"), dpi=200); plt.close(fig)


# ---------- 3. два пожара по часам ----------
def rd(p):
    with rasterio.open(p) as r:
        a = r.read(1).astype(float); nod = r.nodata
    if nod is not None:
        a[a == nod] = -1
    return a


fig, axs = plt.subplots(2, 4, figsize=(14, 7.8))
for row, case in enumerate((4, 21)):
    d = os.path.join(EB, "cases", f"case_{case:02d}")
    t, o = np.load(os.path.join(d, "truth.npz")), np.load(os.path.join(d, "ours.npz"))
    toa, toa2 = rd(os.path.join(d, "elmfire_toa.tif")) / 3600, rd(os.path.join(d, "elmfire_adj0.05_toa.tif")) / 3600
    with rasterio.open(os.path.join(d, "fbfm40.tif")) as r:
        nb = r.read(1) == 99
    start = t["start"] > 0
    for k, h in enumerate(HZ):
        ax = axs[row, k]
        img = np.ones((96, 96, 3)) * np.array([0xFB, 0xF8, 0xF2]) / 255
        img[nb] = np.array([0xD9, 0xD3, 0xC8]) / 255
        img[(toa > 0) & (toa <= h) & ~start] = np.array([0xE6, 0xDC, 0xCC]) / 255 * .55 + np.array([0x7A, 0x6F, 0x62]) / 255 * .45
        img[(t["Y"][k] > 0) & ~start] = np.array([0xD7, 0x26, 0x3D]) / 255
        ax.imshow(img, interpolation="nearest")
        ax.contour(((toa2 > 0) & (toa2 <= h) | start).astype(float), [.5], colors=C["adj"], linewidths=1.6, linestyles="--")
        ax.contour((o["hybrid_mask"][k].astype(bool) | start).astype(float), [.5], colors=C["unet"], linewidths=2)
        ax.contour(start.astype(float), [.5], colors=INK, linewidths=1.6)
        e, a, hy = (next(r for r in rows if r["case"] == str(case) and r["model"] == m and r["h"] == str(h))
                    for m in ("elmfire", "elmfire_adj0.05", "hybrid"))
        ax.set_title(f"{h} ч · факт {float(hy['true_km2']):.0f} км²", fontsize=13, color=INK, loc="left")
        ax.text(0, -.04, f"ELMFIRE {float(e['pred_km2']):.0f} · ×0,05 {float(a['pred_km2']):.0f} · гибрид {float(hy['pred_km2']):.0f} км²\n"
                f"F1: {dec(float(e['F1']))} · {dec(float(a['F1']))} · {dec(float(hy['F1']))}",
                transform=ax.transAxes, va="top", fontsize=10.5, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_color(GRID) for s in ax.spines.values()]
    axs[row, 0].set_ylabel(("3 апреля 2023" if case == 4 else "27 июля 2023"), fontsize=14, color=INK, labelpad=8)
leg = [Patch(color=C["fact"], label="реально сгорело"), Patch(color="#B5AA9A", label="ELMFIRE как есть"),
       Line2D([], [], color=C["adj"], ls="--", lw=1.8, label="ELMFIRE ×0,05"), Line2D([], [], color=C["unet"], lw=2, label="гибрид"),
       Line2D([], [], color=INK, lw=1.6, label="гарь на старте"), Patch(color="#D9D3C8", label="не горит (вода, пашня, город)")]
fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False, fontsize=12, bbox_to_anchor=(.5, 0))
fig.subplots_adjust(left=.03, right=.995, top=.95, bottom=.13, wspace=.05, hspace=.42)
fig.savefig(os.path.join(OUT, "elm_time.png"), dpi=170); plt.close(fig)
print("готово:", OUT)


# ---------- 4. один пожар, 72 ч: факт / ELMFIRE / гибрид (без текста, подписи - на слайде) ----------
def fire_panels(case, out):
    d = os.path.join(EB, "cases", f"case_{case:02d}")
    t, o = np.load(os.path.join(d, "truth.npz")), np.load(os.path.join(d, "ours.npz"))
    toa = rd(os.path.join(d, "elmfire_toa.tif")) / 3600
    with rasterio.open(os.path.join(d, "fbfm40.tif")) as r:
        nb = r.read(1) == 99
    start = t["start"] > 0
    fact = (t["Y"][3] > 0) & ~start
    pred = {"fact": fact, "elm": (toa > 0) & (toa <= 72) & ~start, "hyb": o["hybrid_mask"][3].astype(bool) & ~start}
    col = {"fact": "#D7263D", "elm": "#8C8377", "hyb": "#3E7CB1"}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k in zip(axs, ("fact", "elm", "hyb")):
        img = np.ones((96, 96, 3)) * np.array([0xFF, 0xFD, 0xF9]) / 255
        img[nb] = np.array([0xE4, 0xDD, 0xD1]) / 255
        img[pred[k]] = np.array(matplotlib.colors.to_rgb(col[k]))
        img[start] = np.array([0x1B, 0x1A, 0x18]) / 255
        ax.imshow(img, interpolation="nearest")
        if k != "fact":
            ax.contour(fact.astype(float), [.5], colors=C["fact"], linewidths=1.6)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.08)
    fig.savefig(os.path.join(OUT, out), dpi=200, facecolor="#F7F3EC"); plt.close(fig)


fire_panels(4, "fire_case4.png")
print("fire_case4.png")


# ---------- 5. для отчёта: те же графики, что в презентации (белый фон, подписи на рисунке) ----------
GL, GD, HY = "#CFC6B8", "#8C8377", "#C2501E"


def rep_bars(ax, series, ymax, title):
    n = len(series); w = 0.8 / n
    for j, (name, col, vals) in enumerate(series):
        xs = np.arange(4) + (j - (n - 1) / 2) * w
        ax.bar(xs, vals, w * 0.9, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + ymax * 0.015, dec(v), ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} ч" for h in HZ], fontsize=12, color=INK)
    ax.set_ylim(0, ymax); ax.set_yticks([]); ax.set_facecolor("white")
    for sname in ("top", "right", "left"):
        ax.spines[sname].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=26)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=n, frameon=False, fontsize=10.5, handlelength=1, columnspacing=1.2)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
rep_bars(a1, [("Ротермел", GD, [0.034, 0.065, 0.080, 0.083]), ("Гибрид ИИ + физика", HY, [0.238, 0.239, 0.276, 0.300])], 0.34,
         "Тест 2023–2024 (959 пожаров): ×3,6 на 72 ч")
rep_bars(a2, [("ELMFIRE как есть", GL, pooled["elmfire"]), ("ELMFIRE ×0,05", GD, pooled["elmfire_adj0.05"]),
              ("Гибрид ИИ + физика", HY, pooled["hybrid"])], 0.62, "40 тестовых пожаров: ×2,9 на 72 ч")
fig.subplots_adjust(left=.01, right=.99, top=.84, bottom=.08, wspace=.08)
fig.savefig(os.path.join(OUT, "rep_compare.png"), dpi=200, facecolor="white"); plt.close(fig)

d4 = os.path.join(EB, "cases", "case_04")
t4, o4 = np.load(os.path.join(d4, "truth.npz")), np.load(os.path.join(d4, "ours.npz"))
toa4 = rd(os.path.join(d4, "elmfire_toa.tif")) / 3600
with rasterio.open(os.path.join(d4, "fbfm40.tif")) as r:
    nb4 = r.read(1) == 99
st4 = t4["start"] > 0; fact4 = (t4["Y"][3] > 0) & ~st4
panels = (("Реально сгорело", "23 км² за 72 ч", fact4, "#D7263D"),
          ("ELMFIRE", "1 584 км² · F1 0,03", (toa4 > 0) & (toa4 <= 72) & ~st4, GD),
          ("Гибрид", "22 км² · F1 0,69", o4["hybrid_mask"][3].astype(bool) & ~st4, "#3E7CB1"))
fig, axs = plt.subplots(1, 3, figsize=(13, 5.0), facecolor="white")
for ax, (name, sub, m, col) in zip(axs, panels):
    img = np.ones((96, 96, 3)) * np.array([0xFF, 0xFD, 0xF9]) / 255
    img[nb4] = np.array([0xE4, 0xDD, 0xD1]) / 255
    img[m] = np.array(matplotlib.colors.to_rgb(col)); img[st4] = np.array([0x1B, 0x1A, 0x18]) / 255
    ax.imshow(img, interpolation="nearest")
    if name != "Реально сгорело":
        ax.contour(fact4.astype(float), [.5], colors="#D7263D", linewidths=1.5)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(GRID) for sp in ax.spines.values()]
    ax.set_title(name, loc="left", fontsize=14, fontweight="bold", color=INK)
    ax.text(0, -0.05, sub, transform=ax.transAxes, va="top", fontsize=12.5, color=col)
fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.09, wspace=.06)
fig.savefig(os.path.join(OUT, "rep_fire.png"), dpi=200, facecolor="white"); plt.close(fig)
print("rep_compare.png, rep_fire.png")
