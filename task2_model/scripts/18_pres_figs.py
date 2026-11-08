"""Задание 2, шаг 18. Рисунки для общей презентации: F1 моделей, чувствительность к ветру, пример уточнения меток по снимкам.
Выход: figures/pres/*.png."""
import os, sys, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import OUT, FIG, KZ, HZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task1_dataset", "scripts"))
from kz_common import GH, GW

FP = os.path.join(FIG, "pres"); os.makedirs(FP, exist_ok=True)
INK, EMBER, OCHRE, SAGE, GREY = "#2B2118", "#C8450E", "#E9C46A", "#5B8A45", "#9A9088"
plt.rcParams.update({"font.family": "Calibri", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# ---------- 1. F1 по горизонтам ----------
ev = pd.read_csv(os.path.join(OUT, "eval_all.csv"))
cmp_ = pd.read_csv(os.path.join(OUT, "comparison_v3.csv")); cmp_ = cmp_[cmp_.subset == "all"]
pb = pd.read_csv(os.path.join(OUT, "phys_baseline.csv")) if os.path.exists(os.path.join(OUT, "phys_baseline.csv")) else None


def series(part, src):
    s = {}
    r = cmp_[(cmp_.variant == "ring") & (cmp_.part == part)].sort_values("horizon_h")
    if len(r):
        s["расширение гари"] = (r.F1.values, GREY)
    if pb is not None:
        p = pb[(pb.part == part) & (pb.src == src)].sort_values("horizon_h")
        if len(p):
            s["физика FBP"] = (p.F1.values, OCHRE)
    for name, lab, col in ((f"wx_{src}", "U-Net с погодой", EMBER), (f"swxh_{src}", "сеть → скорость → решатель", SAGE),
                           (f"ens[wx_{src}+swx_{src}]", "ансамбль", INK)):
        e = ev[(ev.model == name) & (ev.part == part)].sort_values("horizon_h")
        if len(e):
            s[lab] = (e.F1.values, col)
    return s


fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, (part, src, ttl) in zip(axs, (("test", "gfsa", "Тест 2023–24, прогноз погоды GFS"), ("reserves", "era5", "Экспертные пожары заповедников"))):
    s = series(part, src); w = 0.8 / len(s); x = np.arange(4)
    for k, (lab, (v, col)) in enumerate(s.items()):
        b = ax.bar(x + (k - (len(s) - 1) / 2) * w, v, w * 0.92, color=col, label=lab)
        if k == len(s) - 1 or lab == "U-Net с погодой":
            ax.bar_label(b, fmt="%.2f", fontsize=9, padding=1)
    if part == "reserves":
        ax.axhline(0.58, color=EMBER, ls="--", lw=1); ax.text(3.45, 0.59, "потолок разметки ≈ 0,58", ha="right", fontsize=10, color=EMBER)
    ax.set_xticks(x, [f"{h} ч" for h in HZ]); ax.set_title(ttl, fontsize=13); ax.set_ylim(0, 0.65)
axs[0].set_ylabel("F1 (новая гарь)"); axs[0].legend(fontsize=10, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FP, "f1_models.png"), dpi=150); plt.close(fig)

# ---------- 2. чувствительность к ветру ----------
f = os.path.join(OUT, "sim_wind_sensitivity.csv")
if os.path.exists(f):
    d = pd.read_csv(f); d = d[d.wind != "calm"]
    s = d.groupby("model").cos.apply(lambda c: (c > 0.5).mean()).reindex(["physics", "wx", "swxh"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b = ax.barh(["физика FBP", "U-Net с погодой", "сеть → скорость →\nрешатель"], s.values, color=[OCHRE, EMBER, SAGE])
    ax.bar_label(b, labels=[f"{v:.0%}" for v in s.values], padding=3)
    ax.axvline(0.25, color=GREY, ls=":", lw=1); ax.text(0.26, 2.45, "случайно ≈ 25 %", fontsize=9, color=GREY)
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("доля прогнозов, смещённых по ветру (cos > 0,5)")
    fig.tight_layout(); fig.savefig(os.path.join(FP, "wind_sensitivity.png"), dpi=150); plt.close(fig)

# ---------- 3. пример уточнения меток по снимку ----------
q = pd.read_csv(os.path.join(KZ, "qa_labels_img.csv"))
q = q[(q.split == "test") & (q.reliable > 0) & q.cells.between(150, 2500)].assign(ch=lambda d: (d.moved_earlier + d.moved_later + d.added) / d.cells).sort_values("ch", ascending=False)
if len(q):
    fid = int(q.fire_id.iloc[0]); y = int(q.year.iloc[0])
    z = np.load(os.path.join(KZ, "imagery", f"{fid}.npz")); K, h, w = z["dn"].shape; r0, c0 = int(z["r0"]), int(z["c0"])
    a0 = np.load(os.path.join(KZ, "arrival", f"arr_{y}.npz")); a1 = np.load(os.path.join(KZ, "arrival_img", f"arr_{y}.npz"))

    def grid(a):
        A = np.full((h, w), np.nan); m = (a["fire_id"] == fid) & (a["r"] >= r0) & (a["r"] < r0 + h) & (a["c"] >= c0) & (a["c"] < c0 + w)
        A[a["r"][m] - r0, a["c"][m] - c0] = a["arr_h"][m]; S = np.zeros((h, w), int); S[a["r"][m] - r0, a["c"][m] - c0] = a["src"][m]
        return A, S
    A0, S0 = grid(a0); A1, S1 = grid(a1)
    k = int(np.argmax(z["reliable"])) if z["reliable"].any() else K // 2; tk = float(z["times_h"][k])
    t0 = np.nanmin(A0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    vmax = np.nanmax(A0) - t0
    im = ax[0].imshow(A0 - t0, cmap="inferno_r", vmin=0, vmax=vmax); ax[0].set_title("Было: время прихода огня\n(GFA + термоточки)", fontsize=12)
    ax[1].imshow(np.where(z["cf"][k] >= 50, z["dn"][k] / 1000, np.nan), cmap="RdYlGn_r", vmin=-0.1, vmax=0.4)
    ax[1].contour((A0 <= tk).astype(float), levels=[0.5], colors="k", linewidths=1)
    ax[1].set_title(f"Снимок {str(z['sensors'][k]).split(':')[0].capitalize()}, {tk - t0:.0f} ч: dNBR\n(чёрный контур - «сгорело» по старой метке)", fontsize=12)
    ax[2].imshow(A1 - t0, cmap="inferno_r", vmin=0, vmax=vmax)
    ch = (S1 >= 6) & (S0 != S1) | (S1 == 7); ax[2].contour(ch.astype(float), levels=[0.5], colors="#00b4d8", linewidths=1.2)
    ax[2].set_title("Стало: уточнено по снимкам\n(голубой - изменённые ячейки)", fontsize=12)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[2], shrink=0.8, label="ч от начала пожара")
    fig.savefig(os.path.join(FP, "imagery_labels.png"), dpi=150); plt.close(fig)
print("ok", os.listdir(FP))
