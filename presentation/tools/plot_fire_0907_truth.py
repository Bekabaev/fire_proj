"""Пожар 7.09.2026: реальная гарь по снимку Sentinel-2 (dNBR > 0,2, снимок 11.09 ~ 82 ч) на подложке со скриншота fires.kz,
сверху - гибрид (старт + прогноз 72 ч) и моделирование fires.kz.
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/plot_fire_0907_truth.py   (из project_till_nov; нужны make_fireskz_0907.py, s2_dnbr.npz)
Выход: presentation/fireskz_0907/truth_s2.png
"""
import os, math, runpy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "presentation", "fireskz_0907")
g = runpy.run_path(os.path.join(ROOT, "presentation", "tools", "make_fireskz_0907.py"))
B, Z, SC, CX, CY, MX0, MY0 = g["B"], g["Z"], g["SC"], g["CX"], g["CY"], g["MX0"], g["MY0"]
burned, hyb, kz_lay = g["burned"], g["hyb"], g["lay"]
cx, cy, half = g["FRAME_ALL"]

z = np.load(os.path.join(D, "s2_dnbr.npz")); dnbr, rgb = z["dnbr"], z["rgb_post"]; W, S, E, N, RES = z["extent"]
# пиксели скриншота -> lon/lat -> пиксель dNBR
H_, W_ = B.shape[:2]
yy, xx = np.mgrid[0:H_, 0:W_]
wx = MX0 + (xx - CX) / SC; wy = MY0 + (yy - CY) / SC; s = 256 * 2 ** Z
lon = wx / s * 360 - 180
lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * wy / s))))
ix = ((lon - W) / RES).astype(int); iy = ((N - lat) / RES).astype(int)
ok = (ix >= 0) & (ix < dnbr.shape[1]) & (iy >= 0) & (iy < dnbr.shape[0])
d = np.full((H_, W_), np.nan); d[ok] = dnbr[iy[ok], ix[ok]]
bright = np.full((H_, W_), False); bright[ok] = np.nanmean(rgb, -1)[iy[ok], ix[ok]] > 0.25        # облака на снимке - не гарь
burn = (np.nan_to_num(d) > 0.2) & ~bright
burn = ndimage.binary_opening(burn, iterations=1); burn = ndimage.binary_closing(burn, iterations=3); burn = ndimage.binary_fill_holes(burn)
lab, k = ndimage.label(burn); sz = ndimage.sum(burn, lab, range(1, k + 1))
px_km2 = (156543.03 * math.cos(math.radians(g["CLAT"])) / 2 ** Z / SC / 1000) ** 2
burn = np.isin(lab, 1 + np.nonzero(sz * px_km2 >= 0.1)[0])              # без мелких пятен < 0,1 км²

# гарь по снимку -> клетки 500 м нашей сетки: клетка сгорела, если ≥ 30 % из 5 × 5 точек внутри неё - гарь
PX, PY, n = g["PX"], g["PY"], g["n"]
t = np.linspace(0.1, 0.9, 5)
s2c = np.zeros((n, n), bool)
for r in range(n):
    for c in range(n):
        sx = PX[r, c] + (PX[r, c + 1] - PX[r, c]) * t[:, None] + (PX[r + 1, c] - PX[r, c]) * t[None, :]
        sy = PY[r, c] + (PY[r, c + 1] - PY[r, c]) * t[:, None] + (PY[r + 1, c] - PY[r, c]) * t[None, :]
        ix_, iy_ = np.round(sx).astype(int), np.round(sy).astype(int)
        okk = (ix_ >= 0) & (ix_ < W_) & (iy_ >= 0) & (iy_ < H_)
        if okk.any() and burn[iy_[okk], ix_[okk]].mean() >= 0.3:
            s2c[r, c] = True
fact = s2c & ~burned
# только наш пожар: пятна гари, которые (с допуском 3 клетки ~ 1,5 км) связаны с гарью на старте; соседний пожар на западе - убираем
grp, _ = ndimage.label(ndimage.binary_dilation(fact | burned, iterations=3), structure=np.ones((3, 3)))
fact = fact & np.isin(grp, np.unique(grp[burned]))
f1 = g["f1"]; km2 = g["km2"]
print("S2 fact km2", km2(fact), "F1 hybrid", f1(hyb[3], fact), "U-Net", f1(g["unet"][3], fact), "fires.kz", f1(g["kz_new"], fact),
      "VIIRS", f1(g["Y"][3], fact))
fig, ax = g["frame"](B, size=7.0)
g["cells"](ax, fact, fc="#e4572e", ec="none", alpha=0.85)
ax.contour(np.arange(W_), np.arange(H_), kz_lay.astype(float), levels=[0.5], colors=["#FFD23F"], linewidths=2.2, linestyles="--")
g["cells"](ax, burned, fc="#3a3a3a", ec="none", alpha=0.95)
g["outline"](ax, hyb[3] | burned, "#2E86AB", 2.6)
g["outline"](ax, hyb[3] | burned, "white", 0.9, (0, (3, 3)))
ax.legend(handles=[Patch(fc="#e4572e", label="реальная гарь по снимку Sentinel-2 (11.09), клетки 500 м"),
                   Patch(fc="#3a3a3a", label="гарь на старте (VIIRS)"),
                   Line2D([], [], c="#2E86AB", lw=2.6, label="гибрид: старт + прогноз 72 ч"),
                   Line2D([], [], c="#FFD23F", lw=2.2, ls="--", label="fires.kz")],
          loc="upper left", fontsize=10.5, framealpha=0.88)
ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
fig.savefig(os.path.join(D, "truth_s2.png"), facecolor="black"); plt.close(fig)
print("burn on screen km2", round(float(burn.sum()) * px_km2, 1))
np.save(os.path.join(D, "fact_s2_cells.npy"), fact)
