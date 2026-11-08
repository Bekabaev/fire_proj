"""Сравнение на карте dNBR Sentinel-2 (check_fire_0907_s2.py): гарь по снимку vs факт VIIRS, fires.kz, гибрид, U-Net (пожар 7.09.2026).
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/plot_fire_0907_s2.py   (из project_till_nov; нужен make_fireskz_0907.py)
Выход: presentation/fireskz_0907/s2_check.png, s2_cells.json (доли гари по снимку в клетках каждой маски).
"""
import os, sys, json, runpy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from rasterio.warp import transform as warp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "presentation", "fireskz_0907")
g = runpy.run_path(os.path.join(ROOT, "presentation", "tools", "make_fireskz_0907.py"))      # маски на нашей сетке
burned, Y, hyb, unet, kz, n, r0, c0 = g["burned"], g["Y"], g["hyb"], g["unet"], g["kz_new"], g["n"], g["r0"], g["c0"]
CRS, CELL, X0, Y1 = g["CRS"], g["CELL"], g["X0"], g["Y1"]
z = np.load(os.path.join(D, "s2_dnbr.npz")); dnbr, rgb = z["dnbr"], z["rgb_post"]; W, S, E, N, RES = z["extent"]

rr, cc = np.mgrid[0:n + 1, 0:n + 1]
lon, lat = warp(CRS, "EPSG:4326", (X0 + (c0 + cc) * CELL).ravel().tolist(), (Y1 - (r0 + rr) * CELL).ravel().tolist())
LX = ((np.array(lon) - W) / RES).reshape(n + 1, n + 1); LY = ((N - np.array(lat)) / RES).reshape(n + 1, n + 1)

# доля гари по снимку (dNBR > 0,2) в каждой клетке нашей сетки
frac = np.full((n, n), np.nan)
for r in range(n):
    for c in range(n):
        xs = [LX[r, c], LX[r, c + 1], LX[r + 1, c], LX[r + 1, c + 1]]; ys = [LY[r, c], LY[r, c + 1], LY[r + 1, c], LY[r + 1, c + 1]]
        x0, x1, y0, y1 = int(min(xs)), int(np.ceil(max(xs))), int(min(ys)), int(np.ceil(max(ys)))
        if x0 >= 0 and y0 >= 0 and x1 <= dnbr.shape[1] and y1 <= dnbr.shape[0]:
            frac[r, c] = float(np.nanmean(dnbr[y0:y1, x0:x1] > 0.2))
s2 = np.nan_to_num(frac) >= 0.3                                         # клетка «сгорела по снимку»: ≥ 30 % пикселей dNBR > 0,2
s2_new = s2 & ~burned


def f1(p, y):
    tp = int((p & y).sum()); return round(2 * tp / max(2 * tp + int((p & ~y).sum()) + int((~p & y).sum()), 1), 3)


km2 = lambda m: round(float(m.sum()) * (CELL / 1000) ** 2, 2)
out = dict(s2_new_km2=km2(s2_new), s2_start_overlap=round(float(np.nanmean(frac[burned])), 2),
           viirs72_in_s2=round(float(s2[Y[3]].mean()), 2) if Y[3].any() else None,
           F1_vs_s2=dict(viirs_fact=f1(Y[3], s2_new), hybrid=f1(hyb[3], s2_new), unet=f1(unet[3], s2_new), fireskz=f1(kz, s2_new)),
           km2=dict(viirs_fact=km2(Y[3]), hybrid=km2(hyb[3]), unet=km2(unet[3]), fireskz=km2(kz)),
           kz_in_s2=round(float(s2[kz].mean()), 3) if kz.any() else None, hyb_in_s2=round(float(s2[hyb[3]].mean()), 3))
json.dump(out, open(os.path.join(D, "s2_cells.json"), "w"), indent=1)
print(json.dumps(out, ensure_ascii=False))

# картинка: снимок после пожара + dNBR + контуры
anyc = burned | Y[3] | hyb[3] | kz | s2_new
ar, ac = np.nonzero(anyc)
xx = np.r_[LX[ar, ac], LX[ar + 1, ac + 1]]; yy = np.r_[LY[ar, ac], LY[ar + 1, ac + 1]]
cx, cy, half = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2, max(np.ptp(xx), np.ptp(yy)) / 2 * 1.1


def outline(ax, m, color, lw, ls="-"):
    for r, c in zip(*np.nonzero(m)):
        for dr, dc, a, b in ((-1, 0, (r, c), (r, c + 1)), (1, 0, (r + 1, c), (r + 1, c + 1)), (0, -1, (r, c), (r + 1, c)), (0, 1, (r, c + 1), (r + 1, c + 1))):
            if not (0 <= r + dr < n and 0 <= c + dc < n and m[r + dr, c + dc]):
                ax.plot([LX[a], LX[b]], [LY[a], LY[b]], c=color, lw=lw, ls=ls, solid_capstyle="round")


fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), dpi=110)
img = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)
axs[0].imshow(np.nan_to_num(img)); axs[0].set_title(os.environ.get("TITLE", "Sentinel-2 после пожара: снимок 11.09.2026 (~82 ч после старта прогноза)"), fontsize=12)
im = axs[1].imshow(dnbr, cmap="inferno", vmin=0, vmax=0.5); axs[1].set_title("dNBR: светлое — гарь по снимку", fontsize=12)
for ax in axs:
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    outline(ax, burned, "white", 2.2)
    outline(ax, Y[3], "#FF2A2A", 2.0)
    outline(ax, kz, "#7FD1FF", 1.6, "--")
    outline(ax, hyb[3] | burned, "#36E08A", 1.8, ":")
axs[1].plot([], [], c="white", lw=2, label="старт (VIIRS)"); axs[1].plot([], [], c="#FF2A2A", lw=2, label="факт за 72 ч (VIIRS)")
axs[1].plot([], [], c="#7FD1FF", ls="--", lw=2, label="fires.kz"); axs[1].plot([], [], c="#36E08A", ls=":", lw=2, label="гибрид: старт + прогноз")
axs[1].legend(loc="lower right", fontsize=10, framealpha=0.8)
fig.tight_layout(); fig.savefig(os.path.join(D, "s2_check.png")); plt.close(fig)
print("saved")
