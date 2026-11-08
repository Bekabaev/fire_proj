"""Пожар 7 сентября 2026 (49,11° с. ш., 74,87° в. д., Карагандинская): моделирование fires.kz против U-Net и гибрида.
Картинки для слайдов 15, 16, 20, 25 на одной подложке (спутник Esri со скриншота fires.kz), цвета - как на нашем сайте.

Вход (presentation/fireskz_0907/): kz_model.jpg - скриншот fires.kz со слоем моделирования (шаг 24/24), kz_plain.jpg - тот же вид
без слоя. Вид: z = 12, центр 49,1142° с. ш., 74,82312° в. д.; масштаб скриншота откалиброван сдвигом карты (0,595 px на пиксель
мира 256 · 2^z), центр карты - (630, 392) px (карта 1260 × 784 без боковой панели).
  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_fireskz_0907.py      (из project_till_nov)
Выход: phys_kz.png, unet_P.png, cmp_kz.png, cmp_unet.png, cmp_hybrid.png, hz_12h..72h.png, numbers.json.
"""
import os, sys, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_geom, transform as warp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import prod.live as live
live.SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
from prod.engine import FireEngine
from prod.config import CELL, CRS, X0, Y1, HZ

D = os.path.join(ROOT, "presentation", "fireskz_0907")
DATE, NEAR = "2026-09-07", (49.1146, 74.8734)
Z, CLAT, CLON, SC, CX, CY, MW, MH = 12, 49.1142, 74.82312, 0.595, 630, 392, 1260, 625
RED, BLUE, OVER, ASH = "#e4572e", "#2e86ab", "#b07566", "#3a3a3a"


def merc(lon, lat):
    s = 256 * 2 ** Z
    return (np.asarray(lon) + 180) / 360 * s, (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * s


MX0, MY0 = merc(CLON, CLAT)


def to_px(lon, lat):
    x, y = merc(lon, lat)
    return CX + (x - MX0) * SC, CY + (y - MY0) * SC


# ---------- наш прогноз ----------
E = FireEngine(); L = live.Live(E); stash = {}
_s, _p = E._scene_at, E.predict_scene
E._scene_at = lambda *a, **k: (lambda r: (stash.__setitem__("rc", (r[3], r[4])), r)[1])(_s(*a, **k))
E.predict_scene = lambda *a, **k: (lambda r: (stash.__setitem__("out", r), r)[1])(_p(*a, **k))
res = L.forecast(None, DATE, "expected", near=NEAR)
o = stash["out"]; r0, c0 = stash["rc"]; n = o["mask"].shape[-1]
tr = E.window_transform(r0, c0)


def ras(fc):
    g = [transform_geom("EPSG:4326", CRS, f["geometry"]) for f in fc.get("features", [])]
    return rasterize([(x, 1) for x in g], out_shape=(n, n), transform=tr).astype(bool) if g else np.zeros((n, n), bool)


burned = ras(res["start"])
Y = np.stack([ras(res["horizons"][f"{h}h"].get("true", {})) for h in HZ])
Yv = Y.copy()                                                           # факт по термоточкам VIIRS (кадр считается по нему)
FACT = os.environ.get("FACT", "viirs")                                 # FACT=s2: факт за 72 ч - гарь по снимку Sentinel-2 11.09 (клетки 500 м)
SUF = "_s2" if FACT == "s2" else ""
F_S2 = os.path.join(D, "fact_s2_cells.npy")
if FACT == "s2" and os.path.exists(F_S2):
    Y[3] = np.load(F_S2)
hyb = np.stack([o["mask"][k] & ~burned for k in range(len(HZ))])
unetP = o["unet"]; unet = (unetP >= 0.5) & ~burned[None]

# ---------- клетки сетки -> пиксели скриншота ----------
rr, cc = np.mgrid[0:n + 1, 0:n + 1]
xs, ys = X0 + (c0 + cc) * CELL, Y1 - (r0 + rr) * CELL
lon, lat = warp(CRS, "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist())
PX, PY = to_px(np.array(lon), np.array(lat)); PX = PX.reshape(n + 1, n + 1); PY = PY.reshape(n + 1, n + 1)


def cell_poly(r, c):
    return [(PX[r, c], PY[r, c]), (PX[r, c + 1], PY[r, c + 1]), (PX[r + 1, c + 1], PY[r + 1, c + 1]), (PX[r + 1, c], PY[r + 1, c])]


# ---------- маска fires.kz на нашей сетке: разница скриншотов со слоем и без ----------
A = np.asarray(Image.open(os.path.join(D, "kz_model.jpg")).convert("RGB")).astype(int)[:MH, :MW]
B = np.asarray(Image.open(os.path.join(D, "kz_plain.jpg")).convert("RGB")).astype(int)[:MH, :MW]
A[340:380, 565:595] = A[340:380, 525:555]; B[340:380, 565:595] = B[340:380, 525:555]   # курсор мыши на скриншоте - заплатка
A = A.astype(np.uint8); B = B.astype(np.uint8)
lay = np.abs(A.astype(int) - B.astype(int)).sum(2) > 60
from scipy import ndimage
lay = ndimage.binary_closing(ndimage.binary_opening(lay, iterations=1), iterations=3)
lay = ndimage.binary_fill_holes(lay)
px_km2 = (156543.03 * math.cos(math.radians(CLAT)) / 2 ** Z / SC / 1000) ** 2
kz = np.zeros((n, n), bool)
for r in range(n):
    for c in range(n):
        sx = PX[r, c] + (PX[r, c + 1] - PX[r, c]) * np.array([.2, .5, .8])[:, None] + (PX[r + 1, c] - PX[r, c]) * np.array([.2, .5, .8])[None, :]
        sy = PY[r, c] + (PY[r, c + 1] - PY[r, c]) * np.array([.2, .5, .8])[:, None] + (PY[r + 1, c] - PY[r, c]) * np.array([.2, .5, .8])[None, :]
        ix, iy = np.round(sx).astype(int), np.round(sy).astype(int)
        ok = (ix >= 0) & (ix < MW) & (iy >= 0) & (iy < MH)
        if ok.any() and lay[iy[ok], ix[ok]].mean() >= 0.5:
            kz[r, c] = True
kz_new = kz & ~burned
valid = np.ones((n, n), bool)


def f1(p, y):
    tp = int((p & y).sum()); return round(2 * tp / max(2 * tp + int((p & ~y).sum()) + int((~p & y).sum()), 1), 3)


km2 = lambda m: round(float(m.sum()) * (CELL / 1000) ** 2, 2)
num = dict(date=DATE, t_utc=res["t_utc"], fire_start=res["fire_start_utc"], state_km2=km2(burned), weather=res["weather_src"],
           kz_layer_km2_screen=round(float(lay.sum()) * px_km2, 1), kz_km2=km2(kz_new), kz_F1_72=f1(kz_new, Y[3]),
           true_km2={f"{h}h": km2(Y[k]) for k, h in enumerate(HZ)}, hyb_km2={f"{h}h": km2(hyb[k]) for k, h in enumerate(HZ)},
           hyb_F1={f"{h}h": f1(hyb[k], Y[k]) for k, h in enumerate(HZ)}, unet_km2_72=km2(unet[3]), unet_F1_72=f1(unet[3], Y[3]),
           kz_F1={f"{h}h": f1(kz_new, Y[k]) for k, h in enumerate(HZ)}, wind=res.get("wind"))
json.dump(num, open(os.path.join(D, f"numbers{SUF}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(num, ensure_ascii=False, default=str))

# ---------- кадр: квадрат вокруг всего интересного ----------
anyc = burned | Yv.any(0) | hyb.any(0) | unet[3] | kz_new
ar, ac = np.nonzero(anyc)
xx = np.concatenate([PX[ar, ac], PX[ar + 1, ac + 1]]); yy = np.concatenate([PY[ar, ac], PY[ar + 1, ac + 1]])
cx, cy = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
half = max(xx.max() - xx.min(), yy.max() - yy.min()) / 2 * 1.12
FRAME_ALL = (cx, cy, half)
tc = burned | Yv.any(0) | hyb.any(0) | unet[3]                         # плотный кадр: только наш пожар, без эллипса fires.kz
tr_, tcc = np.nonzero(tc)
tx = np.concatenate([PX[tr_, tcc], PX[tr_ + 1, tcc + 1]]); ty = np.concatenate([PY[tr_, tcc], PY[tr_ + 1, tcc + 1]])
FRAME_TIGHT = ((tx.min() + tx.max()) / 2, (ty.min() + ty.max()) / 2, max(tx.max() - tx.min(), ty.max() - ty.min()) / 2 * 1.35)
km_px = 1000 / (156543.03 * math.cos(math.radians(CLAT)) / 2 ** Z / SC)


def frame(img, size=6.0, fr=None):
    cx, cy, half = fr or FRAME_ALL
    km = 2 if half * 2 / km_px > 8 else 1
    fig, ax = plt.subplots(figsize=(size, size), dpi=134)
    ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    x0, yb = cx - half + 0.05 * half, cy + half - 0.06 * half
    ax.plot([x0, x0 + km * km_px], [yb, yb], c="white", lw=4, solid_capstyle="butt")
    ax.text(x0 + km * km_px / 2, yb - 0.035 * half, f"{km} км", color="white", ha="center", fontsize=12, weight="bold")
    return fig, ax


def cells(ax, m, **kw):
    for r, c in zip(*np.nonzero(m)):
        ax.add_patch(MPoly(cell_poly(r, c), closed=True, **kw))


def outline(ax, m, color, lw=2.2, ls="-"):
    for r, c in zip(*np.nonzero(m)):                                    # контур по внешним рёбрам клеток
        for dr, dc, a, b in ((-1, 0, (r, c), (r, c + 1)), (1, 0, (r + 1, c), (r + 1, c + 1)), (0, -1, (r, c), (r + 1, c)), (0, 1, (r, c + 1), (r + 1, c + 1))):
            rr_, cc_ = r + dr, c + dc
            if not (0 <= rr_ < n and 0 <= cc_ < n and m[rr_, cc_]):
                ax.plot([PX[a], PX[b]], [PY[a], PY[b]], c=color, lw=lw, ls=ls, solid_capstyle="round")


def vs(pred, y, name, img=B, fr=None):
    fig, ax = frame(img, fr=fr)
    cells(ax, y & ~pred, fc=RED, ec="none", alpha=0.85)
    cells(ax, pred & ~y, fc=BLUE, ec="none", alpha=0.45)
    cells(ax, pred & y, fc=OVER, ec="none", alpha=0.9)
    cells(ax, burned, fc=ASH, ec="none", alpha=0.92)
    outline(ax, pred | burned, "#0b3c5d", 1.8, (0, (3, 2)))
    fig.savefig(os.path.join(D, name.replace(".png", SUF + ".png")), facecolor="black"); plt.close(fig)


# слайд 15: физика fires.kz (их слой как есть) + реальная гарь контуром
fig, ax = frame(A)
outline(ax, Y[3], "#FF2A2A", 2.4)
cells(ax, burned, fc=ASH, ec="none", alpha=0.92)
fig.savefig(os.path.join(D, f"phys_kz{SUF}.png"), facecolor="black"); plt.close(fig)

# слайд 16: вероятность U-Net
fig, ax = frame(B, fr=FRAME_TIGHT)
cmap = plt.get_cmap("Blues")
for r, c in zip(*np.nonzero((unetP[3] >= 0.1) & ~burned)):
    p = float(unetP[3, r, c]); ax.add_patch(MPoly(cell_poly(r, c), closed=True, fc=cmap(0.35 + 0.65 * p), ec="none", alpha=0.85))
outline(ax, unet[3], "black", 1.8)
outline(ax, Y[3], "#FF2A2A", 2.2, (0, (4, 3)))
cells(ax, burned, fc=ASH, ec="none", alpha=0.92)
fig.savefig(os.path.join(D, f"unet_P{SUF}.png"), facecolor="black"); plt.close(fig)

# слайд 20: три модели на 72 ч
vs(kz_new, Y[3], "cmp_kz.png"); vs(unet[3], Y[3], "cmp_unet.png"); vs(hyb[3], Y[3], "cmp_hybrid.png")
# слайд 25: гибрид на 4 горизонтах
for k, h in enumerate(HZ):
    vs(hyb[k], Y[k], f"hz_{h}h.png", fr=FRAME_TIGHT)
print("saved", D)
