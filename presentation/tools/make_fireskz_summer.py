"""Летнее сравнение с fires.kz (август 2026): как make_fireskz.py, но прогноз считается движком prod напрямую
(сервер принимает даты только за 60 дней), а кадр расширяется за пределы их скриншота - летние пожары крупнее.

  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_fireskz_summer.py     (из project_till_nov)
Выход: presentation/fireskz/S<n>.png, summer_cases.json.
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from prod.engine import FireEngine
from prod.live import Live

D = os.path.join(ROOT, "presentation", "fireskz")
MAP_W, MAP_H, MAP_BOTTOM = 1292, 770, 605

# (дата нашего прогноза: срок 23:00 UTC, ближайший к их старту; lat, lon пожара; вид z/lat/lon; их термоточка; их старт; их площадь га; шагов; подпись)
CASES = {
    "S1": ("2026-08-15", 48.5033, 82.2622, (12, 48.44362, 82.26122), (82.24947, 48.49296), "15.08 21:38", 5621.6, 24, "Абайская"),
    "S2": ("2026-08-11", 49.5906, 73.9225, (12, 49.43409, 73.83573), (73.87949, 49.48957), "12.08 04:27", 10323.7, 24, "Карагандинская"),
    "S3": ("2026-08-14", 48.7741, 74.4445, (12, 48.66237, 74.51085), (74.50891, 48.73559), "15.08 10:49", 23508.1, 24, "Карагандинская"),
    "S4": ("2026-08-17", 47.5760, 81.1028, (12, 47.58703, 81.06598), (81.10180, 47.57446), "17.08 08:50", 21083.6, 24, "Абайская"),
    "S5": ("2026-08-15", 51.1801, 67.4015, (12, 51.11687, 67.40051), (67.40383, 51.18185), "15.08 20:53", 5742.6, 24, "Акмолинская"),
}


def merc(lon, lat, z):
    s = 256 * 2 ** z
    return (lon + 180) / 360 * s, (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * s


def to_px(lon, lat, view, off=(0, 0)):
    z, clat, clon = view
    x0, y0 = merc(clon, clat, z); x, y = merc(lon, lat, z)
    return MAP_W / 2 + (x - x0) + off[0], MAP_H / 2 + (y - y0) + off[1]


def polys(fc):
    for f in fc.get("features", []):
        g = f["geometry"]
        for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            yield p[0]


L = Live(FireEngine())
out = {}
for k, (date, lat, lon, view, hot, their_t, their_ha, steps, region) in CASES.items():
    res, past, after = L.clusters(date)
    f = min([f for f in res["fires"] if not f["flare"]], key=lambda f: (f["lat"] - lat) ** 2 + (f["lon"] - lon) ** 2)
    fc = {m: L.forecast(f["id"], date, m, near=(f["lat"], f["lon"])) for m in ("expected", "active")}
    H = {m: fc[m]["horizons"]["72h"] for m in fc}

    img = np.asarray(Image.open(os.path.join(D, f"{k}_raw.jpg")).convert("RGB"))[:MAP_BOTTOM, :MAP_W]
    a = img.astype(int); red = (a[..., 0] > 170) & (a[..., 1] < 80) & (a[..., 2] < 80)
    ry, rx = np.nonzero(red); hx0, hy0 = to_px(*hot, view)
    off = (np.median(rx) - hx0, np.median(ry) - hy0) if len(rx) > 50 else (0, 0)
    P = lambda lo, la: to_px(lo, la, view, off)

    shapes = {n: [[P(*q) for q in p] for p in polys(fc_)] for n, fc_ in
              (("start", fc["expected"]["start"]), ("exp", H["expected"]["pred"]), ("act", H["active"]["pred"]), ("true", H["expected"]["true"]))}
    aft = np.array([P(x, y) for x, y, _ in fc["expected"].get("after_points", [])]).reshape(-1, 2)
    # кадр: их моделирование (красное ядро и контур) + наш старт, прогноз и факт; квадрат с полями
    sel = (a.max(2) - a.min(2) > 90)                                   # насыщенные цвета их контуров
    sy, sx = np.nonzero(sel)
    xs = [np.percentile(sx, 1), np.percentile(sx, 99)] if len(sx) else []
    ys = [np.percentile(sy, 1), np.percentile(sy, 99)] if len(sy) else []
    for n in ("start", "exp", "true"):
        for p in shapes[n]:
            xs += [q[0] for q in p]; ys += [q[1] for q in p]
    if len(aft):
        xs += list(aft[:, 0]); ys += list(aft[:, 1])
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys), 560) / 2 * 1.08
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.patch.set_facecolor("#2A2825"); ax.set_facecolor("#2A2825")
    ax.imshow(img, extent=(0, MAP_W, MAP_BOTTOM, 0))
    ax.add_patch(plt.Rectangle((0, 0), MAP_W, MAP_BOTTOM, fc="none", ec="#8C8377", lw=1, ls=":"))
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    for p in shapes["true"]:
        ax.add_patch(MPoly(p, closed=True, fc=(0.89, 0.34, 0.18, 0.45), ec="none"))
    for p in shapes["exp"]:
        ax.add_patch(MPoly(p, closed=True, fc=(0.18, 0.53, 0.67, 0.5), ec="#2E86AB", lw=1.6))
    for p in shapes["act"]:
        ax.add_patch(MPoly(p, closed=True, fc="none", ec="white", lw=1.3, ls="--"))
    for p in shapes["start"]:
        ax.add_patch(MPoly(p, closed=True, fc="none", ec="black", lw=2))
    if len(aft):
        ax.scatter(aft[:, 0], aft[:, 1], s=8, c="#FF2A2A", lw=0, zorder=6)
    hx, hy = P(*hot); ax.scatter([hx], [hy], marker="*", s=320, c="white", edgecolors="black", lw=0.8, zorder=7)
    mpp = 156543.03 * math.cos(math.radians(view[1])) / 2 ** view[0]; km = 5 if half * 2 * mpp > 25000 else 2; Lp = km * 1000 / mpp
    x0, yb = cx - half + 20, cy + half - 22
    ax.plot([x0, x0 + Lp], [yb, yb], c="white", lw=4, solid_capstyle="butt")
    ax.text(x0 + Lp / 2, yb - 12, f"{km} км", color="white", ha="center", fontsize=11, weight="bold")
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(os.path.join(D, f"{k}.png"), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
    out[k] = dict(region=region, their_start=their_t, their_km2=round(their_ha / 100, 1), their_steps=steps, our_start=fc["expected"]["t_utc"],
                  state_km2=fc["expected"]["state_km2"], our_exp_km2=H["expected"]["pred_km2"], our_act_km2=H["active"]["pred_km2"],
                  true_km2=H["expected"]["true_km2"], F1_exp=H["expected"].get("F1"), F1_act=H["active"].get("F1"),
                  move=H["expected"].get("move"), wind=fc["expected"]["wind"]["24h"], offset_px=[round(float(v)) for v in off],
                  lat=f["lat"], lon=f["lon"])
    print(k, out[k], flush=True)
json.dump(out, open(os.path.join(D, "summer_cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
