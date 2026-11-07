"""Сравнение с fires.kz: на скриншот их моделирования (последний шаг) накладываются наш прогноз на 72 ч и факт FIRMS.

  ../.venv/Scripts/python.exe -X utf8 presentation/tools/make_fireskz.py     (из project_till_nov, сервер prod запущен)

Скриншоты: presentation/fireskz/<K>_raw.jpg (1568 x 770, карта 0..1292 px по x, центр вида = центр карты).
Выход: presentation/fireskz/<K>.png и cases.json (площади, время, F1).
"""
import json, math, os, sqlite3, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "presentation", "fireskz")
API = "http://127.0.0.1:8000"
MAP_W, MAP_H = 1292, 770                  # карта OpenLayers на скриншоте; панель модели снизу перекрывает y > ~610

# ключ: (дата для нашего прогноза, lat, lon пожара у нас, вид их карты z/lat/lon, их термоточка lon/lat, их время, их площадь га, шагов)
CASES = {
    "A": ("2026-11-06", 49.5757, 71.7547, (11.96, 49.63136, 71.74257), (71.75609, 49.57699), "06.11 08:28", 13708.6, 24),
    "B": ("2026-11-05", 49.8845, 70.6954, (12, 49.85704, 70.69975), (70.70434, 49.89331), "05.11 10:19", 2783.9, 24),
    "C": ("2026-11-06", 49.5965, 72.3138, (12, 49.6299, 72.30642), (72.31191, 49.59933), "06.11 08:28", 7029.3, 20),
    "D": ("2026-11-06", 51.2612, 70.5195, (12, 51.25551, 70.56075), (70.52315, 51.26410), "06.11 07:43", 2815.4, 24),
    "E": ("2026-11-05", 54.2408, 70.0349, (12, 54.26538, 70.0166), (70.03416, 54.24663), "05.11 08:05", 8508.5, 24),
}


def get(path):
    with urllib.request.urlopen(API + path, timeout=600) as r:
        return json.load(r)


def merc(lon, lat, z):
    s = 256 * 2 ** z
    return (lon + 180) / 360 * s, (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * s


def to_px(lon, lat, view):
    off = (0, 0)
    if isinstance(view[0], tuple):
        view, off = view
    z, clat, clon = view
    x0, y0 = merc(clon, clat, z); x, y = merc(lon, lat, z)
    return MAP_W / 2 + (x - x0) + off[0], MAP_H / 2 + (y - y0) + off[1]


def polys(fc):
    for f in fc.get("features", []):
        g = f["geometry"]
        for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            yield p[0]


db = sqlite3.connect(os.path.join(ROOT, "prod", "live.db"))
out = {}
for k, (date, lat, lon, view, hot, their_t, their_ha, steps) in CASES.items():
    fires = get(f"/live/fires?date={date}")
    f = min(fires["fires"], key=lambda f: (f["lat"] - lat) ** 2 + (f["lon"] - lon) ** 2)
    fc = {m: get(f"/live/forecast/{f['id']}?date={date}&mode={m}&lat={f['lat']}&lon={f['lon']}") for m in ("expected", "active")}
    t0 = fc["expected"]["t_utc"]
    pts = db.execute("select lon, lat, t from points where t > ? and t <= datetime(?, '+72 hours') and abs(lat-?)<0.15 and abs(lon-?)<0.25",
                     (t0, t0, f["lat"], f["lon"])).fetchall()
    before = db.execute("select lon, lat from points where t <= ? and abs(lat-?)<0.15 and abs(lon-?)<0.25 and t > datetime(?, '-72 hours')",
                        (t0, f["lat"], f["lon"], t0)).fetchall()
    H = {m: fc[m]["horizons"]["72h"] for m in fc}

    img = np.asarray(Image.open(os.path.join(D, f"{k}_raw.jpg")).convert("RGB"))[:, :MAP_W]
    # привязка: адрес вида в URL после анимации не обновляется -> их термоточка (точка возгорания) = центр красного ядра контуров
    a = img[:600].astype(int); red = (a[..., 0] > 170) & (a[..., 1] < 80) & (a[..., 2] < 80)
    ry, rx = np.nonzero(red); hx0, hy0 = to_px(*hot, view)
    off = (np.median(rx) - hx0, np.median(ry) - hy0) if len(rx) > 50 else (0, 0)
    view = (view, off)
    # кадр: вокруг их контура и нашего пожара, квадрат 560 px (~14 км при z12), не ниже панели модели
    cx, cy = to_px(f["lon"], f["lat"], view)
    half = 300; y0 = 5                                         # вся высота карты над панелью модели (0..610 px)
    x0 = int(min(max(cx - half, 0), MAP_W - 2 * half))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(img); ax.set_xlim(x0, x0 + 2 * half); ax.set_ylim(y0 + 2 * half, y0); ax.axis("off")
    for p in polys(fc["expected"]["start"]):
        ax.add_patch(MPoly([to_px(*q, view) for q in p], closed=True, fc="none", ec="black", lw=2.2))
    for p in polys(H["expected"]["pred"]):
        ax.add_patch(MPoly([to_px(*q, view) for q in p], closed=True, fc=(0.18, 0.53, 0.67, 0.55), ec="#2E86AB", lw=2))
    for p in polys(H["active"]["pred"]):
        ax.add_patch(MPoly([to_px(*q, view) for q in p], closed=True, fc="none", ec="white", lw=1.6, ls="--"))
    if before:
        xy = np.array([to_px(a, b, view) for a, b in before]); ax.scatter(xy[:, 0], xy[:, 1], s=40, c="black", edgecolors="white", lw=0.6, zorder=5)
    if pts:
        xy = np.array([to_px(a, b, view) for a, b, _ in pts]); ax.scatter(xy[:, 0], xy[:, 1], s=60, c="#FF2A2A", edgecolors="white", lw=0.8, zorder=6)
    hx, hy = to_px(*hot, view); ax.scatter([hx], [hy], marker="*", s=320, c="white", edgecolors="black", lw=0.8, zorder=7)
    # масштаб 2 км
    mpp = 156543.03 * math.cos(math.radians(view[0][1])) / 2 ** view[0][0]; L = 2000 / mpp
    ax.plot([x0 + 20, x0 + 20 + L], [y0 + 2 * half - 22] * 2, c="white", lw=4, solid_capstyle="butt")
    ax.text(x0 + 20 + L / 2, y0 + 2 * half - 32, "2 км", color="white", ha="center", fontsize=11, weight="bold")
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(os.path.join(D, f"{k}.png"), dpi=100); plt.close(fig)
    out[k] = dict(their_start=their_t, their_km2=round(their_ha / 100, 1), their_steps=steps, our_start=t0,
                  state_km2=fc["expected"]["state_km2"], our_exp_km2=H["expected"]["pred_km2"], our_act_km2=H["active"]["pred_km2"],
                  true_km2=H["expected"]["true_km2"], n_after=len(pts), last_after=max([p[2] for p in pts], default=None),
                  wind=fc["expected"]["wind"]["24h"], offset_px=[round(float(v)) for v in view[1]], lat=f["lat"], lon=f["lon"])
    print(k, out[k])
json.dump(out, open(os.path.join(D, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
