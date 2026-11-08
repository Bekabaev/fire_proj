"""Данные для интерактивной визуализации ELMFIRE / гибрид / FBP / факт на 40 тестовых пожарах.

  python -X utf8 elmfire_bench/export_viz.py <template.html> <out.html>

Слои на клетку (uint8, 96 x 96): flags (бит0 негорючее, бит1 старт, бит2 в области данных),
fact / hyb = первый горизонт (1..4 = 12/24/48/72 ч, 0 = нет), fbp / elm = время прихода, ч (ceil, 255 = нет).
Все 40 кейсов сжаты zlib и встроены в страницу base64 (страница разжимает DecompressionStream).
"""
import base64, csv, json, os, sys, zlib
import numpy as np, rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
HZ = (12, 24, 48, 72)


def first_h(stack):
    out = np.zeros(stack.shape[1:], np.uint8)
    for k in range(3, -1, -1):
        out[stack[k]] = k + 1
    return out


def hours(a):
    a = np.where(np.isfinite(a) & (a > 0), np.ceil(a), 255)
    return np.clip(a, 1, 255).astype(np.uint8)


res = {}
for r in csv.DictReader(open(os.path.join(HERE, "results.csv"), encoding="utf-8")):
    res.setdefault(int(r["case"]), {}).setdefault(r["model"], {})[r["h"]] = dict(
        f1=float(r["F1"]), tol=float(r["F1_tol1km"]), pred=float(r["pred_km2"]), true=float(r["true_km2"]))

blob, cases = bytearray(), []
for i in range(40):
    d = os.path.join(HERE, "cases", f"case_{i:02d}")
    m = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    t, o = np.load(os.path.join(d, "truth.npz")), np.load(os.path.join(d, "ours.npz"))
    with rasterio.open(os.path.join(d, "fbfm40.tif")) as r:
        fb = r.read(1)
    with rasterio.open(os.path.join(d, "elmfire_toa.tif")) as r:
        toa = r.read(1).astype(np.float64); nod = r.nodata
    if nod is not None:
        toa[toa == nod] = -1
    start = t["start"] > 0
    flags = (fb == 99).astype(np.uint8) | (start.astype(np.uint8) << 1) | (t["valid"].astype(np.uint8) << 2)
    fact = first_h((t["Y"] > 0) & ~start[None])
    hyb = first_h(o["hybrid_mask"].astype(bool) & ~start[None])
    fbp = hours(o["fbp_arrival_h"].astype(np.float64)); fbp[start] = 255
    elm = hours(toa / 3600.0); elm[start] = 255
    for a in (flags, fact, hyb, fbp, elm):
        blob += a.tobytes()
    wx = []
    for w in m["weather_windows"]:
        e = w.get("elmfire") or {}
        raw = w.get("raw") or {}
        wx.append(dict(h=w["window_h"][1], ok=w["ok"], from_deg=e.get("wind_from_deg"), ws=raw.get("ws_mean_10m_ms"),
                       t=raw.get("T_max_C"), rh=raw.get("RH_min_pct"), m1=e.get("m1_pct")))
    cases.append(dict(i=i, date=m["t0_utc"][:16].replace("T", " "), lat=round(m["lat"], 2), lon=round(m["lon"], 2),
                      state=round(float(m["state_km2"]), 2), ndvi=m.get("ndvi_median"), wx=wx, res=res.get(i, {})))

payload = dict(n=96, cell_km=0.5, hz=HZ, cases=cases, layers=base64.b64encode(zlib.compress(bytes(blob), 9)).decode())
tpl = open(sys.argv[1], encoding="utf-8").read()
out = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
open(sys.argv[2], "w", encoding="utf-8").write(out)
print("ok", len(out) // 1024, "KB")
