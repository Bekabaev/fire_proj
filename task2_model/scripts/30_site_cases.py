"""Шаг 30. Проверка сайта на известных пожарах: запросы /predict через публичный адрес сервера (как внешний пользователь)
от реального состояния пожара (ячейки гари на срок среза), сравнение контуров 12/24/48/72 ч с фактической новой гарью.
Погода: для прошлых дат сервер берёт ERA5 («идеальный прогноз», weather_src в ответе).
Запуск: SITE=https://... python -X utf8 30_site_cases.py   -> out_v3/site_cases/ (PNG по пожарам, cases.csv, cases.json)
"""
import os, sys, json, time
import numpy as np, pandas as pd, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_geom, transform as warp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from prod.config import CELL, CRS, X0, Y1, HZ

SITE = os.environ.get("SITE", "http://127.0.0.1:8000").rstrip("/")
CACHE = os.path.join(ROOT, "task2_model", "cache_v3"); OUT = os.path.join(ROOT, "task2_model", "out_v3", "site_cases")
B = 0                                                     # канал burned_frac в *_M.npy (MAPS[0])
N_TEST, N_RES = 6, 4


def pick(part, n, seed):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    ok = (si.state_km2.between(2, 150)) & (si.new_72h_km2.between(10, 400)) & (si.valid_frac > 0.95) & (si.new_12h_km2 > 0)
    c = si[ok].copy(); c["i"] = c.index; key = "event" if part == "reserves" else "fire_id"
    c = c.sample(frac=1, random_state=seed).drop_duplicates(key).head(n)
    return si, c.sort_values("t_utc")


def lonlat(r, c):
    lon, lat = warp(CRS, "EPSG:4326", (X0 + (np.asarray(c) + 0.5) * CELL).tolist(), (Y1 - (np.asarray(r) + 0.5) * CELL).tolist())
    return np.array(lon), np.array(lat)


def raster(fc, r0, c0, n=96):
    """GeoJSON (lon/lat) -> маска n × n в окне среза (r0, c0)."""
    g = [transform_geom("EPSG:4326", CRS, f["geometry"]) for f in fc["features"]]
    if not g:
        return np.zeros((n, n), bool)
    return rasterize([(x, 1) for x in g], out_shape=(n, n), transform=from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL)).astype(bool)


def f1(p, y):
    tp = (p & y).sum(); return 2 * tp / max(2 * tp + (p & ~y).sum() + (~p & y).sum(), 1)


def hotspots(m, r0, c0):
    """Термоточки VIIRS/MODIS из каналов среза -> [(lon, lat, часов до t, frp)] (как их передал бы пользователь сайта)."""
    a12, a24, aal = (np.rint(np.expm1(np.asarray(m[k], np.float32))).astype(int) for k in (2, 4, 5))
    frp = np.expm1(np.asarray(m[3], np.float32) * 3); out = []
    for r, c in zip(*np.nonzero(aal > 0)):
        lon, lat = lonlat([r0 + r], [c0 + c]); lo, la = round(float(lon[0]), 5), round(float(lat[0]), 5)
        n12, n24 = a12[r, c], max(a24[r, c] - a12[r, c], 0); nold = max(aal[r, c] - a24[r, c], 0)
        out += [[lo, la, 6.0, float(frp[r, c] / max(n12, 1))]] * n12 + [[lo, la, 18.0, 10.0]] * n24 + [[lo, la, 48.0, 10.0]] * nold
    return out


def post(q):
    t0 = time.time(); r = requests.post(SITE + "/predict", json=q, timeout=300); r.raise_for_status(); return r.json(), time.time() - t0


def run(part, si, rows, M, Y, P, lev, out):
    for _, r in rows.iterrows():
        i = int(r.i); m = M[i]; b = np.asarray(m[B], np.float32) > 0.5
        rr, cc = np.nonzero(b); lon, lat = lonlat(r.r0 + rr, r.c0 + cc)
        age = float(np.median(np.asarray(m[1], np.float32)[b]) * 240)
        q = dict(burned=np.c_[lon, lat].round(6).tolist(), t_utc=str(r.t_utc), weather="gfs", mode="expected", age_h=max(age, 1.0))
        dA, sec = post(q)
        hs = hotspots(m, int(r.r0), int(r.c0)); dB, _ = post(dict(q, hotspots=hs) if hs else q)
        ev = r.get("event", r.fire_id)
        rec = dict(part=part, fire=str(ev), t_utc=str(r.t_utc), lon=round(float(lon.mean()), 4), lat=round(float(lat.mean()), 4),
                   state_km2=float(b.sum() * 0.25), hotspots=len(hs), weather_src=dA["weather"].get("weather_src"), sec=round(sec, 1))
        res = {"A": [], "B": [], "C": [], "true": []}; prev = np.zeros_like(b)
        for k, h in enumerate(HZ):
            y = np.asarray(Y[i, k]) > 0; res["true"].append(y)
            res["A"].append(raster(dA["contours"][f"{h}h"], int(r.r0), int(r.c0)) & ~b)
            res["B"].append(raster(dB["contours"][f"{h}h"], int(r.r0), int(r.c0)) & ~b)
            prev = prev | (np.asarray(P[i, k]) >= lev[f"{h}h"]); res["C"].append(prev & ~b)
            rec[f"true_km2_{h}h"] = float(y.sum() * 0.25)
            for v in "ABC":
                rec[f"F1_{v}_{h}h"] = round(float(f1(res[v][k], y)), 3); rec[f"km2_{v}_{h}h"] = float(res[v][k].sum() * 0.25)
        out.append(rec); print({k: v for k, v in rec.items() if k.startswith(("fire", "F1", "hotspots"))}, flush=True)
        plot(rec, b, res)


CM = ListedColormap(["#f4f1ea", "#3a3a3a", "#e4572e", "#2e86ab", "#f2c14e"])   # фон, гарь на старте, только факт, только прогноз, совпало


ROWS = (("B", "Сайт: контур гари + термоточки"), ("C", "Офлайн: полные входы модели"))


def plot(rec, b, res):
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(len(ROWS), 4, figsize=(16, 4.9 * len(ROWS) + 0.8))
    anyp = np.any([np.any(res[v], 0) for v, _ in ROWS], 0)
    ys, xs = np.nonzero(b | np.any(res["true"], 0) | anyp)
    pad = 5; cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; s = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + pad
    y0, x0 = int(max(cy - s, 0)), int(max(cx - s, 0)); y1, x1 = int(min(cy + s + 1, 96)), int(min(cx + s + 1, 96))
    for j, (v, name) in enumerate(ROWS):
        for k, h in enumerate(HZ):
            p, t = res[v][k], res["true"][k]; img = np.zeros(b.shape, np.uint8)
            img[t & ~p] = 2; img[p & ~t] = 3; img[p & t] = 4; img[b] = 1
            a = ax[j, k]; a.imshow(img[y0:y1, x0:x1], cmap=CM, vmin=0, vmax=4, interpolation="nearest")
            a.set_title(f"+{h} ч · F1 {rec[f'F1_{v}_{h}h']:.2f}\nфакт {rec[f'true_km2_{h}h']:.1f} км² · прогноз {rec[f'km2_{v}_{h}h']:.1f} км²", fontsize=10.5)
            a.set_xticks([]); a.set_yticks([])
            if k == 0:
                a.set_ylabel(name, fontsize=11)
    kind = "заповедник, экспертный контур" if rec["part"] == "reserves" else "тест 2023–24, гарь по VIIRS/MODIS"
    fig.suptitle(f"Пожар {rec['fire']} ({kind}) · старт {rec['t_utc']} UTC · {rec['lat']}° с. ш., {rec['lon']}° в. д. · гарь на старте {rec['state_km2']:.1f} км² · клетка 500 м",
                 fontsize=12)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CM.colors[1:], ["гарь на старте", "сгорело, но не предсказано", "предсказано, но не сгорело", "предсказано и сгорело"])],
               loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(os.path.join(OUT, f"{rec['part']}_{rec['fire']}.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); out = []
    print("health", requests.get(SITE + "/health", timeout=60).json(), flush=True)
    for part, n, seed in (("test", N_TEST, 1), ("reserves", N_RES, 1)):
        si, rows = pick(part, n, seed)
        M = np.load(os.path.join(CACHE, f"{part}_M.npy"), mmap_mode="r"); Y = np.load(os.path.join(CACHE, f"{part}_Y.npy"), mmap_mode="r")
        src = "gfsa" if part == "test" else "era5"
        P = np.load(os.path.join(ROOT, "task2_model", "out_v3", f"probs_all4_hybrid_{src}_{part}.npy"), mmap_mode="r")   # офлайн = прод-гибрид all4
        lev = json.load(open(os.path.join(ROOT, "task2_model", "out_v3", "prod_hybrid.json")))["deploy"]["level"]
        run(part, si, rows, M, Y, P, lev, out)
    df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, "cases.csv"), index=False)
    json.dump(out, open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(df[[c for c in df if c.startswith("F1")]].mean().round(3).to_dict())
