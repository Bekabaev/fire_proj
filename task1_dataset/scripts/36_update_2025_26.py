"""Шаг 36 (прод, 04.11.2026). Обновление сезонных слоёв для оперативного режима: гари и NDVI 2025–2026.

Global Fire Atlas v2 (шаг 11) заканчивается 2024 годом, поэтому дата выгорания за 2025–26 берётся прямо из MODIS MCD64A1 C6.1
(Planetary Computer, modis-64A1-061, слой Burn_Date, 500 м, ежемесячно; задержка выпуска ~2 мес.) - это тот же продукт,
на котором построен GFA. Растр за год = первая дата выгорания ячейки (день года; 0 - не горело) на национальной сетке 500 м,
формат как doy_<год>.npy шага 11. Проверка согласованности: тот же расчёт за 2024 против doy_2024 (GFA).
NDVI MOD13A1 2025–26 - шаг 12 (ONLY_YEARS=2025,2026), здесь после него: ndvi_peak_<год>.npy и last_burn_<год>.npy (как шаг 19).
Климатология NDVI (ndvi_clim.npz) не пересчитывается - норма 2012–2024, как в обучении.
Запуск: python 36_update_2025_26.py burns  ->  ONLY_YEARS=2025,2026 python 12_static_kz.py  ->  python 36_update_2025_26.py fuel
Выход: data/ext_kz/doy_{2025,2026}.npy, mcd64_months_<год>.json (какие месяцы есть), ndvi_peak_{2025,2026}.npy,
       last_burn_{2025,2026}.npy, qa_mcd64_vs_gfa_2024.txt.
"""
import os, sys, json, time
import numpy as np, requests, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BBOX = [46.6, 40.6, 87.3, 55.4]
NEW = (2025, 2026)


def items(coll, dt):
    j = requests.post(STAC, json={"collections": [coll], "bbox": BBOX, "limit": 1000, "datetime": dt}, timeout=180).json(); out = j["features"]
    while (nx := [l for l in j.get("links", []) if l["rel"] == "next"]):
        j = requests.post(STAC, json=nx[0]["body"], timeout=180).json(); out += j["features"]
    return out


_tok = {}


def signed(href):
    acc = href.split("//")[1].split(".")[0]; cont = href.split(".net/")[1].split("/")[0]
    if (acc, cont) not in _tok or time.time() - _tok[(acc, cont)][1] > 1800:
        t = requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{acc}/{cont}", timeout=60).json()["token"]
        _tok[(acc, cont)] = (t, time.time())
    return href + "?" + _tok[(acc, cont)][0]


def burn_year(y):
    """Первая дата выгорания за год y на национальной сетке (int16, 0 - нет) + список месяцев."""
    feats = [f for f in items("modis-64A1-061", f"{y}-01-01T00:00:00Z/{y}-12-31T23:59:59Z") if f["id"].startswith("MCD64A1")]
    months = sorted({f["properties"]["start_datetime"][:7] for f in feats})
    print(y, "MCD64A1 items", len(feats), "months", months[0] if months else "-", "...", months[-1] if months else "-", flush=True)
    out = np.zeros((GH, GW), np.int16)

    def read(f):
        for k in range(4):
            try:
                with rasterio.open("/vsicurl/" + signed(f["assets"]["Burn_Date"]["href"])) as s:
                    a = s.read(1).astype(np.float32); a[a <= 0] = np.nan      # 0 - не горело, −1/−2 - вода/нет данных
                    dst = np.full((GH, GW), np.nan, np.float32)
                    reproject(a, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=GTR, dst_crs=CRS,
                              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
                    return dst
            except Exception as e:
                if k == 3:
                    print("  failed", f["id"], str(e)[:100], flush=True); return None
                time.sleep(5 * (k + 1))
    with ThreadPoolExecutor(6) as ex:
        for dst in ex.map(read, feats):
            if dst is None:
                continue
            ok = np.isfinite(dst); d = dst.astype(np.int16)
            first = ok & ((out == 0) | (d < out))
            out[first] = d[first]
    return out, months


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "burns"
    if phase == "burns":
        fq = os.path.join(KZ, "qa_mcd64_vs_gfa_2024.txt")
        if not os.path.exists(fq):                              # проверка метода на 2024: MCD64A1 против GFA
            m24, _ = burn_year(2024); g24 = np.asarray(load_doy(2024))
            a, b = m24 > 0, g24 > 0; both = a & b
            txt = (f"2024: ячеек гари MCD64A1 {int(a.sum())}, GFA {int(b.sum())}, общих {int(both.sum())}; "
                   f"IoU {both.sum() / max((a | b).sum(), 1):.3f}; доля GFA, найденная MCD64A1 {both.sum() / max(b.sum(), 1):.3f}; "
                   f"совпадение дня (±1) {float((np.abs(m24[both].astype(int) - g24[both].astype(int)) <= 1).mean()):.3f}")
            open(fq, "w", encoding="utf8").write(txt + "\n"); print(txt, flush=True)
        for y in NEW:
            f = doy_path(y)
            if os.path.exists(f):
                continue
            d, months = burn_year(y)
            np.save(f, d); json.dump(dict(months=months, source="MCD64A1 C6.1 Burn_Date (Planetary Computer)"),
                                     open(os.path.join(KZ, f"mcd64_months_{y}.json"), "w"))
            print(y, "saved, burned cells", int((d > 0).sum()), "km²", float((d > 0).sum() * 0.25), flush=True)
    else:                                                       # fuel: после шага 12 с ONLY_YEARS=2025,2026
        for y in NEW:
            fn = os.path.join(KZ, f"ndvi_{y}.npz"); fp = os.path.join(KZ, f"ndvi_peak_{y}.npy")
            if os.path.exists(fn) and not os.path.exists(fp):
                a = np.load(fn)["ndvi"].astype(np.float32); a[a < -20000] = np.nan
                np.save(fp, np.nan_to_num(np.nanmax(a, 0), nan=-32768).astype(np.int16)); print("peak", y, flush=True)
        last = np.asarray(np.load(os.path.join(KZ, "last_burn_2024.npy")))
        last = np.where(np.asarray(load_doy(2024)) > 0, 2024, last).astype(np.int16)    # last_burn_2025: гари ≤ 2024
        for y in NEW:
            np.save(os.path.join(KZ, f"last_burn_{y}.npy"), last); print("last_burn", y, flush=True)
            if os.path.exists(doy_path(y)):
                last = np.where(np.asarray(load_doy(y)) > 0, y, last).astype(np.int16)
    print("done", flush=True)
