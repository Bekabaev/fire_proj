"""Шаг 28 (датасет v3). Канадские индексы пожарной опасности (FWI System) для всех срезов v3 и срезов заповедников.

Свой расчёт FWI для Казахстана невозможен: окна погоды шага 13 короткие (−7 сут), а DC / DMC требуют расчёта с весны.
Источник: NASA GFWED v2.0 (Field et al. 2015), MERRA-2 с поправкой осадков по наблюдениям (MERRA2.CORRECTED),
суточно, 0,625° × 0,5°, без ключа: portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.MERRA2/Default/MERRA2.CORRECTED/.
Индексы рассчитаны на местный полдень; на момент t доступен последний день d, для которого полдень (≈ 07 UTC в Казахстане) ≤ t:
d = дата(t − 7 ч). Признаки: gfwed_{FFMC,DMC,DC,ISI,BUI,FWI,DSR} в ближайшей ячейке с данными (до 2 ячеек), + gfwed_FWI_max3d
(максимум FWI за d−2…d). Скачиваются только нужные даты; глобальный файл обрезается до Казахстана и удаляется.
Выход: data/ext_cache/gfwed/kz_<год>.npz (dates, vars, data [n, 7, h, w], lat, lon), колонки gfwed_* в slices_v3_index.csv
и slices_v3_reserves_index.csv.
"""
import os, sys, time
import numpy as np, pandas as pd, requests, rasterio
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OD = os.path.join(CACHE, "gfwed"); os.makedirs(OD, exist_ok=True)
URL = "https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.MERRA2/Default/MERRA2.CORRECTED/{y}/FWI.MERRA2.CORRECTED.Daily.Default.{d}.nc"
VARS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"]
LON0, LON1, LAT0, LAT1 = 45.0, 89.0, 39.0, 57.0
IDX = [os.path.join(KZ, "slices_v3_index.csv"), os.path.join(KZ, "slices_v3_reserves_index.csv")]


def avail_day(t):
    return (t - pd.Timedelta(hours=7)).dt.normalize()


def fetch(d):
    """Скачать день d, вернуть вырезку [7, h, w] и координаты (lat, lon)."""
    f = os.path.join(OD, f"tmp_{d:%Y%m%d}.nc")
    for k in range(6):
        try:
            r = requests.get(URL.format(y=d.year, d=f"{d:%Y%m%d}"), timeout=180)
            if r.status_code == 404:
                return d, None
            r.raise_for_status(); open(f, "wb").write(r.content); break
        except Exception as e:
            print("  retry", d.date(), str(e)[:80], flush=True); time.sleep(10 * (k + 1))
    else:
        return d, None
    out = []
    for v in VARS:
        with rasterio.open(f'netcdf:"{f}":MERRA2.CORRECTED_{v}') as s:
            a = s.read(1).astype(np.float32); tr = s.transform
            a[a > 1e30] = np.nan
            lon = tr.c + (np.arange(s.width) + 0.5) * tr.a; lat = tr.f + (np.arange(s.height) + 0.5) * tr.e
            ci = (lon >= LON0) & (lon <= LON1); ri = (lat >= LAT0) & (lat <= LAT1)
            out.append(a[np.ix_(ri, ci)])
    os.remove(f)
    return d, (np.stack(out), lat[ri], lon[ci])


tabs = [pd.read_csv(f, parse_dates=["t_utc"], low_memory=False) for f in IDX]
need = pd.concat([pd.concat([avail_day(t.t_utc) - pd.Timedelta(days=k) for k in range(3)]) for t in tabs]).drop_duplicates().sort_values()
print("days needed:", len(need), flush=True)
cube = {}
groups = list(need.groupby(need.dt.year))
if os.environ.get("REVERSE"):        # второй параллельный процесс: только скачивание, годы с конца (основной пропустит готовые годы)
    groups = groups[::-1]
for y, dd in groups:
    if os.environ.get("REVERSE") and os.path.exists(os.path.join(OD, f"kz_{y}.npz")):
        break
    fy = os.path.join(OD, f"kz_{y}.npz")
    have = {}
    if os.path.exists(fy):
        z = np.load(fy); have = {pd.Timestamp(str(d)): z["data"][i] for i, d in enumerate(z["dates"])}; lat, lon = z["lat"], z["lon"]
    todo = [d for d in dd if d not in have]
    if todo:
        with ThreadPoolExecutor(4) as ex:
            for i, (d, res) in enumerate(ex.map(fetch, todo)):
                if res is not None:
                    have[d], lat, lon = res
                if i % 100 == 0:
                    print(f"  {y}: {i}/{len(todo)}", flush=True)
        ds = sorted(have)
        np.savez_compressed(fy, dates=np.array([str(d.date()) for d in ds]), vars=np.array(VARS),
                            data=np.stack([have[d] for d in ds]).astype(np.float32), lat=lat, lon=lon)
    cube[y] = (have, lat, lon)
    print(f"{y}: days {len(have)} (missing {sum(d not in have for d in dd)})", flush=True)

if os.environ.get("REVERSE"):
    sys.exit(0)


def cell(lat, lon, la, lo, field):
    """Ближайшая ячейка с данными (FWI не считается над водой / пустыней без растительности - там NaN), поиск до 2 ячеек."""
    i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
    best = None
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ii, jj = i + di, j + dj
            if 0 <= ii < len(lat) and 0 <= jj < len(lon) and np.isfinite(field[ii, jj]):
                dist = np.hypot(lat[ii] - la, (lon[jj] - lo) * np.cos(np.deg2rad(la)))
                if best is None or dist < best[0]:
                    best = (dist, ii, jj)
    return best


for f, t in zip(IDX, tabs):
    t = t.drop(columns=[c for c in t if c.startswith("gfwed_")])
    d0 = avail_day(t.t_utc); rows = []
    for r, d in zip(t.itertuples(), d0):
        have, lat, lon = cube[d.year]
        o = {}
        if d in have:
            b = cell(lat, lon, r.lat, r.lon, have[d][VARS.index("FWI")])
            if b is not None:
                _, ii, jj = b
                o = {f"gfwed_{v}": float(have[d][k, ii, jj]) for k, v in enumerate(VARS)}
                prev = [have[d - pd.Timedelta(days=k)][VARS.index("FWI"), ii, jj] for k in range(3) if d - pd.Timedelta(days=k) in have]
                o["gfwed_FWI_max3d"] = float(np.nanmax(prev))
        rows.append(o)
    t = pd.concat([t, pd.DataFrame(rows, index=t.index)], axis=1)
    t.to_csv(f, index=False)
    ok = t.get("gfwed_FWI", pd.Series(np.nan, index=t.index)).notna()
    print(os.path.basename(f), "| slices", len(t), "| with FWI", round(float(ok.mean()), 3), flush=True)
    print(t.loc[ok, [f"gfwed_{v}" for v in VARS]].describe().round(1).to_string(), flush=True)
