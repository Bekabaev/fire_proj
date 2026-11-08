"""Маска постоянных источников тепла (заводы, газовые факелы, ТЭЦ) по архиву VIIRS 2012-2025 (task1 шаг 10).

В архиве FIRMS (SP) у каждой точки есть поле type: 0 - пожар растительности, 2 - другой постоянный источник на суше,
3 - источник в море. Пожар растительности горит на клетке 1-3 дня за сезон; завод или факел - десятки дней каждый год.
Клетка 500 м считается постоянным источником, если:
  - точки type 2/3 были хотя бы в STATIC_DAYS разных днях, или
  - в среднем по годам с точками - не меньше PERSIST_DAYS разных дней в году и точки были не меньше чем в PERSIST_YEARS годах.
Маска расширяется на 1 клетку (500 м). Выход: prod/static_sources.npz (r, c - ячейки национальной сетки; days, years - для справки).

Сборка (один раз):  ../.venv/Scripts/python.exe -X utf8 -m prod.static_mask
"""
import os, glob
import numpy as np, pandas as pd
from rasterio.warp import transform as warp
from scipy import ndimage

from .config import ROOT, CRS, CELL, X0, Y1, GH, GW

HERE = os.path.dirname(os.path.abspath(__file__))
F_OUT = os.path.join(HERE, "static_sources.npz")
STATIC_DAYS, PERSIST_DAYS, PERSIST_YEARS = 3, 8, 4


def cells(lon, lat):
    x, y = warp("EPSG:4326", CRS, list(lon), list(lat))
    return ((Y1 - np.array(y)) // CELL).astype(int), ((np.array(x) - X0) // CELL).astype(int)


def build():
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "task1_dataset", "data", "ext_cache", "firms_kz", "firms_kz_*.csv.gz"))):
        a = pd.read_csv(f, usecols=["lat", "lon", "time_utc", "type", "source"])
        a = a[a.source.str.startswith("VIIRS")]
        r, c = cells(a.lon, a.lat)
        parts.append(pd.DataFrame(dict(r=r, c=c, day=a.time_utc.str[:10].values, st=a.type.isin([2, 3]).values)))
        print(os.path.basename(f), len(a), flush=True)
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.r >= 0) & (d.r < GH) & (d.c >= 0) & (d.c < GW)].drop_duplicates()
    d["year"] = d.day.str[:4]
    g = d.groupby(["r", "c"])
    days = g.day.nunique(); years = g.year.nunique(); sdays = d[d.st].groupby(["r", "c"]).day.nunique().reindex(days.index, fill_value=0)
    per_year = days / years
    ok = (sdays >= STATIC_DAYS) | ((per_year >= PERSIST_DAYS) & (years >= PERSIST_YEARS))
    idx = days.index[ok.values]
    rr = np.array([k[0] for k in idx]); cc = np.array([k[1] for k in idx])
    # расширение на 1 клетку - через локальные окна, без растра на всю страну
    r_, c_ = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r_.append(rr + dr); c_.append(cc + dc)
    m = pd.DataFrame(dict(r=np.concatenate(r_), c=np.concatenate(c_))).drop_duplicates()
    m = m[(m.r >= 0) & (m.r < GH) & (m.c >= 0) & (m.c < GW)]
    np.savez_compressed(F_OUT, r=m.r.values.astype(np.int32), c=m.c.values.astype(np.int32),
                        core_r=rr.astype(np.int32), core_c=cc.astype(np.int32),
                        core_days=days[ok].values.astype(np.int32), core_years=years[ok].values.astype(np.int16))
    print(f"клеток с точками {len(days)}; постоянных источников {ok.sum()} (по type {int((sdays >= STATIC_DAYS).sum())}); "
          f"с буфером {len(m)}", flush=True)


def load():
    if not os.path.exists(F_OUT):
        return set()
    z = np.load(F_OUT)
    return set(zip(z["r"].tolist(), z["c"].tolist()))


if __name__ == "__main__":
    build()
