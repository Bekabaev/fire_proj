"""Шаг 20 (датасет v3). Наблюдения метеостанций Казахстана (NOAA ISD-Lite, синоп, ≈ 3-часовые сроки), 2013–2024.

Нужны для (а) проверки ERA5/GFS по станциям, (б) признаков «наблюдённая погода до момента прогноза» (доступны оперативно).
ISD-Lite: https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/ (T, точка росы, давление, направление/скорость ветра, облачность, осадки).
Станции: isd-history.csv, страна KZ, работавшие в 2013–2024.
Выход: data/ext_cache/isd/<usaf>-<wban>-<год>.gz (кэш), data/ext_kz/stations/stations.csv,
       data/ext_kz/stations/obs_<год>.csv.gz (station, time_utc, T, Td, RH, ws, wd, P1h)
"""
import os, sys, io, gzip, time
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

C = os.path.join(CACHE, "isd"); os.makedirs(C, exist_ok=True)
OD = os.path.join(KZ, "stations"); os.makedirs(OD, exist_ok=True)
URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{f}"

hist = os.path.join(C, "isd-history.csv")
if not os.path.exists(hist):
    open(hist, "wb").write(requests.get("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv", timeout=120).content)
st = pd.read_csv(hist, dtype=str)
st = st[(st.CTRY == "KZ") & (st.END >= "20130101") & (st.BEGIN <= "20241231")].copy()
st["lat"] = st.LAT.astype(float); st["lon"] = st.LON.astype(float); st["elev"] = pd.to_numeric(st["ELEV(M)"], errors="coerce")
st["station"] = st.USAF + "-" + st.WBAN
st = st[st.lat.notna() & (st.lat != 0)]
st[["station", "STATION NAME", "lat", "lon", "elev", "BEGIN", "END"]].rename(columns={"STATION NAME": "name"}).to_csv(os.path.join(OD, "stations.csv"), index=False)
print("KZ stations active 2013–2024:", len(st), flush=True)


def fetch(a):
    s, y = a
    f = os.path.join(C, f"{s}-{y}.gz")
    if os.path.exists(f) or os.path.exists(f + ".missing"):
        return
    for k in range(4):
        try:
            r = requests.get(URL.format(y=y, f=f"{s}-{y}.gz"), timeout=120)
            if r.status_code == 404:
                open(f + ".missing", "w").close(); return
            if r.status_code == 200:
                open(f + ".part", "wb").write(r.content); os.replace(f + ".part", f); return
        except Exception:
            pass
        time.sleep(5 * (k + 1))


jobs = [(s, y) for y in range(2013, 2025) for s, b, e in zip(st.station, st.BEGIN, st.END) if b[:4] <= str(y) <= e[:4]]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, jobs))
print("downloaded", len(jobs), "station-years requested", flush=True)

names = ["y", "m", "d", "h", "T", "Td", "slp", "wd", "ws", "sky", "p1", "p6"]
for y in range(2013, 2025):
    fo = os.path.join(OD, f"obs_{y}.csv.gz")
    if os.path.exists(fo):
        continue
    parts = []
    for s in st.station:
        f = os.path.join(C, f"{s}-{y}.gz")
        if not os.path.exists(f):
            continue
        a = pd.read_csv(f, sep=r"\s+", header=None, names=names, na_values=[-9999], compression="gzip")
        if a.empty:
            continue
        a = a[(a.m >= 3) & (a.m <= 11)]
        a["time_utc"] = pd.to_datetime(dict(year=a.y, month=a.m, day=a.d, hour=a.h))
        a["T"] /= 10; a["Td"] /= 10; a["ws"] /= 10; a["p1"] /= 10
        es = lambda t: 6.112 * np.exp(17.62 * t / (243.12 + t))
        a["RH"] = (100 * es(a.Td) / es(a["T"])).clip(0, 100)
        a["station"] = s
        parts.append(a[["station", "time_utc", "T", "Td", "RH", "ws", "wd", "p1"]])
    if parts:
        o = pd.concat(parts); o.to_csv(fo, index=False, compression="gzip")
        print(y, "stations:", o.station.nunique(), "obs:", len(o), "| median obs per station-day:",
              round(float(o.groupby([o.station, o.time_utc.dt.date]).size().median()), 1), flush=True)
print("done")
