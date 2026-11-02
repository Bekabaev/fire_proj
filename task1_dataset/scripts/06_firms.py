"""Шаг 6. Термоточки NASA FIRMS (архив SP: MODIS C6.1, VIIRS S-NPP, VIIRS NOAA-20, VIIRS NOAA-21) по каждому пожару.

Окно: bbox куба события (+0.05°), даты: от (возгорание − 1 сут) до даты снимка (или +10 сут, если длительность неизвестна),
не более 45 сут. API area/csv принимает до 5 суток за запрос. Ключ - файл .firms_key (в логи не выводится).
Выход: data/firms/<event>.csv (все точки окна, UTC time, source, dist_to_poly_m, in_poly), data/firms_summary.csv
"""
import os, sys, io, json, time
import numpy as np, pandas as pd, geopandas as gpd, requests
from rasterio.warp import transform_bounds
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

KEY = open(os.path.join(TASK, ".firms_key")).read().strip()
OD = os.path.join(DATA, "firms"); os.makedirs(OD, exist_ok=True)
SRC = [("MODIS_SP", "2000-11-01"), ("VIIRS_SNPP_SP", "2012-01-20"), ("VIIRS_NOAA20_SP", "2018-04-01"), ("VIIRS_NOAA21_SP", "2023-03-01")]
FOOT = {"MODIS_SP": 1000.0, "VIIRS_SNPP_SP": 375.0, "VIIRS_NOAA20_SP": 375.0, "VIIRS_NOAA21_SP": 375.0}   # номинальный пиксель, м


def get(url):
    for k in range(6):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and not r.text.startswith("Invalid"):
                return r.text
            print("  http", r.status_code, r.text[:100])
            if r.status_code == 400:
                break
        except Exception as e:
            print("  retry", k, str(e)[:100])
        time.sleep(15 * (k + 1))
    raise RuntimeError("FIRMS request failed")


ev = load_events(); summ = []
for e in ev.itertuples():
    meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json")))
    t = meta["transform"]; x0, y1 = t[2], t[5]; x1 = x0 + meta["W"] * t[0]; y0 = y1 + meta["H"] * t[4]
    lo0, la0, lo1, la1 = transform_bounds(meta["crs"], "EPSG:4326", x0, y0, x1, y1, densify_pts=21)
    bbox = f"{lo0 - 0.05:.4f},{la0 - 0.05:.4f},{lo1 + 0.05:.4f},{la1 + 0.05:.4f}"
    d0 = e.ign_time_utc.normalize() - pd.Timedelta(days=1)
    d1 = (e.image_date if e.duration_ok else e.ign_time_utc.normalize() + pd.Timedelta(days=10))
    d1 = min(d1, d0 + pd.Timedelta(days=45))
    f = os.path.join(OD, e.event + ".csv")
    if not os.path.exists(f):
        parts = []
        for src, start in SRC:
            if d1 < pd.Timestamp(start):
                continue
            d = max(d0, pd.Timestamp(start))
            while d <= d1:
                n = int(min(5, (d1 - d).days + 1))
                txt = get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{src}/{bbox}/{n}/{d.date()}")
                q = pd.read_csv(io.StringIO(txt))
                if len(q):
                    q["source"] = src; parts.append(q)
                d += pd.Timedelta(days=n)
        a = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "source"])
        if len(a):
            a["time_utc"] = pd.to_datetime(a.acq_date) + pd.to_timedelta(a.acq_time.astype(int) // 100, "h") + pd.to_timedelta(a.acq_time.astype(int) % 100, "m")
            g = gpd.GeoDataFrame(a, geometry=gpd.points_from_xy(a.longitude, a.latitude), crs=4326).to_crs(meta["crs"])
            poly = read_poly(e.site, e.event).to_crs(meta["crs"]).geometry.union_all()
            a["x"] = g.geometry.x.round(1); a["y"] = g.geometry.y.round(1)
            a["dist_to_poly_m"] = g.geometry.distance(poly).round(0)
            a["foot_m"] = a.source.map(FOOT)
            a["in_poly"] = a.dist_to_poly_m <= a.foot_m      # с учётом размера пикселя сенсора
            a["h_since_ign"] = ((a.time_utc - e.ign_time_utc).dt.total_seconds() / 3600).round(2)
            a = a.sort_values("time_utc")
        a.to_csv(f, index=False)
    a = pd.read_csv(f, parse_dates=["time_utc"]) if os.path.getsize(f) > 50 else pd.DataFrame()
    s = dict(event=e.event, n_window=len(a))
    if len(a):
        ip = a[a.in_poly & (a.h_since_ign >= -24)]
        s.update(n_in_poly=len(ip), n_viirs_in_poly=int(ip.source.str.startswith("VIIRS").sum()),
                 n_overpasses=int(ip.time_utc.nunique()), first_h=ip.h_since_ign.min() if len(ip) else np.nan,
                 last_h=ip.h_since_ign.max() if len(ip) else np.nan, frp_sum=round(ip.frp.sum(), 1) if len(ip) else 0)
    summ.append(s)
    print(f"{e.event:24s} {bbox:40s} {d0.date()}..{d1.date()} " + " ".join(f"{k}={v}" for k, v in s.items() if k != "event"), flush=True)
sm = pd.DataFrame(summ); sm.to_csv(os.path.join(DATA, "firms_summary.csv"), index=False)
print("events with >=1 detection in polygon:", int((sm.n_in_poly > 0).sum()), "/", len(sm),
      "| >=3 overpasses:", int((sm.n_overpasses >= 3).sum()), "| total points in polygons:", int(sm.n_in_poly.sum()))
