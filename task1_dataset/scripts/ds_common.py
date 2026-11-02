"""Общие пути и функции для датасета распространения пожаров (задание 1)."""
import os, re, json
import numpy as np, pandas as pd, geopandas as gpd

ROOT = r"C:\Users\Ionos Tech\Desktop\burned_area"
TASK = os.path.join(ROOT, "project_till_nov", "task1_dataset")
DATA = os.path.join(TASK, "data")
FIG = os.path.join(TASK, "figures")
for d in (DATA, FIG, os.path.join(DATA, "weather_hourly"), os.path.join(DATA, "weather_daily"), os.path.join(DATA, "cubes")):
    os.makedirs(d, exist_ok=True)

ARCH_EVENTS = os.path.join(ROOT, "analysis", "out", "events.csv")   # 44 события из статьи (01_build_dataset.py)
GEE_MAP = os.path.join(ROOT, "analysis", "out", "gee_map.csv")      # какие GEE-композиты есть у события
GEE_DIR = os.path.join(ROOT, "ext_gee")
RU = {"AltynDala": "алтындала", "Korgalzhyn": "коргалжын", "Naurzum": "наурзум", "Mirgorod": "миргород"}
GRID_M = 30.0        # шаг сетки кубов, м
BUFFER_M = 3000.0    # окно = bbox полигона + 3 км (как в статье)


def poly_path(site, event):
    return os.path.join(ROOT, RU[site], "POLYGONS", event.split(site + "_")[1] + ".shp")


def read_poly(site, event):
    return gpd.read_file(poly_path(site, event))


def parse_coord(s):
    """'49°11'2" N 63°13'23" E' или '49°18'37" N 64,76811 E' -> (lat, lon); None если не разобрать."""
    if not isinstance(s, str):
        return None
    parts = re.split(r"\s*[NС]\s*", s.strip(), maxsplit=1)
    if len(parts) != 2:
        return None

    def one(p):
        p = p.strip().rstrip("EВ").strip().replace(",", ".")
        m = re.match(r"^(\d+(?:\.\d+)?)°\s*(?:(\d+)'\s*)?(?:(\d+(?:\.\d+)?)\"?)?$", p)
        if m:
            d, mi, se = m.groups()
            return float(d) + float(mi or 0) / 60 + float(se or 0) / 3600
        try:
            return float(p)
        except ValueError:
            return None

    lat, lon = one(parts[0]), one(parts[1])
    return (lat, lon) if lat is not None and lon is not None else None


def load_events():
    ev = pd.read_csv(os.path.join(DATA, "events_spread.csv"), parse_dates=["ign_time_utc", "image_date"])
    return ev


def circ_mean_deg(deg, w=None):
    a = np.deg2rad(np.asarray(deg, float)); w = np.ones_like(a) if w is None else np.asarray(w, float)
    return float(np.rad2deg(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) % 360)


def read_firms(event):
    """Термоточки события (data/firms/<event>.csv); пустой DataFrame, если точек нет."""
    a = pd.read_csv(os.path.join(DATA, "firms", event + ".csv"))
    if "time_utc" in a.columns:
        a["time_utc"] = pd.to_datetime(a.time_utc)
    return a
