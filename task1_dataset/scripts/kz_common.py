"""Общее для расширенной выборки «Казахстан, Global Fire Atlas» (шаги 10–14): национальная сетка 500 м EPSG:6933."""
import os
import numpy as np, rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from ds_common import DATA, TASK

KZ = os.path.join(DATA, "ext_kz"); os.makedirs(KZ, exist_ok=True)
CACHE = os.path.join(DATA, "ext_cache")
GFA_DIR = os.path.join(CACHE, "gfa")
KZ_BOUNDARY = os.path.join(CACHE, "kaz_boundary.gpkg")
CRS = "EPSG:6933"; CELL = 500.0; N = 96
YEARS = list(range(2012, 2025))
# сетка: bbox Казахстана (46.6–87.3° в. д., 40.6–55.4° с. ш.) + 60 км
_x, _y = warp_transform("EPSG:4326", CRS, [46.6, 87.3, 46.6, 87.3], [40.6, 40.6, 55.4, 55.4])
X0 = np.floor((min(_x) - 60e3) / CELL) * CELL; Y1 = np.ceil((max(_y) + 60e3) / CELL) * CELL
GW = int(np.ceil((max(_x) + 60e3 - X0) / CELL)); GH = int(np.ceil((Y1 - (min(_y) - 60e3)) / CELL))
GTR = from_origin(X0, Y1, CELL, CELL)


def rc_of_lonlat(lon, lat):
    x, y = warp_transform("EPSG:4326", CRS, list(np.atleast_1d(lon)), list(np.atleast_1d(lat)))
    return ((Y1 - np.array(y)) // CELL).astype(int), ((np.array(x) - X0) // CELL).astype(int)


def doy_path(y):
    return os.path.join(KZ, f"doy_{y}.npy")


def load_doy(y):
    """Дата выгорания (день года, 0 = не горело) на национальной сетке, int16 [GH, GW]."""
    return np.load(doy_path(y), mmap_mode="r")
