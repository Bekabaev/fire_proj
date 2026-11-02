"""Шаг 19 (датасет v3). Состояние горючего и контекст тушения на национальной сетке 500 м (из уже скачанных данных + OSM).

Горючее (MOD13A1 NDVI, шаг 12; GFA, шаг 11):
- ndvi_clim.npz   - многолетняя медиана NDVI 2012–2024 для каждого из 17 композитов (сетка 1 км, int16 ×1e-4)
                    -> в срезах: аномалия NDVI = NDVI − норма того же композита (засуха / ранняя «зрелость» травы)
- ndvi_peak_<год>.npy - максимум NDVI за сезон (сетка 1 км) -> в срезах берётся прошлый год: запас прошлогодней сухой травы
- last_burn_<год>.npy - последний год выгорания до этого года (GFA 2012…год−1; 0 - не горело) -> «лет с прошлой гари»
Контекст тушения (OSM, Geofabrik, снимок 2026; поселения и дороги меняются медленно):
- osm.npz - расстояние до ближайшего населённого пункта (city/town/village/hamlet), до дороги (motorway…tertiary + track),
            плотность дорог (км/км² в окне 5 × 5 ячеек). Прямых данных о тушении в открытом доступе нет - это заместители
            «доступности для тушения» и источников антропогенных возгораний.
"""
import os, sys, zipfile
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

# ---------- NDVI норма и пик сезона
fc = os.path.join(KZ, "ndvi_clim.npz")
nd = {y: np.load(os.path.join(KZ, f"ndvi_{y}.npz")) for y in YEARS}
doys = np.array(sorted(set().union(*[set(nd[y]["doy"].tolist()) for y in YEARS])))   # в 2022, 2023 нет по одному композиту
if not os.path.exists(fc):
    clim = []
    for k in range(len(doys)):
        s = np.stack([nd[y]["ndvi"][list(nd[y]["doy"]).index(doys[k])] for y in YEARS if doys[k] in nd[y]["doy"]]).astype(np.float32)
        s[s < -20000] = np.nan
        clim.append(np.nan_to_num(np.nanmedian(s, 0), nan=-32768).astype(np.int16)); print("clim composite", doys[k], flush=True)
    np.savez_compressed(fc, ndvi=np.stack(clim), doy=doys)
for y in YEARS:
    fp = os.path.join(KZ, f"ndvi_peak_{y}.npy")
    if not os.path.exists(fp):
        a = nd[y]["ndvi"].astype(np.float32); a[a < -20000] = np.nan
        np.save(fp, np.nan_to_num(np.nanmax(a, 0), nan=-32768).astype(np.int16))
print("ndvi done", flush=True)

# ---------- последний год выгорания
last = np.zeros((GH, GW), np.int16)
for y in YEARS:
    fl = os.path.join(KZ, f"last_burn_{y}.npy")
    if not os.path.exists(fl):
        np.save(fl, last)
    last = np.where(np.asarray(load_doy(y)) > 0, y, last).astype(np.int16)
print("last burn done", flush=True)

# ---------- OSM
fo = os.path.join(KZ, "osm.npz")
if not os.path.exists(fo):
    z = os.path.join(CACHE, "osm", "kazakhstan-latest-free.shp.zip")
    pl = gpd.read_file(f"zip://{z}!gis_osm_places_free_1.shp")
    pl = pl[pl.fclass.isin(["city", "town", "village", "hamlet"])].to_crs(CRS)
    rd = gpd.read_file(f"zip://{z}!gis_osm_roads_free_1.shp", columns=["fclass"])
    rd = rd[rd.fclass.isin(["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "track",
                            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"])].to_crs(CRS)
    print("places:", len(pl), "roads:", len(rd), flush=True)
    sh = (GH, GW)
    pm = rasterize(((g, 1) for g in pl.geometry), out_shape=sh, transform=GTR, fill=0, dtype="uint8")
    rm = rasterize(((g, 1) for g in rd.geometry), out_shape=sh, transform=GTR, fill=0, dtype="uint8", all_touched=True)
    d_pl = ndimage.distance_transform_edt(pm == 0) * CELL / 1000
    d_rd = ndimage.distance_transform_edt(rm == 0) * CELL / 1000
    # длина дорог в ячейке ≈ число ячеек дороги × 0,5 км; плотность в окне 5 × 5 (2,5 × 2,5 км)
    dens = ndimage.uniform_filter(rm.astype(np.float32), 5) * 25 * 0.5 / (25 * 0.25)
    np.savez_compressed(fo, dist_settlement_km=d_pl.astype(np.float16), dist_road_km=d_rd.astype(np.float16),
                        road_density=dens.astype(np.float16))
    print("osm done | median dist settlement / road km:", float(np.median(d_pl)), float(np.median(d_rd)), flush=True)
print("done")
