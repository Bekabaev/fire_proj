"""Шаг 1. Таблица событий для модели распространения.

Из атрибутов полигонов заповедников берём: время первой термоточки (FireData, UTC), координату термоточки
(Coordinate) = точка возгорания, сенсор термоточки; из полигона - финальный контур; из events.csv - дату снимка,
по которому оцифрован контур (верхняя граница длительности пожара).
Производные: площадь, периметр, максимальная дистанция от точки возгорания до контура, азимут основного
направления распространения, нижняя оценка средней скорости распространения (ROS).
Выход: data/events_spread.csv, data/events_spread.gpkg (слои perimeter, ignition).
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

arch = pd.read_csv(ARCH_EVENTS)
gm = pd.read_csv(GEE_MAP).set_index("event")
rows, perims, igns = [], [], []
for r in arch.itertuples():
    g = read_poly(r.site, r.event)
    crs = g.crs
    geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.unary_union
    a = g.iloc[0]
    ign_time = pd.to_datetime(str(a["FireData"]), format="%Y-%m-%d-%H:%M")
    # точка возгорания: координата термоточки, проверяем, что она у контура (<= 5 км)
    ll = parse_coord(a["Coordinate"]); src, dist_km = "firms_point", np.nan
    if ll is not None:
        p = gpd.GeoSeries([Point(ll[1], ll[0])], crs=4326).to_crs(crs).iloc[0]
        dist_km = geom.distance(p) / 1000
        if dist_km > 5:
            src = "centroid(bad_coord)"
    else:
        src = "centroid(no_coord)"
    if src != "firms_point":
        p = geom.representative_point()
    p_ll = gpd.GeoSeries([p], crs=crs).to_crs(4326).iloc[0]
    # геометрия распространения
    bnd = np.asarray(geom.convex_hull.exterior.coords)
    d = np.hypot(bnd[:, 0] - p.x, bnd[:, 1] - p.y); far = bnd[d.argmax()]
    c = geom.centroid
    az_far = (np.degrees(np.arctan2(far[0] - p.x, far[1] - p.y))) % 360    # азимут (от севера по часовой)
    az_cen = (np.degrees(np.arctan2(c.x - p.x, c.y - p.y))) % 360
    img = pd.Timestamp(r.image_date)
    dur_h = (img + pd.Timedelta(hours=6) - ign_time).total_seconds() / 3600  # снимок ~06 UTC
    if dur_h < 6:   # снимок раньше/в момент термоточки: контур нарисован не после пожара или дата термоточки не первая
        dur_h = np.nan
    area_ha = geom.area / 1e4; perim_km = geom.length / 1000
    m = gm.loc[r.event] if r.event in gm.index else None
    rows.append(dict(
        event=r.event, site=r.site, year=ign_time.year, month=ign_time.month, season=r.season,
        ign_time_utc=ign_time, af_sensor=str(a["Sentinel"]).strip(), ign_lat=round(p_ll.y, 5), ign_lon=round(p_ll.x, 5),
        ign_src=src, ign_to_poly_km=round(dist_km, 3) if np.isfinite(dist_km) else np.nan,
        utm_epsg=crs.to_epsg(), ign_x=round(p.x, 1), ign_y=round(p.y, 1),
        image_date=img, image_sensor=r.sensor, lag_days=r.lag_days, max_duration_h=round(dur_h, 1),
        area_ha=round(area_ha, 1), perimeter_km=round(perim_km, 2), n_parts=getattr(geom, "geoms", [geom]).__len__() if geom.geom_type.startswith("Multi") else 1,
        shape_index=round(perim_km * 1000 / (2 * np.sqrt(np.pi * geom.area)), 3),
        max_spread_km=round(d.max() / 1000, 2), az_far_deg=round(az_far, 1), az_centroid_deg=round(az_cen, 1),
        centroid_offset_km=round(np.hypot(c.x - p.x, c.y - p.y) / 1000, 2),
        ros_lb_m_per_h=round(d.max() / dur_h, 1), duration_ok=bool(np.isfinite(dur_h)),
        gee_post=(m is not None and isinstance(m["post"], str)), gee_pre=(m is not None and isinstance(m["pre"], str)),
        scene_id=r.scene_id))
    perims.append(dict(event=r.event, geometry=gpd.GeoSeries([geom], crs=crs).to_crs(4326).iloc[0]))
    igns.append(dict(event=r.event, ign_src=src, geometry=p_ll))

ev = pd.DataFrame(rows).sort_values(["site", "ign_time_utc"]).reset_index(drop=True)
# GEE: годность композитов из статьи (снег -> непригоден)
ge = pd.read_csv(os.path.join(ROOT, "analysis", "out", "gee_events.csv")).set_index("event")
ev["gee_post_ok"] = ev.event.map(ge.post_ok).fillna(False).astype(bool)
ev["gee_pre_ok"] = ev.event.map(ge.pre_ok).fillna(False).astype(bool)
# разбиения: по заповедникам (LORO) и по времени
ev["split_time"] = np.where(ev.year <= 2019, "train", np.where(ev.year == 2020, "val", "test"))
ev.to_csv(os.path.join(DATA, "events_spread.csv"), index=False)
gp = os.path.join(DATA, "events_spread.gpkg")
if os.path.exists(gp):
    os.remove(gp)
pg = gpd.GeoDataFrame(pd.DataFrame(perims).merge(ev, on="event"), geometry="geometry", crs=4326)
for col in ("ign_time_utc", "image_date"):
    pg[col] = pg[col].astype(str)
pg.to_file(gp, layer="perimeter", driver="GPKG")
gpd.GeoDataFrame(pd.DataFrame(igns), geometry="geometry", crs=4326).to_file(gp, layer="ignition", driver="GPKG")
pd.set_option("display.width", 250)
print(ev[["event", "ign_time_utc", "af_sensor", "ign_src", "ign_to_poly_km", "area_ha", "max_spread_km", "az_far_deg",
          "max_duration_h", "ros_lb_m_per_h", "gee_post_ok", "gee_pre_ok", "split_time"]].to_string())
print("events:", len(ev), "| ign_src:", ev.ign_src.value_counts().to_dict())
