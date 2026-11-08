"""Пакет для сравнения с ELMFIRE: 40 тестовых пожаров сайта (2023–24, prod/demo_cases.*) -> elmfire_bench/cases/case_XX/.

Запуск (на Windows-ПК с проектом, сервер не нужен):
  ../.venv/Scripts/python.exe -X utf8 elmfire_bench/prepare_package.py        (из project_till_nov)

В каждой case_XX/:
  fbfm40.tif, elev.tif, slope.tif, aspect.tif  - ландшафт ELMFIRE (сетка EPSG:6933, 500 м, 96 x 96)
  start_burned.tif                              - гарь на момент старта t0 (1 = сгорело) -> начальный фронт ELMFIRE
  truth.npz   Y[4,96,96] (1 = сгорело за 12/24/48/72 ч, включая старт), valid, lsrc (источник метки), start
  ours.npz    hybrid_prob[4], hybrid_mask[4], unet_prob[4], fbp_arrival_h (наша физика FBP, ч)
  meta.json   t0, координаты, геопривязка, погода по окнам (сырая GFS + производная для ELMFIRE), FWI
"""
import json, math, os, sys
import numpy as np, rasterio
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prod.config import CH, HZ, CELL, CRS, X0, Y1          # noqa: E402
from prod.engine import FireEngine                         # noqa: E402

OUT = os.path.join(ROOT, "elmfire_bench", "cases")
z = np.load(os.path.join(ROOT, "prod", "demo_cases.npz"))
meta_all = json.load(open(os.path.join(ROOT, "prod", "demo_cases.json"), encoding="utf-8"))
E = FireEngine()


def emc(t_c, rh):
    """Равновесная влажность мёртвого топлива, % (Simard 1968; NFDRS). t_c - °C, rh - %."""
    t = t_c * 9 / 5 + 32
    if rh < 10:
        return 0.03229 + 0.281073 * rh - 0.000578 * rh * t
    if rh < 50:
        return 2.22749 + 0.160107 * rh - 0.01478 * t
    return 21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t - 0.483199 * rh


def tif(path, a, transform, dtype, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=dtype,
                       crs=CRS, transform=transform, nodata=nodata, compress="deflate") as f:
        f.write(a.astype(dtype), 1)


for i, c in enumerate(meta_all["cases"]):
    d = os.path.join(OUT, f"case_{i:02d}"); os.makedirs(d, exist_ok=True)
    X = z["X"][i].astype(np.float32); Y = z["Y"][i]; valid = z["valid"][i]; lsrc = z["lsrc"][i]
    r0, c0, row = c["r0"], c["c0"], c["row"]
    tr = from_origin(X0 + c0 * CELL, Y1 - r0 * CELL, CELL, CELL)
    g = lambda k: X[CH.index(k)]
    fb = np.nan_to_num(g("fbfm40"), nan=99).astype(np.int16)
    fb[~valid] = 99                                                          # вне области данных - негорючее
    elev = np.nan_to_num(g("elev"), nan=float(np.nanmedian(g("elev"))))
    slope = np.nan_to_num(g("slope_deg"), nan=0.0)
    gy, gx = np.gradient(elev, CELL)                                         # строки идут на юг: gy = dz/d(юг)
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360                   # куда обращён склон, от севера по часовой
    aspect[slope < 0.1] = 0
    start = g("burned_frac") > 0.5
    tif(os.path.join(d, "fbfm40.tif"), fb, tr, "int16")
    tif(os.path.join(d, "elev.tif"), elev, tr, "float32")
    tif(os.path.join(d, "slope.tif"), slope, tr, "float32")
    tif(os.path.join(d, "aspect.tif"), aspect, tr, "float32")
    tif(os.path.join(d, "start_burned.tif"), start, tr, "uint8")
    np.savez_compressed(os.path.join(d, "truth.npz"), Y=Y, valid=valid, lsrc=lsrc, start=start)

    o = E.predict_scene(X[None], valid[None], row)
    np.savez_compressed(os.path.join(d, "ours.npz"), hybrid_prob=np.asarray(o["prob"], np.float32),
                        hybrid_mask=np.asarray(o["mask"], bool), unet_prob=np.asarray(o["unet"], np.float32),
                        fbp_arrival_h=np.asarray(o["arrival_h"], np.float32))

    nd = g("ndvi"); nd = float(np.nanmedian(nd[valid])) if np.isfinite(nd[valid]).any() else 0.3
    m_live_herb = float(np.clip(0.3 + (nd - 0.15) / 0.45 * 1.7, 0.3, 2.0) * 100)
    wx = []
    for h in HZ:
        u, v, ws = (row.get(f"fc_gfsa_{k}_{h}h") for k in ("wind_u", "wind_v", "ws_mean"))
        t_max, rh_min = row.get(f"fc_gfsa_T_max_{h}h"), row.get(f"fc_gfsa_RH_min_{h}h")
        ok = all(x is not None and np.isfinite(x) for x in (u, v, ws, t_max, rh_min))
        m1 = max(emc(t_max, rh_min), 2.0) if ok else None
        wx.append(dict(window_h=[0, h], ok=ok,
                       raw=dict(ws_mean_10m_ms=ws, gust_max_ms=row.get(f"fc_gfsa_gust_max_{h}h"), u_ms=u, v_ms=v,
                                T_max_C=t_max, RH_min_pct=rh_min, P_sum_mm=row.get(f"fc_gfsa_P_sum_{h}h")),
                       elmfire=None if not ok else dict(
                           wind_from_deg=round(math.degrees(math.atan2(u, v)) % 360, 1),       # u,v в датасете - вектор «откуда»
                           ws_20ft_mph=round(ws * 0.87 * 2.23694, 2),                          # 10 м -> 20 фт (x0,87), м/с -> миль/ч
                           m1_pct=round(m1, 1), m10_pct=round(m1 + 1, 1), m100_pct=round(m1 + 2, 1),
                           mlh_pct=round(m_live_herb, 0), mlw_pct=120.0)))
    meta = dict(case=i, title=c["title"], fire_id=c["fire_id"], t0_utc=c["t_utc"], lat=c["lat"], lon=c["lon"],
                crs=CRS, cell_m=CELL, size=[int(fb.shape[0]), int(fb.shape[1])], transform_gdal=list(tr.to_gdal()),
                r0=r0, c0=c0, state_km2=c["state_km2"], horizons_h=list(HZ), our_F1_site=c["F1"],
                fuel_codes_present=sorted(np.unique(fb).tolist()), canopy="нет (степь): CC=CH=CBH=CBD=0",
                weather_windows=wx, weather_note="окно k = [t0, t0+H], средние/экстремумы прогноза GFS (прогон t0-12 ч); "
                "для ELMFIRE погоду брать кусочно-постоянной: 0-12 ч из окна 12, 12-24 из 24, 24-48 из 48, 48-72 из 72",
                fwi=dict(FFMC=row.get("gfwed_FFMC"), DMC=row.get("gfwed_DMC"), DC=row.get("gfwed_DC"), ISI=row.get("gfwed_ISI"),
                         BUI=row.get("gfwed_BUI"), FWI=row.get("gfwed_FWI")),
                ndvi_median=round(nd, 3))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(i, c["title"], "fuel", meta["fuel_codes_present"], flush=True)
print("готово:", OUT)
