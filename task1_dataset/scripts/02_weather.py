"""Шаг 2. Метеорологические данные ERA5 (через Open-Meteo Historical Weather API, models=era5, 0.25°).

Для каждого события - почасовой ряд в точке возгорания от min(1 марта, дата пожара − 60 сут) до даты снимка + 1 сут.
Дневной ряд в местный полдень + индексы FWI (FFMC, DMC, DC, ISI, BUI, FWI), разгон с 1 марта / −60 сут.
Признаки события: погода в час возгорания, агрегаты за 24/48/72 ч, осадки до пожара (антецедентные), влажность почвы,
FWI в день пожара, согласованность направления ветра с направлением распространения.
Выход: data/weather_hourly/<event>.csv, data/weather_daily/<event>.csv, data/event_weather_features.csv
"""
import os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
import fwi

VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "snowfall", "snow_depth",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "vapour_pressure_deficit", "surface_pressure",
        "cloud_cover", "shortwave_radiation", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm", "soil_temperature_0_to_7cm"]
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(ev):
    f = os.path.join(DATA, "weather_hourly", ev.event + ".csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["time"])
    t0 = ev.ign_time_utc.normalize()
    start = min(pd.Timestamp(year=t0.year, month=3, day=1), t0 - pd.Timedelta(days=60))
    end = max(ev.image_date, t0 + pd.Timedelta(days=3)) + pd.Timedelta(days=1)
    q = dict(latitude=ev.ign_lat, longitude=ev.ign_lon, start_date=start.date().isoformat(), end_date=end.date().isoformat(),
             hourly=",".join(VARS), models="era5", wind_speed_unit="ms", timezone="GMT")
    for k in range(5):
        try:
            r = requests.get(URL, params=q, timeout=120); r.raise_for_status(); break
        except Exception as e:
            print("  retry", k, e); time.sleep(10 * (k + 1))
    j = r.json()
    h = pd.DataFrame(j["hourly"]); h["time"] = pd.to_datetime(h["time"])
    h.insert(1, "grid_lat", j["latitude"]); h.insert(2, "grid_lon", j["longitude"]); h.insert(3, "grid_elev", j.get("elevation"))
    h.to_csv(f, index=False)
    time.sleep(1)
    return h


def daily_fwi(h, lon):
    noon = int(round(12 - lon / 15))          # местный солнечный полдень в UTC
    h = h.set_index("time")
    rain24 = h["precipitation"].rolling("24h").sum()   # сумма за 24 ч, оканчивающиеся в полдень
    d = h[h.index.hour == noon].copy()
    d = pd.DataFrame(dict(date=d.index.normalize(), t=d.temperature_2m, rh=d.relative_humidity_2m.clip(0, 100),
                          ws_kmh=d.wind_speed_10m * 3.6, rain24=rain24.loc[d.index].values,
                          snow_depth=d.snow_depth, vpd=d.vapour_pressure_deficit, sm0_7=d.soil_moisture_0_to_7cm)).reset_index(drop=True)
    d = d.dropna(subset=["t", "rh", "ws_kmh"]).reset_index(drop=True)
    d = pd.concat([d, pd.DataFrame(fwi.run(d))], axis=1)
    d["precip_day"] = d["rain24"]
    return d


def features(ev, h, d):
    h = h.set_index("time"); t0 = ev.ign_time_utc.floor("h")
    o = dict(event=ev.event, era5_grid_lat=h.grid_lat.iloc[0], era5_grid_lon=h.grid_lon.iloc[0], era5_grid_elev=h.grid_elev.iloc[0])
    r0 = h.loc[t0]
    o.update(ign_T=r0.temperature_2m, ign_RH=r0.relative_humidity_2m, ign_VPD=r0.vapour_pressure_deficit,
             ign_ws=r0.wind_speed_10m, ign_wd=r0.wind_direction_10m, ign_gust=r0.wind_gusts_10m, ign_hour_utc=t0.hour)
    for H in (24, 48, 72):
        w = h.loc[t0:t0 + pd.Timedelta(hours=H - 1)]
        o.update({f"T_mean_{H}h": w.temperature_2m.mean(), f"T_max_{H}h": w.temperature_2m.max(),
                  f"RH_min_{H}h": w.relative_humidity_2m.min(), f"RH_mean_{H}h": w.relative_humidity_2m.mean(),
                  f"VPD_mean_{H}h": w.vapour_pressure_deficit.mean(), f"ws_mean_{H}h": w.wind_speed_10m.mean(),
                  f"ws_max_{H}h": w.wind_speed_10m.max(), f"gust_max_{H}h": w.wind_gusts_10m.max(),
                  f"wd_mean_{H}h": circ_mean_deg(w.wind_direction_10m, w.wind_speed_10m),
                  f"wd_steadiness_{H}h": float(np.hypot((w.wind_speed_10m * np.sin(np.deg2rad(w.wind_direction_10m))).sum(),
                                                        (w.wind_speed_10m * np.cos(np.deg2rad(w.wind_direction_10m))).sum())
                                               / max(w.wind_speed_10m.sum(), 1e-6)),
                  f"P_sum_{H}h": w.precipitation.sum()})
    # осадки во время пожара: часы до первых осадков >= 1 мм/ч и до накопления 2 мм
    end = ev.image_date + pd.Timedelta(hours=6) if ev.duration_ok else t0 + pd.Timedelta(days=5)
    w = h.loc[t0:end]
    o["P_during_fire"] = w.precipitation.sum()
    hit = w.index[w.precipitation >= 1.0]
    o["h_to_rain1mm"] = (hit[0] - t0).total_seconds() / 3600 if len(hit) else np.nan
    cum = w.precipitation.cumsum(); hit = cum.index[cum >= 2.0]
    o["h_to_cum2mm"] = (hit[0] - t0).total_seconds() / 3600 if len(hit) else np.nan
    # антецедентные условия
    for D in (7, 14, 30, 60):
        o[f"P_prev_{D}d"] = h.loc[t0 - pd.Timedelta(days=D):t0 - pd.Timedelta(hours=1)].precipitation.sum()
    dd = h.precipitation.resample("D").sum(); dd = dd[dd.index < t0.normalize()]
    wet = dd.index[dd >= 1.0]
    o["days_since_rain1mm"] = (t0.normalize() - wet[-1]).days if len(wet) else np.nan
    pw = h.loc[t0 - pd.Timedelta(days=30):t0]
    o.update(T_mean_prev30d=pw.temperature_2m.mean(), VPD_mean_prev7d=h.loc[t0 - pd.Timedelta(days=7):t0].vapour_pressure_deficit.mean(),
             SM0_7_prev7d=h.loc[t0 - pd.Timedelta(days=7):t0].soil_moisture_0_to_7cm.mean(),
             SM7_28_prev7d=h.loc[t0 - pd.Timedelta(days=7):t0].soil_moisture_7_to_28cm.mean(),
             snow_depth_ign=r0.snow_depth)
    # FWI: день пожара (местная дата) и максимум за первые 3 дня
    dloc = (ev.ign_time_utc + pd.Timedelta(hours=5)).normalize()
    fd = d[d.date == dloc]
    for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI"):
        o[f"{k}_fire_day"] = float(fd[k].iloc[0]) if len(fd) else np.nan
    o["FWI_max_3d"] = float(d[(d.date >= dloc) & (d.date < dloc + pd.Timedelta(days=3))].FWI.max())
    o["fwi_spinup_days"] = int((dloc - d.date.iloc[0]).days)
    # ветер vs наблюдаемое направление распространения (ветер дует ИЗ wd -> огонь идёт в wd+180)
    down = (o["wd_mean_24h"] + 180) % 360
    o["downwind_az_24h"] = down
    o["align_far_24h"] = float(np.cos(np.deg2rad(ev.az_far_deg - down)))
    o["align_centroid_24h"] = float(np.cos(np.deg2rad(ev.az_centroid_deg - down)))
    return o


ev = load_events(); rows = []
for e in ev.itertuples():
    h = fetch(e)
    d = daily_fwi(h, e.ign_lon); d.to_csv(os.path.join(DATA, "weather_daily", e.event + ".csv"), index=False)
    o = features(e, h, d); rows.append(o)
    print(f"{e.event:24s} n_h={len(h):5d} T={o['ign_T']:5.1f} RH={o['ign_RH']:4.0f} ws={o['ign_ws']:4.1f} wd={o['ign_wd']:5.0f} "
          f"P30={o['P_prev_30d']:5.1f} FWI={o['FWI_fire_day']:5.1f} align_far={o['align_far_24h']:+.2f}", flush=True)
wf = pd.DataFrame(rows)
wf.to_csv(os.path.join(DATA, "event_weather_features.csv"), index=False)
m = ev.merge(wf, on="event")
ok = m[m.ign_src == "firms_point"]
print("\nevents:", len(wf), "| features:", wf.shape[1] - 1)
print("mean cos(spread_az - downwind_24h): far %.2f, centroid %.2f (n=%d; >0 = fire spread downwind)" %
      (ok.align_far_24h.mean(), ok.align_centroid_24h.mean(), len(ok)))
print("share align_centroid>0: %.2f" % (ok.align_centroid_24h > 0).mean())
print(wf.describe().T[["mean", "min", "max"]].round(2).to_string())
