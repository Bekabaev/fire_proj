"""Шаг 15 (расширение выборки). Погода ERA5 на каждый срез Казахстана: окна [t, t + 24/48/72 ч) и 7 сут до t.
Вход: data/ext_kz/slices_base.csv (шаг 14), data/ext_kz/weather/ (шаг 13). Выход: data/ext_kz/slices_index.csv
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

HZ = (24, 48, 72)
si = pd.read_csv(os.path.join(KZ, "slices_base.csv"), parse_dates=["t_utc"])
wmap = pd.read_csv(os.path.join(KZ, "weather_map.csv")).set_index("fire_id").key.to_dict()


def wfeat(h, t):
    o = {}
    for H in HZ:
        w = h.loc[t:t + pd.Timedelta(hours=H - 1)]
        if len(w) < H * 0.8:
            continue
        u = (w.wind_speed_10m * np.sin(np.deg2rad(w.wind_direction_10m))).mean(); v = (w.wind_speed_10m * np.cos(np.deg2rad(w.wind_direction_10m))).mean()
        o.update({f"T_mean_{H}h": w.temperature_2m.mean(), f"T_max_{H}h": w.temperature_2m.max(), f"RH_min_{H}h": w.relative_humidity_2m.min(),
                  f"VPD_mean_{H}h": w.vapour_pressure_deficit.mean(), f"ws_mean_{H}h": w.wind_speed_10m.mean(),
                  f"gust_max_{H}h": w.wind_gusts_10m.max(), f"wind_u_{H}h": u, f"wind_v_{H}h": v, f"P_sum_{H}h": w.precipitation.sum()})
    a = h.loc[t - pd.Timedelta(days=7):t]
    if len(a) > 100:
        o.update(P_sum_prev7d=a.precipitation.sum(), VPD_mean_prev7d=a.vapour_pressure_deficit.mean(),
                 soilm_t=a.soil_moisture_0_to_7cm.iloc[-1], RH_min_prev24h=a.relative_humidity_2m.iloc[-24:].min())
    return o


cache, feats = {}, []
for r in si.itertuples():
    k = wmap.get(r.fire_id); f = os.path.join(KZ, "weather", f"{k}.csv")
    if k not in cache:
        cache = {k: pd.read_csv(f, parse_dates=["time"]).set_index("time") if os.path.exists(f) else None}
    h = cache[k]
    feats.append(wfeat(h, r.t_utc) if h is not None else {})
W = pd.DataFrame(feats, index=si.index)
si = pd.concat([si, W], axis=1); si["weather_ok"] = si.get("ws_mean_24h", pd.Series(np.nan, index=si.index)).notna()
si.to_csv(os.path.join(KZ, "slices_index.csv"), index=False)
print("slices:", len(si), "| with weather:", int(si.weather_ok.sum()), "| by year with weather:",
      si.groupby("year").weather_ok.mean().round(2).to_dict())
