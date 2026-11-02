"""Шаг 23 (датасет v3). Погода на каждый срез v3 (шаг 22) из трёх источников с явной пометкой доступности на момент t.

era5_*   ERA5 (шаг 13) в точке пожара: окна [t, t+12/24/48/72 ч) и 7 сут до t. Реанализ = «идеальный прогноз», для обучения;
         в оперативном режиме будущих значений нет.
st_*     метеостанция ISD (шаг 20), ближайшая в радиусе 75 км с ≥ 4 сроками за 24 ч до t: наблюдения за [t − 24 ч, t] -
         ДОСТУПНЫ на момент прогноза. st_fut_* - наблюдения в окнах после t: только для проверки, НЕ признаки.
fc_<m>_* прогнозы GFS / ECMWF IFS (шаг 21, только 2024): окно [t + 24k, t + 24(k+1)) из прогона previous_day(k+2) -
         выпущен не позже t, ДОСТУПНЫ на момент прогноза.
Флаги: weather_ok (ERA5 есть), station_ok, forecast_ok. Веса меток: w_label_<h> - средний вес целей среза по источнику
(1 / (1 + неопределённость/12 ч)) - для взвешенного лосса.
Выход: data/ext_kz/slices_v3_index.csv, data/ext_kz/qa_weather_v3.csv (ERA5 / прогноз против станций)
RESERVES=1 - то же для срезов заповедников (шаг 27): ERA5 из data/weather_hourly/<event>.csv (шаг 2, тот же models=era5),
выход slices_v3_reserves_index.csv, qa_weather_v3_reserves.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from rasterio.warp import transform as warp_transform
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

HZ = (12, 24, 48, 72); R_ST = 75e3
RES = bool(os.environ.get("RESERVES")); SFX = "_reserves" if RES else ""
si = pd.read_csv(os.path.join(KZ, f"slices_v3{SFX}_base.csv"), parse_dates=["t_utc"])
wmap = pd.read_csv(os.path.join(KZ, "weather_map.csv")).set_index("fire_id").key.to_dict()
fmap = pd.read_csv(os.path.join(KZ, "forecast_map.csv")).set_index("fire_id").key.to_dict() if os.path.exists(os.path.join(KZ, "forecast_map.csv")) else {}
FC_MODELS = {"gfs": "gfs_seamless", "ifs": "ecmwf_ifs025"}


def uv(ws, wd):
    return (ws * np.sin(np.deg2rad(wd))).mean(), (ws * np.cos(np.deg2rad(wd))).mean()


def era5_feat(h, t):
    o = {}
    for H in HZ:
        w = h.loc[t:t + pd.Timedelta(hours=H - 1)]
        if len(w) < H * 0.8:
            continue
        u, v = uv(w.wind_speed_10m, w.wind_direction_10m)
        o.update({f"era5_T_max_{H}h": w.temperature_2m.max(), f"era5_RH_min_{H}h": w.relative_humidity_2m.min(),
                  f"era5_VPD_mean_{H}h": w.vapour_pressure_deficit.mean(), f"era5_ws_mean_{H}h": w.wind_speed_10m.mean(),
                  f"era5_gust_max_{H}h": w.wind_gusts_10m.max(), f"era5_wind_u_{H}h": u, f"era5_wind_v_{H}h": v,
                  f"era5_P_sum_{H}h": w.precipitation.sum()})
    a = h.loc[t - pd.Timedelta(days=7):t]
    if len(a) > 100:
        o.update(era5_P_sum_prev7d=a.precipitation.sum(), era5_VPD_mean_prev7d=a.vapour_pressure_deficit.mean(),
                 era5_soilm_t=a.soil_moisture_0_to_7cm.iloc[-1], era5_RH_min_prev24h=a.relative_humidity_2m.iloc[-24:].min(),
                 era5_ws_mean_prev24h=a.wind_speed_10m.iloc[-24:].mean())
    return o


def st_win(o, t0, t1):
    w = o[(o.time_utc >= t0) & (o.time_utc < t1)]
    if len(w) == 0:
        return None
    u, v = uv(w.ws.fillna(0), w.wd.fillna(0))
    return dict(ws_mean=w.ws.mean(), ws_max=w.ws.max(), T_max=w["T"].max(), RH_min=w.RH.min(), wind_u=u, wind_v=v,
                P_sum=w.p1.clip(lower=0).sum(min_count=1), n=len(w))


def fc_feat(h, t, tag):
    o = {}
    for k in range(3):
        w = h.loc[t + pd.Timedelta(hours=24 * k):t + pd.Timedelta(hours=24 * k + 23)]
        n = k + 2
        cols = {b: f"{b}_previous_day{n}" for b in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation")}
        if len(w) < 20 or w[cols["wind_speed_10m"]].isna().mean() > 0.2:
            return o
        o[f"_w{k}"] = w[list(cols.values())].rename(columns={v: b for b, v in cols.items()})
    parts = [o.pop(f"_w{k}") for k in range(3)]
    res = {}
    for H in HZ:
        w = pd.concat(parts)[:H] if H > 24 else parts[0][:H]
        u, v = uv(w.wind_speed_10m, w.wind_direction_10m)
        res.update({f"fc_{tag}_T_max_{H}h": w.temperature_2m.max(), f"fc_{tag}_RH_min_{H}h": w.relative_humidity_2m.min(),
                    f"fc_{tag}_ws_mean_{H}h": w.wind_speed_10m.mean(), f"fc_{tag}_gust_max_{H}h": w.wind_gusts_10m.max(),
                    f"fc_{tag}_wind_u_{H}h": u, f"fc_{tag}_wind_v_{H}h": v, f"fc_{tag}_P_sum_{H}h": w.precipitation.sum()})
    return res


# --- станции
sta = pd.read_csv(os.path.join(KZ, "stations", "stations.csv"), dtype={"station": str})
sx, sy = warp_transform("EPSG:4326", CRS, sta.lon.tolist(), sta.lat.tolist()); stree = cKDTree(np.c_[sx, sy])
px, py = warp_transform("EPSG:4326", CRS, si.lon.tolist(), si.lat.tolist())
cand = stree.query_ball_point(np.c_[px, py], R_ST)
obs = {}
feats = []
cache_e, cache_f = {}, {}
for i, r in enumerate(si.itertuples()):
    o = {}
    if RES:
        k = r.event; f = os.path.join(DATA, "weather_hourly", f"{k}.csv")
    else:
        k = wmap.get(r.fire_id); f = os.path.join(KZ, "weather", f"{k}.csv")
    if k not in cache_e:
        cache_e = {k: pd.read_csv(f, parse_dates=["time"]).set_index("time") if os.path.exists(f) else None}
    if cache_e[k] is not None:
        o.update(era5_feat(cache_e[k], r.t_utc))
    if r.year not in obs:
        obs = {r.year: {s: g.sort_values("time_utc") for s, g in pd.read_csv(os.path.join(KZ, "stations", f"obs_{r.year}.csv.gz"),
                                                                              dtype={"station": str}, parse_dates=["time_utc"]).groupby("station")}}
    js = sorted(cand[i], key=lambda j: np.hypot(sx[j] - px[i], sy[j] - py[i]))
    for j in js:
        g = obs[r.year].get(sta.station[j])
        if g is None:
            continue
        p = st_win(g, r.t_utc - pd.Timedelta(hours=24), r.t_utc + pd.Timedelta(minutes=1))
        if p is None or p["n"] < 4:
            continue
        o.update({f"st_{a}_prev24h": b for a, b in p.items()}); o["st_id"] = sta.station[j]
        o["st_dist_km"] = float(np.hypot(sx[j] - px[i], sy[j] - py[i]) / 1000)
        for H in HZ:
            q = st_win(g, r.t_utc, r.t_utc + pd.Timedelta(hours=H))
            if q is not None:
                o.update({f"st_fut_{a}_{H}h": b for a, b in q.items() if a in ("ws_mean", "ws_max", "T_max", "RH_min", "wind_u", "wind_v")})
        break
    fk = fmap.get(r.fire_id)
    if fk is not None:
        for tag, m in FC_MODELS.items():
            ff = os.path.join(KZ, "forecast", m, f"{fk}.csv")
            if (m, fk) not in cache_f:
                cache_f = {kk: vv for kk, vv in cache_f.items() if kk[0] != m}
                cache_f[(m, fk)] = pd.read_csv(ff, parse_dates=["time"]).set_index("time") if os.path.exists(ff) else None
            if cache_f[(m, fk)] is not None:
                o.update(fc_feat(cache_f[(m, fk)], r.t_utc, tag))
    feats.append(o)
    if i % 10000 == 0:
        print(f"  {i}/{len(si)}", flush=True)
W = pd.DataFrame(feats, index=si.index)
si = pd.concat([si, W], axis=1)
si["weather_ok"] = si.get("era5_ws_mean_24h", pd.Series(np.nan, index=si.index)).notna()
si["station_ok"] = si.get("st_ws_mean_prev24h", pd.Series(np.nan, index=si.index)).notna()
si["forecast_ok"] = si.get("fc_gfs_ws_mean_24h", pd.Series(np.nan, index=si.index)).notna()
# средний вес целей по источнику метки: src1 6 ч, src2/3 12 ч, src4 24 ч, src5 6 ч -> w = 1 / (1 + unc / 12)
w_af, w_day = 1 / (1 + 9 / 12), 1 / (1 + 24 / 12)        # src 1–2 (≈ 9 ч в среднем) / src 4; остальное ≈ src 3/5
for H in HZ:
    af, day = si[f"new_{H}h_src_af"].fillna(0), si[f"new_{H}h_src_day"].fillna(0)
    si[f"w_label_{H}h"] = (af * w_af + day * w_day + (1 - af - day).clip(lower=0) * (1 / (1 + 10 / 12))).where(si[f"new_{H}h_km2"] > 0)
si.to_csv(os.path.join(KZ, f"slices_v3{SFX}_index.csv"), index=False)
print("slices:", len(si), "| ERA5:", round(si.weather_ok.mean(), 3), "| station:", round(si.station_ok.mean(), 3),
      "| forecast (2024):", int(si.forecast_ok.sum()), flush=True)
print("by year ERA5 / station share:\n", si.groupby("year")[["weather_ok", "station_ok"]].mean().round(2).to_string())

# --- QA: ERA5 и прогнозы против станций (окна после t, 24 ч)
rows = []
for src, cols in [("era5", "era5_{}_24h"), ("gfs", "fc_gfs_{}_24h"), ("ifs", "fc_ifs_{}_24h")]:
    for v in ("ws_mean", "T_max", "RH_min"):
        a, b = cols.format(v), f"st_fut_{v}_24h"
        if a not in si or b not in si:
            continue
        m = si[[a, b]].dropna()
        if src != "era5":
            m = si.loc[si.forecast_ok, [a, b]].dropna()
        if len(m) < 30:
            continue
        rows.append(dict(source=src, var=v, n=len(m), bias=round(float((m[a] - m[b]).mean()), 2),
                         mae=round(float((m[a] - m[b]).abs().mean()), 2), r=round(float(m[a].corr(m[b])), 3)))
    if src == "era5" and "era5_ws_mean_24h" in si:      # то же подмножество 2024 для честного сравнения с прогнозами
        for v in ("ws_mean", "T_max", "RH_min"):
            a, b = f"era5_{v}_24h", f"st_fut_{v}_24h"
            m = si.loc[si.forecast_ok, [a, b]].dropna() if b in si else pd.DataFrame()
            if len(m) >= 30:
                rows.append(dict(source="era5_2024", var=v, n=len(m), bias=round(float((m[a] - m[b]).mean()), 2),
                                 mae=round(float((m[a] - m[b]).abs().mean()), 2), r=round(float(m[a].corr(m[b])), 3)))
q = pd.DataFrame(rows); q.to_csv(os.path.join(KZ, f"qa_weather_v3{SFX}.csv"), index=False)
print("\nweather vs stations (24 h after t):\n", q.to_string())
