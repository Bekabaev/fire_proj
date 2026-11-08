"""Погода для прод-движка в формате признаков обучения (task1 шаг 23, task2 common_v3.raw_weather).

Оперативный режим: прогноз GFS (Open-Meteo, models=gfs_seamless - тот же источник, что архив прогнозов шага 21) в точке пожара:
окна [t, t+12/24/48/72 ч): T max, RH min, ветер средний, порывы max, вектор ветра u/v «откуда» (метеоконвенция, как в датасете),
осадки; прошлые 7 сут и 24 ч - из той же выдачи (past_days). FWI - свой расчёт (task1 fwi.py, Van Wagner 1987) по полуденным
значениям за до 92 сут до t (в обучении - NASA GFWED; расчёт по той же методике, на коротком разгоне DMC/DC занижаются - см. README).
Сценарий (симулятор): окна задаются пользователем (ветер, направление, T, RH, осадки), прошлое и FWI - из реальной погоды точки
или значения по умолчанию для сухого лета.
Префикс признаков - fc_gfsa_ (как «прогноз GFS» на тесте), прошлые - era5_* (имена признаков обучения).
"""
import math, sys
import numpy as np, pandas as pd, requests

from .config import HZ, SCRIPTS_T1

sys.path.insert(0, SCRIPTS_T1)
import fwi as fwi_mod                                           # noqa: E402

URL = "https://api.open-meteo.com/v1/forecast"
URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
BASE = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation", "vapour_pressure_deficit"]
DRY_SUMMER = dict(era5_P_sum_prev7d=0.0, era5_VPD_mean_prev7d=2.0, era5_soilm_t=0.15, gfwed_FFMC=90.0, gfwed_DMC=60.0,
                  gfwed_DC=450.0, gfwed_BUI=85.0)


def _get(url, q, timeout):
    r = requests.get(url, params=q, timeout=timeout); r.raise_for_status()
    d = pd.DataFrame(r.json()["hourly"]); d["time"] = pd.to_datetime(d.time)
    return d.set_index("time").astype(float)


class Series:
    """fut - прогноз GFS (будущие окна; последние дни прошлого), hist - ERA5 (прошлое и FWI, как в обучении) с дополнением GFS."""
    def __init__(self, fut, hist):
        self.fut, self.hist = fut, hist


def fetch(lat, lon, t=None, hist_days=100, timeout=60):
    """Ряды Open-Meteo (UTC) для срока t: прогноз GFS (gfs_seamless, past_days=10, 5 сут вперёд) + архив ERA5 за hist_days до t.
    Если t старше 10 сут (повтор прошлого пожара) - прогноза нет, будущие окна берутся из ERA5 («идеальный прогноз», флаг weather_src)."""
    t = pd.Timestamp(t or pd.Timestamp.utcnow().tz_localize(None)).floor("h")
    now = pd.Timestamp.utcnow().tz_localize(None)
    loc = dict(latitude=round(float(lat), 4), longitude=round(float(lon), 4), wind_speed_unit="ms", timezone="GMT")
    fut = None
    if t > now - pd.Timedelta(days=9):
        fut = _get(URL, dict(loc, hourly=",".join(BASE + ["soil_moisture_0_to_10cm"]), models="gfs_seamless", past_days=10, forecast_days=5), timeout)
        fut["soil_moisture_0_to_7cm"] = fut.pop("soil_moisture_0_to_10cm")
    end = min(t + pd.Timedelta(days=4), now - pd.Timedelta(days=1))
    hist = _get(URL_ARCHIVE, dict(loc, hourly=",".join(BASE + ["soil_moisture_0_to_7cm"]), models="era5",
                                  start_date=str((t - pd.Timedelta(days=hist_days)).date()), end_date=str(end.date())), timeout)
    hist = hist.dropna(how="all")
    if fut is not None:
        hist = hist.combine_first(fut[fut.index <= now])
    return Series(fut if fut is not None else hist, hist)


def _uv(ws, wd):
    return float((ws * np.sin(np.deg2rad(wd))).mean()), float((ws * np.cos(np.deg2rad(wd))).mean())


def windows(h, t):
    """Будущие окна [t, t+H) -> fc_gfsa_* (как era5_feat шага 23)."""
    o = {}
    for H in HZ:
        w = h.loc[t:t + pd.Timedelta(hours=H - 1)]
        if len(w) < H * 0.8:
            continue
        u, v = _uv(w.wind_speed_10m, w.wind_direction_10m)
        o.update({f"fc_gfsa_T_max_{H}h": w.temperature_2m.max(), f"fc_gfsa_RH_min_{H}h": w.relative_humidity_2m.min(),
                  f"fc_gfsa_ws_mean_{H}h": w.wind_speed_10m.mean(), f"fc_gfsa_gust_max_{H}h": w.wind_gusts_10m.max(),
                  f"fc_gfsa_wind_u_{H}h": u, f"fc_gfsa_wind_v_{H}h": v, f"fc_gfsa_P_sum_{H}h": w.precipitation.sum()})
    return o


def past(h, t):
    a = h.loc[t - pd.Timedelta(days=7):t]
    if len(a) < 100:
        return {}
    return dict(era5_P_sum_prev7d=a.precipitation.sum(), era5_VPD_mean_prev7d=a.vapour_pressure_deficit.mean(),
                era5_soilm_t=a.soil_moisture_0_to_7cm.iloc[-1], era5_RH_min_prev24h=a.relative_humidity_2m.iloc[-24:].min(),
                era5_ws_mean_prev24h=a.wind_speed_10m.iloc[-24:].mean())


def fwi_codes(h, t, lon):
    """FWI по полуденным (местное солнечное время) значениям до t; признаки gfwed_* (последний день с полуднем ≤ t)."""
    noon_utc = int(round(12 - lon / 15)) % 24
    d = h[h.index.hour == noon_utc]
    d = d[d.index <= t]
    if len(d) < 20:
        return {}
    rain = h.precipitation.rolling(24, min_periods=12).sum().reindex(d.index).fillna(0)
    daily = pd.DataFrame(dict(date=d.index, t=d.temperature_2m.values, rh=d.relative_humidity_2m.values,
                              ws_kmh=d.wind_speed_10m.values * 3.6, rain24=rain.values))
    r = pd.DataFrame(fwi_mod.run(daily)); last = r.iloc[-1]
    return dict(gfwed_FFMC=last.FFMC, gfwed_DMC=last.DMC, gfwed_DC=last.DC, gfwed_ISI=last.ISI, gfwed_BUI=last.BUI,
                gfwed_FWI=last.FWI, gfwed_FWI_max3d=float(r.FWI.iloc[-3:].max()))


def history_row(h, t, lon):
    """Прошлая погода и FWI на срок t из Series."""
    t = pd.Timestamp(t).floor("h")
    return {**past(h.hist, t), **fwi_codes(h.hist, t, lon)}


def forecast_row(lat, lon, t, h=None):
    """Строка погодных признаков на срок t (UTC) по прогнозу GFS. h - готовый Series fetch() (для повторных вызовов прокрутки)."""
    t = pd.Timestamp(t).floor("h")
    h = fetch(lat, lon, t) if h is None else h
    w = windows(h.fut, t)
    src = "era5" if h.fut is h.hist else "gfs"
    if len(w) < 7 * len(HZ):                                 # прогноз короче 72 ч от t (прокрутка) или прошлое - добираем ERA5
        w = {**windows(h.hist, t), **w}; src = "gfs+era5" if h.fut is not h.hist else "era5"
    return {**w, **history_row(h, t, lon), "weather_src": src}, h


def scenario_row(winds, T=30.0, RH=20.0, P=0.0, base=None):
    """Сценарий симулятора. winds - 4 пары (скорость м/с, направление «откуда», °) для окон 12/24/48/72 ч или одна пара на все.
    base - строка прошлой погоды/FWI (реальная точка), иначе сухое лето. FFMC/ISI/FWI пересчитываются под ветер сценария."""
    if len(winds) == 2 and np.isscalar(winds[0]):
        winds = [winds] * len(HZ)
    row = dict(DRY_SUMMER) if base is None else {k: v for k, v in base.items() if not k.startswith("fc_")}
    for H, (ws, wd) in zip(HZ, winds):
        u, v = ws * math.sin(math.radians(wd)), ws * math.cos(math.radians(wd))
        row.update({f"fc_gfsa_T_max_{H}h": T, f"fc_gfsa_RH_min_{H}h": RH, f"fc_gfsa_ws_mean_{H}h": ws, f"fc_gfsa_gust_max_{H}h": 1.5 * ws + 2,
                    f"fc_gfsa_wind_u_{H}h": u, f"fc_gfsa_wind_v_{H}h": v, f"fc_gfsa_P_sum_{H}h": P * H / 24})
    row.setdefault("era5_RH_min_prev24h", RH); row.setdefault("era5_ws_mean_prev24h", winds[0][0])
    ws24 = winds[1][0] * 3.6
    ffmc = row.get("gfwed_FFMC", 90.0)
    ffmc = 90.0 if ffmc is None or not np.isfinite(ffmc) else float(ffmc)
    for _ in range(3):                                          # тонкое топливо за 3 сут приходит к условиям сценария (T, RH, ветер, осадки)
        ffmc = fwi_mod.ffmc(T, RH, ws24, P, ffmc)
    u = row.get("gfwed_BUI", 85.0); u = 85.0 if u is None or not np.isfinite(u) else float(u)
    i = fwi_mod.isi(ws24, ffmc)
    row.update(gfwed_FFMC=ffmc, gfwed_ISI=i, gfwed_FWI=fwi_mod.fwi(i, u), weather_src="scenario")
    row["gfwed_FWI_max3d"] = max(row.get("gfwed_FWI_max3d", 0) or 0, row["gfwed_FWI"])
    return row
