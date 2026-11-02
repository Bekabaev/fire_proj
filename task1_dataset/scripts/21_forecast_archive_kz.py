"""Шаг 21 (датасет v3). Архив оперативных прогнозов погоды с известной заблаговременностью для пожаров 2024 г. (тест).

Open-Meteo Previous Runs API (previous-runs-api.open-meteo.com): переменная `<var>_previous_dayN` - значение на час τ из
прогноза, выпущенного примерно за N суток до τ. Архив с заблаговременностью есть только с 2024 г.; для 2021–2023 у Open-Meteo
есть лишь «склеенные» первые часы прогонов (почти анализ) - для честной проверки «на прогнозе» не годятся.
Модели: GFS (gfs_seamless) и ECMWF IFS 0,25° (ecmwf_ifs025). N = 1…4. Переменные: T, RH, ветер (скорость, направление, порывы), осадки.
Правило без утечки (шаг 23): для окна [t + 24k, t + 24(k+1)) берётся previous_day(k+2) - прогон не позже t − 0…24 ч,
с запасом на задержку публикации (GFS ≈ 4–5 ч).
Группировка запросов - как в шаге 13 (ячейка 0,25°, пересекающиеся окна пожаров одного года).
Выход: data/ext_kz/forecast/<model>/<key>.csv, data/ext_kz/forecast_map.csv (fire_id -> key)
"""
import os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

MODELS = ["gfs_seamless", "ecmwf_ifs025"]
BASE = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation"]
VARS = ",".join(f"{v}_previous_day{n}" for v in BASE for n in (1, 2, 3, 4))
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv")); ev = ev[ev.year >= 2024]
ev["t0"] = pd.to_datetime(ev.year.astype(str)) + pd.to_timedelta(ev.doy_min - 1 - 3, "D")
ev["t1"] = pd.to_datetime(ev.year.astype(str)) + pd.to_timedelta(ev.doy_max - 1 + 5, "D")
ev["glat"] = (ev.lat / 0.25).round() * 0.25; ev["glon"] = (ev.lon / 0.25).round() * 0.25
groups, mp = [], []
for (y, la, lo), g in ev.sort_values("t0").groupby(["year", "glat", "glon"]):
    cur = None
    for r in g.itertuples():
        if cur is not None and r.t0 <= cur["t1"] and (max(cur["t1"], r.t1) - cur["t0"]).days <= 40:
            cur["t1"] = max(cur["t1"], r.t1); cur["fires"].append(r.fire_id)
        else:
            cur = dict(year=y, lat=la, lon=lo, t0=r.t0, t1=r.t1, fires=[r.fire_id]); groups.append(cur)
for g in groups:
    g["key"] = f"{g['year']}_{g['lat']:.2f}_{g['lon']:.2f}_{g['t0']:%m%d}"
    mp += [dict(fire_id=f, key=g["key"]) for f in g["fires"]]
pd.DataFrame(mp).to_csv(os.path.join(KZ, "forecast_map.csv"), index=False)
print(f"fires {len(ev)} -> requests {len(groups)} per model", flush=True)
for m in MODELS:
    od = os.path.join(KZ, "forecast", m); os.makedirs(od, exist_ok=True)
    for i, g in enumerate(groups):
        f = os.path.join(od, g["key"] + ".csv")
        if os.path.exists(f):
            continue
        q = dict(latitude=g["lat"], longitude=g["lon"], start_date=str(g["t0"].date()), end_date=str(g["t1"].date()), models=m,
                 hourly=VARS, wind_speed_unit="ms", timezone="GMT")
        for k in range(10):
            try:
                r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params=q, timeout=120)
                if r.status_code == 200:
                    pd.DataFrame(r.json()["hourly"]).to_csv(f, index=False); break
                print("  http", r.status_code, r.text[:120], flush=True)
                time.sleep(60 if r.status_code == 429 else 10 * (k + 1))
            except Exception as e:
                print("  retry", str(e)[:100], flush=True); time.sleep(10 * (k + 1))
        time.sleep(0.3)
        if i % 100 == 0:
            print(f"  {m} {i}/{len(groups)}", flush=True)
print("done")
