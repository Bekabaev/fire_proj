"""Шаг 13 (расширение выборки). Почасовая погода ERA5 (Open-Meteo archive, models=era5) для пожаров Казахстана из шага 11.

Пожары одного года в одной ячейке 0.25° (сетка ERA5) объединяются, если их окна пересекаются; окно = [начало − 7 сут, конец + 4 сут].
Переменные (8): T, RH, ветер 10 м (скорость, направление, порывы), осадки, VPD, влажность почвы 0–7 см.
Лимит Open-Meteo ~10 тыс. запросов/сут: при 429 скрипт ждёт; перезапуск продолжает с места остановки.
Выход: data/ext_kz/weather/<key>.csv, data/ext_kz/weather_map.csv (fire_id -> key)
"""
import os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OD = os.path.join(KZ, "weather"); os.makedirs(OD, exist_ok=True)
VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,vapour_pressure_deficit,soil_moisture_0_to_7cm"
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv")); ev = ev[ev.year >= 2013]   # 2012 - только прошлые гари
ev["t0"] = pd.to_datetime(ev.year.astype(str)) + pd.to_timedelta(ev.doy_min - 1 - 7, "D")
ev["t1"] = pd.to_datetime(ev.year.astype(str)) + pd.to_timedelta(ev.doy_max - 1 + 4, "D")
ev["glat"] = (ev.lat / 0.25).round() * 0.25; ev["glon"] = (ev.lon / 0.25).round() * 0.25
groups, mp = [], []
for (y, la, lo), g in ev.sort_values("t0").groupby(["year", "glat", "glon"]):
    cur = None
    for r in g.itertuples():
        if cur is not None and r.t0 <= cur["t1"] and (max(cur["t1"], r.t1) - cur["t0"]).days <= 40:
            cur["t1"] = max(cur["t1"], r.t1); cur["fires"].append(r.fire_id)
        else:
            cur = dict(year=y, lat=la, lon=lo, t0=r.t0, t1=r.t1, fires=[r.fire_id]); groups.append(cur)
for i, g in enumerate(groups):
    g["key"] = f"{g['year']}_{g['lat']:.2f}_{g['lon']:.2f}_{g['t0']:%m%d}"
    mp += [dict(fire_id=f, key=g["key"]) for f in g["fires"]]
pd.DataFrame(mp).to_csv(os.path.join(KZ, "weather_map.csv"), index=False)
days = sum((g["t1"] - g["t0"]).days + 1 for g in groups)
print(f"fires {len(ev)} -> requests {len(groups)}, total days {days}", flush=True)
t_last = 0
# сначала годы ≥ 2021 (нужны поправке GBM, физике и тесту; U-Net погоду не использует), затем остальные
groups.sort(key=lambda g: (g["year"] < 2021, -g["year"]))
if os.environ.get("REVERSE"):   # второй параллельный процесс: приоритетные годы с другого конца (2021 -> 2024)
    pri = [g for g in groups if g["year"] >= 2021]; groups = pri[::-1] + [g for g in groups if g["year"] < 2021][::-1]
for i, g in enumerate(groups):
    f = os.path.join(OD, g["key"] + ".csv")
    if os.path.exists(f):
        continue
    q = dict(latitude=g["lat"], longitude=g["lon"], start_date=str(g["t0"].date()), end_date=str(g["t1"].date()), models="era5",
             hourly=VARS, wind_speed_unit="ms", timezone="GMT")
    for k in range(10):
        try:
            time.sleep(max(0, 0.25 - (time.time() - t_last))); t_last = time.time()
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=q, timeout=120)
            if r.status_code == 200:
                h = pd.DataFrame(r.json()["hourly"]); h.to_csv(f, index=False); break
            print("  http", r.status_code, r.text[:120], flush=True)
            time.sleep(60 if r.status_code == 429 else 10 * (k + 1))
        except Exception as e:
            print("  retry", str(e)[:100], flush=True); time.sleep(10 * (k + 1))
    if i % 200 == 0:
        print(f"  {i}/{len(groups)}", flush=True)
print("done", len(os.listdir(OD)))
