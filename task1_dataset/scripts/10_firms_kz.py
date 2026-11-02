"""Шаг 10 (расширение выборки). Термоточки NASA FIRMS по всему Казахстану, архив SP, сезоны 2012–2024 (март–ноябрь).

Нужны для срезов пожаров Global Fire Atlas (шаги 11–13): канал «термоточки за 24 ч до t», как в FireSpread_Report_2.
Источники: VIIRS S-NPP (375 м) и MODIS C6.1 (1 км), VIIRS NOAA-20 с 2018-04. API area/csv: ≤ 5 суток за запрос.
Ключ - файл .firms_key (в логи не выводится). Кэш запросов: data/ext_cache/firms_kz/raw/.
Выход: data/ext_cache/firms_kz/firms_kz_<год>.csv.gz (lat, lon, time_utc, frp, conf, daynight, type, source)
"""
import os, sys, io, time
import pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

KEY = open(os.path.join(TASK, ".firms_key")).read().strip()
OD = os.path.join(DATA, "ext_cache", "firms_kz"); RAW = os.path.join(OD, "raw"); os.makedirs(RAW, exist_ok=True)
BBOX = "46.4,40.5,87.4,55.5"            # Казахстан
SRC = [("VIIRS_SNPP_SP", "2012-01-20"), ("MODIS_SP", "2000-11-01"), ("VIIRS_NOAA20_SP", "2018-04-01")]
YEARS = range(2012, 2025)


def get(url):
    for k in range(8):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200 and not r.text.startswith("Invalid"):
                return r.text
            print("  http", r.status_code, r.text[:80], flush=True)
        except Exception as e:
            print("  retry", k, str(e)[:80], flush=True)
        time.sleep(20 * (k + 1))
    raise RuntimeError("FIRMS request failed")


YRS = [int(v) for v in os.environ["ONLY_YEARS"].split(",")] if os.environ.get("ONLY_YEARS") else YEARS   # ONLY_YEARS=2018,2019 - доп. процесс
for y in (YRS[::-1] if os.environ.get("REVERSE") else YRS):   # REVERSE=1 - второй параллельный процесс с конца
    fy = os.path.join(OD, f"firms_kz_{y}.csv.gz")
    if os.path.exists(fy):
        continue
    parts = []
    for src, start in SRC:
        d, d1 = max(pd.Timestamp(f"{y}-03-01"), pd.Timestamp(start)), pd.Timestamp(f"{y}-11-30")
        while d <= d1:
            n = int(min(5, (d1 - d).days + 1)); f = os.path.join(RAW, f"{src}_{d.date()}_{n}.csv")
            if not os.path.exists(f):
                txt = get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{src}/{BBOX}/{n}/{d.date()}")
                open(f + ".part", "w", encoding="utf8").write(txt); os.replace(f + ".part", f)
            q = pd.read_csv(f)
            if len(q):
                q["source"] = src; parts.append(q)
            d += pd.Timedelta(days=n)
    a = pd.concat(parts, ignore_index=True)
    a["time_utc"] = pd.to_datetime(a.acq_date) + pd.to_timedelta(a.acq_time.astype(int) // 100, "h") + pd.to_timedelta(a.acq_time.astype(int) % 100, "m")
    a = a.rename(columns={"latitude": "lat", "longitude": "lon", "confidence": "conf"})
    a = a[["lat", "lon", "time_utc", "frp", "conf", "daynight", "type", "source"]].sort_values("time_utc")
    a.to_csv(fy, index=False, compression="gzip")
    print(y, len(a), a.source.value_counts().to_dict(), flush=True)
print("done")
