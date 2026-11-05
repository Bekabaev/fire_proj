"""Задание 2, итерации (ITERATIONS.md, it6+), шаг 28. Ошибки прогноза GFS относительно ERA5 на val 2022 в нормированном
пространстве признаков погоды (weather_vec): R = vec(gfsa) - vec(era5). При обучении (07, AUG_GFS=1) к погоде ERA5 случайного
среза добавляется случайная строка R - модель привыкает к неточному прогнозу, как на сайте (в train 2013-21 GFS нет).
Выход: out_v3/gfs_resid.npy [n, 43] f32. Тест не используется."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

si = pd.read_csv(os.path.join(CACHE, "val_idx.csv"), parse_dates=["t_utc"], low_memory=False)
stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
a, b = weather_vec(si, stats, "gfsa"), weather_vec(si, stats, "era5")
ok = (a[:, -3] == 0) & (b[:, -3] == 0)                      # будущее есть в обоих источниках
R = (a - b)[ok]; R[:, -3:] = 0
np.save(os.path.join(OUT, "gfs_resid.npy"), R.astype(np.float32))
print("rows", len(R), "of", len(si), "| mean |R| by dim:", np.round(np.abs(R).mean(0)[:40], 2).tolist())
