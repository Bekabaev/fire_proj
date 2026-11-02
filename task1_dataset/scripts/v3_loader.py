"""Загрузка датасета v3 (шаги 18–23) для обучения. Пример:

    import sys; sys.path.insert(0, "project_till_nov/task1_dataset/scripts")
    from v3_loader import *
    si = load_index(split="train")                       # индекс срезов + погода + флаги
    for f, rows in si.groupby("file"):
        X, Y, valid, lsrc = load_chunk(f, rows.slice.values)    # X [n,23,96,96] f32 (NaN → 0 + маски), Y [n,4,96,96]
        W = label_weights(lsrc, Y, horizon=1)            # веса ячеек для горизонта 24 ч
Экспертные пожары заповедников (шаг 27, метки по контурам экспертов): load_index(kind="reserves") / load_chunk(f, sl, kind="reserves"),
сцены 1 км - kind="reserves_1km"; колонка gfa_fire_ids - те же пожары в GFA, исключать их из обучения при тесте на заповедниках.
Источник метки lsrc: 1 VIIRS, 2 MODIS, 3 интерполяция, 4 только дата MCD64A1, 5 VIIRS вне контура MCD64A1 (0 - не этот пожар).
"""
import os, json
import numpy as np, pandas as pd

KZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ext_kz")
META = json.load(open(os.path.join(KZ, "slices_v3_channels.json"), encoding="utf8"))
CH, TARGETS = META["channels"], META["targets"]          # TARGETS: new_12h, new_24h, new_48h, new_72h
UNC = {int(k): v for k, v in META["lsrc_uncertainty_h"].items()}
INDEX = {"500m": "slices_v3_index.csv", "1km": "slices_1km_base.csv",
         "reserves": "slices_v3_reserves_index.csv", "reserves_1km": "slices_1km_reserves_base.csv"}
DIRS = {"500m": "slices_v3", "1km": "slices_1km", "reserves": "slices_v3_reserves", "reserves_1km": "slices_1km_reserves"}


def load_index(split=None, need_weather=False, need_station=False, kind="500m"):
    """Индекс срезов. kind="1km" - крупные пожары (сцена 96 км). need_* - оставить срезы с ERA5 / станцией."""
    f = os.path.join(KZ, INDEX[kind])
    if kind in ("500m", "reserves") and not os.path.exists(f):
        f = f.replace("_index.csv", "_base.csv")         # до шага 23 - без погоды
    si = pd.read_csv(f, parse_dates=["t_utc"], low_memory=False)
    if split is not None:
        si = si[si.split == split]
    if need_weather:
        si = si[si.weather_ok]
    if need_station:
        si = si[si.station_ok]
    return si


def load_chunk(file, slices=None, kind="500m"):
    d = np.load(os.path.join(KZ, DIRS[kind], file))
    sl = slice(None) if slices is None else np.asarray(slices)
    return d["X"][sl].astype(np.float32), d["Y"][sl], d["valid"][sl], d["lsrc"][sl]


def label_weights(lsrc, Y, horizon, mcd_only=False, day_only_weight=None):
    """Вес ячейки для лосса на горизонте TARGETS[horizon]: 1 / (1 + неопределённость / 12 ч) для целевых ячеек, 1 - для фона.
    mcd_only=True - вариант без расширения VIIRS (src=5 исключаются из целей и из лосса).
    day_only_weight - явный вес для src=4 (например 0 на 12 ч, где ошибка даты MCD64A1 больше горизонта)."""
    w = np.ones(lsrc.shape, np.float32)
    tgt = Y[:, horizon].astype(bool)
    for k, u in UNC.items():
        w[tgt & (lsrc == k)] = 1 / (1 + u / 12)
    if day_only_weight is not None:
        w[tgt & (lsrc == 4)] = day_only_weight
    if mcd_only:
        w[lsrc == 5] = 0
    return w


def fill_nan(X):
    """NaN → 0 и маски наличия для каналов с пропусками (ndvi, ndvi_anom, ndvi_peak_prev, elev, slope_deg)."""
    m = ~np.isfinite(X)
    miss = [CH.index(c) for c in ("ndvi", "ndvi_anom", "ndvi_peak_prev", "elev", "slope_deg")]
    masks = (~m[:, miss]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0), masks
