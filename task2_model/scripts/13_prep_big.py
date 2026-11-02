"""Задание 2 v3, шаг 13. Дополнительный кэш обучения «trainx»: все срезы train с ростом за 72 ч ≥ 10 км², не вошедшие в cache_v3/train
(там не более 4 срезов на пожар). Проверка гипотезы «модели не хватает примеров крупного роста». Формат - как шаг 6."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *
import importlib
prep = importlib.import_module("06_prep_v3")

if __name__ == "__main__":
    tr = prep.parts()["train"][0]
    si = load_index(); rs = load_index(kind="reserves")
    excl = {int(x) for s in rs.gfa_fire_ids.dropna() for x in str(s).split()}
    used = set(zip(tr.file, tr.slice))
    x = si[(si.split == "train") & ~si.fire_id.isin(excl) & (si.new_72h_km2 >= 10)]
    x = x[[k not in used for k in zip(x.file, x.slice)]]
    print("extra big-growth slices:", len(x), "fires", x.fire_id.nunique(), flush=True)
    if not os.path.exists(os.path.join(CACHE, "trainx_ok")):
        prep.build("trainx", x, "500m")
