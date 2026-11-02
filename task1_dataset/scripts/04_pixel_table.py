"""Шаг 4. Табличная выборка пикселей для ML (GBM / MLP): пиксельные признаки куба + признаки погоды события.

Для каждого события: до N_POS пикселей гари и до N_NEG негорелых пикселей окна (равномерно), без пикселей ignore.
w = 1 / вероятность попадания в выборку (для несмещённых оценок по окну).
Геометрия относительно возгорания в системе ветра за первые 24 ч: along_wind_m (>0 = по ветру), cross_wind_m.
Цель: label (1 = сгорело в этом пожаре). Колонки qa_* (после пожара) - только для контроля, не входы.
Выход: data/pixel_samples.csv.gz
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

N_POS, N_NEG, SEED = 5000, 5000, 0
ev = load_events().merge(pd.read_csv(os.path.join(DATA, "event_weather_features.csv")), on="event")
# признаки события, известные в момент возгорания (финальная площадь/ROS/азимут гари - НЕ включаем, это цель)
WCOLS = [c for c in pd.read_csv(os.path.join(DATA, "event_weather_features.csv")).columns
         if c not in ("event", "era5_grid_lat", "era5_grid_lon", "align_far_24h", "align_centroid_24h", "h_to_rain1mm",
                      "h_to_cum2mm", "P_during_fire")]
rng = np.random.default_rng(SEED); parts = []
for e in ev.itertuples():
    d = np.load(os.path.join(DATA, "cubes", e.event + ".npz")); meta = json.load(open(os.path.join(DATA, "cubes", e.event + ".json")))
    CH = meta["channels"]; X = d["x"]; lab = d["label"].astype(bool); ok = ~d["ignore"]
    pos = np.flatnonzero(lab & ok); neg = np.flatnonzero(~lab & ok)
    sp = rng.choice(pos, min(N_POS, len(pos)), replace=False); sn = rng.choice(neg, min(N_NEG, len(neg)), replace=False)
    sel = np.concatenate([sp, sn])
    w = np.concatenate([np.full(len(sp), len(pos) / max(len(sp), 1)), np.full(len(sn), len(neg) / max(len(sn), 1))])
    r, c = np.unravel_index(sel, lab.shape)
    t = pd.DataFrame({k: X[i].ravel()[sel].astype(np.float32) for i, k in enumerate(CH)})
    t = t.rename(columns={"post_NBR": "qa_post_NBR", "dNBR": "qa_dNBR"})
    t["landcover"] = d["landcover"].ravel()[sel]
    t["prior_burned"] = np.isfinite(t.prior_burn_days).astype(np.uint8)
    # смещение от точки возгорания (восток, север) и в системе ветра
    dx = (c - meta["ign_rc"][1]) * GRID_M; dy = (meta["ign_rc"][0] - r) * GRID_M
    down = np.deg2rad(e.downwind_az_24h)
    t["along_wind_m"] = dx * np.sin(down) + dy * np.cos(down)
    t["cross_wind_m"] = -dx * np.cos(down) + dy * np.sin(down)
    t["row"], t["col"] = r, c
    t["label"] = lab.ravel()[sel].astype(np.uint8); t["w"] = w
    t.insert(0, "event", e.event); t.insert(1, "site", e.site); t.insert(2, "year", e.year)
    t.insert(3, "split_time", e.split_time); t.insert(4, "gee_pre_ok", e.gee_pre_ok)
    for k in WCOLS:
        t[k] = getattr(e, k)
    parts.append(t)
    print(f"{e.event:24s} pos {len(sp):5d}/{len(pos):8d} neg {len(sn):5d}/{len(neg):8d}", flush=True)
tab = pd.concat(parts, ignore_index=True)
tab.to_csv(os.path.join(DATA, "pixel_samples.csv.gz"), index=False, float_format="%.5g")
print("rows:", len(tab), "| columns:", tab.shape[1], "| burned share:", round(tab.label.mean(), 3))
print("rows with pre-fire fuel:", int(tab.pre_NBR.notna().sum()))
print(tab.groupby("site").agg(events=("event", "nunique"), rows=("label", "size"), pos=("label", "sum")).to_string())
json.dump(dict(pixel_features=[c for c in tab.columns[5:] if c not in ("label", "w", "row", "col") and not c.startswith("qa_") and c not in WCOLS],
               event_features=WCOLS, target="label", weight="w", group_cols=["event", "site", "year", "split_time"],
               qa_only=["qa_post_NBR", "qa_dNBR"]), open(os.path.join(DATA, "pixel_samples_columns.json"), "w"), indent=1)
