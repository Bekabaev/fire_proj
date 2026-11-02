"""Шаг 32. Рисунки для презентации датасета (без упоминания версий): figures/pres/*.png"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OUT = os.path.join(FIG, "pres")
os.makedirs(OUT, exist_ok=True)

# карта: пожары, вошедшие в срезы, по частям разбиения
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv"))
si = pd.read_csv(os.path.join(KZ, "slices_v3_index.csv"), usecols=["fire_id", "split"]).drop_duplicates("fire_id")
ev = ev.drop(columns=[c for c in ["split"] if c in ev]).merge(si, on="fire_id")
LAB = {"train": "обучение (≤ 2021)", "val": "настройка (2022)", "test": "тест (2023–2024)"}
COL = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#d62728"}
kz = gpd.read_file(KZ_BOUNDARY)
res = pd.read_csv(os.path.join(DATA, "events_spread.csv"))
fig, ax = plt.subplots(figsize=(11, 6))
kz.boundary.plot(ax=ax, color="k", lw=0.8)
for p in ["train", "val", "test"]:
    g = ev[ev.split == p]
    ax.scatter(g.lon, g.lat, s=np.clip(np.sqrt(g.area_km2) * 1.5, 1, 40), c=COL[p], alpha=0.45, lw=0,
               label=f"{LAB[p]}: {len(g)} пожаров")
ax.scatter(res.ign_lon, res.ign_lat, marker="*", s=90, c="lime", edgecolor="k", lw=0.6,
           label="экспертные пожары заповедников (44)", zorder=5)
ax.set_xlim(46, 88); ax.set_ylim(40.3, 55.8); ax.set_xlabel("долгота, °"); ax.set_ylabel("широта, °")
ax.legend(loc="lower left", fontsize=8.5, markerscale=1.5)
ax.set_title("Пожары Казахстана в датасете, 2013–2024 (размер точки ~ площадь)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "kz_map.png"), dpi=150); plt.close()
print("fires on map:", len(ev), ev.split.value_counts().to_dict())


def crop(src, box, dst):
    im = Image.open(os.path.join(FIG, src))
    w, h = im.size
    im.crop(tuple(int(round(v)) for v in (box[0] * w, box[1] * h, box[2] * w, box[3] * h))).save(os.path.join(OUT, dst))


crop("fig10_v3_label_sources.png", (0, 0, 0.425, 1), "label_sources.png")      # только источники по годам
crop("fig13_perims_vs_labels.png", (0, 0.2, 0.755, 0.78), "perims.png")      # три снимка без диаграммы версий
crop("fig11_v3_slice_example.png", (0, 0.055, 1, 1), "slice_example.png")    # без заголовка «Срез v3»
