"""Шаг 16. Рисунки и сводная статистика расширенной выборки (Казахстан, Global Fire Atlas) для отчёта по датасету.
Выход: figures/fig7_kz_map.png, fig8_kz_years.png, fig9_kz_slice_example.png, data/ext_kz/kz_stats.json
"""
import os, sys, json
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

ev = pd.read_csv(os.path.join(KZ, "events_kz.csv")); ev = ev[ev.year >= 2013]
si = pd.read_csv(os.path.join(KZ, "slices_index.csv"))
part = lambda y: "обучение U-Net (2013–2020)" if y <= 2020 else ("поправка и настройка (2021–2022)" if y <= 2022 else "тест (2023–2024)")
ev["part"] = ev.year.map(part); si["part"] = si.year.map(part)
COL = {"обучение U-Net (2013–2020)": "#1f77b4", "поправка и настройка (2021–2022)": "#ff7f0e", "тест (2023–2024)": "#d62728"}
kz = gpd.read_file(KZ_BOUNDARY)
res = pd.read_csv(os.path.join(DATA, "events_spread.csv"))

fig, ax = plt.subplots(figsize=(11, 6))
kz.boundary.plot(ax=ax, color="k", lw=0.8)
for p, g in ev.groupby("part"):
    ax.scatter(g.lon, g.lat, s=np.clip(np.sqrt(g.area_km2) * 1.5, 1, 40), c=COL[p], alpha=0.45, lw=0, label=f"{p}: {len(g)} пожаров")
ax.scatter(res.ign_lon, res.ign_lat, marker="*", s=90, c="lime", edgecolor="k", lw=0.6, label="экспертные пожары заповедников (44)", zorder=5)
ax.set_xlim(46, 88); ax.set_ylim(40.3, 55.8); ax.set_xlabel("долгота, °"); ax.set_ylabel("широта, °")
ax.legend(loc="lower left", fontsize=8.5, markerscale=1.5); ax.set_title("Пожары Global Fire Atlas в Казахстане, 2013–2024 (размер точки ~ площадь)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig7_kz_map.png"), dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
y = ev.groupby("year").agg(fires=("fire_id", "size"), area=("area_km2", "sum")); s_ = si.groupby("year").size()
axs[0].bar(y.index, y.fires, color=[COL[part(v)] for v in y.index]); axs[0].set_ylabel("пожаров")
ax2 = axs[0].twinx(); ax2.plot(s_.index, s_.values, "k.-"); ax2.set_ylabel("срезов (линия)")
axs[0].set_title("Пожары и срезы по годам (2024 - до августа)", fontsize=10)
axs[1].hist(np.log10(ev.area_km2), bins=40, color="#7f7f7f"); axs[1].set_xlabel("площадь пожара, км² (лог. шкала)")
axs[1].set_xticks([0, 1, 2, 3]); axs[1].set_xticklabels(["1", "10", "100", "1000"]); axs[1].set_ylabel("пожаров")
axs[1].set_title(f"Площадь: медиана {ev.area_km2.median():.1f} км², 90 % - {ev.area_km2.quantile(.9):.0f} км²", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig8_kz_years.png"), dpi=150); plt.close()

ch = json.load(open(os.path.join(KZ, "slices_channels.json")))["channels"]
q = si[(si.year == 2019) & (si.new_72h_km2 >= 15) & (si.af_24h > 5) & (si.state_km2 > 10)].sort_values("new_72h_km2").iloc[0]
d = np.load(os.path.join(KZ, "slices", "2019.npz")); X = d["X"][q.slice].astype(np.float32); Y = d["Y"][q.slice]; V = d["valid"][q.slice]
rr, cc = np.nonzero((X[ch.index("burned_frac")] > .5) | (Y[2] > 0)); r0 = int(np.clip(rr.mean() - 24, 0, 48)); c0 = int(np.clip(cc.mean() - 24, 0, 48))
X = X[:, r0:r0 + 48, c0:c0 + 48]; Y = Y[:, r0:r0 + 48, c0:c0 + 48]; V = V[r0:r0 + 48, c0:c0 + 48]
show = [("burned_frac", "Выгорело к моменту t", "Reds"), ("af_count_24h", "Термоточки FIRMS за сутки", "hot_r"), ("prior_burn", "Прошлые гари", "Purples"),
        ("grass_frac", "Доля степи (WorldCover)", "YlGn"), ("ndvi", "NDVI до пожара (MOD13A1)", "BrBG"), ("elev", "Высота, м (GLO-90)", "terrain")]
fig, axs = plt.subplots(2, 5, figsize=(19, 7.6))
for ax, (k, t, cm) in zip(axs.ravel()[:6], show):
    im = ax.imshow(X[ch.index(k)], cmap=cm, interpolation="nearest"); ax.set_title(t, fontsize=10); plt.colorbar(im, ax=ax, fraction=.046)
for i, h in enumerate((24, 48, 72)):
    ax = axs.ravel()[6 + i]; ax.imshow(np.where(V, Y[i], np.nan), cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(X[ch.index("burned_frac")] > .5, levels=[.5], colors="r", linewidths=.8); ax.set_title(f"Цель: новое выгорание за {h} ч", fontsize=10)
ax = axs.ravel()[9]; ax.imshow(V, cmap="gray", interpolation="nearest"); ax.set_title("valid (чёрное - исключено: уже выгорело\nили чужой пожар)", fontsize=10)
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Срез Казахстана: пожар {q.fire_id}, {q.t_utc[:10]}, {q.lat:.2f}° с. ш., {q.lon:.2f}° в. д.; выгорело {q.state_km2:.0f} км², "
             f"прирост 24/48/72 ч = {q.new_24h_km2:.0f}/{q.new_48h_km2:.0f}/{q.new_72h_km2:.0f} км² (центральный фрагмент 48×48 ячеек = 24×24 км)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig9_kz_slice_example.png"), dpi=120); plt.close()

nf = sum(len(pd.read_csv(os.path.join(CACHE, "firms_kz", f"firms_kz_{y}.csv.gz"), usecols=["type"])) for y in range(2012, 2025))
st = dict(fires=len(ev), slices=len(si), years="2013–2024",
          by_part={p: dict(fires=int((ev.part == p).sum()), slices=int((si.part == p).sum())) for p in COL},
          by_year=y.assign(slices=s_).reset_index().to_dict("records"),
          area_median=float(ev.area_km2.median()), area_p90=float(ev.area_km2.quantile(.9)), area_max=float(ev.area_km2.max()),
          days_median=float(ev.n_days.median()), slices_per_fire_median=float(si.groupby("fire_id").size().median()),
          growth_cells_median={h: float((si[f"new_{h}h_km2"] / 0.25).median()) for h in (24, 48, 72)},
          growth_share={h: float((si[f"new_{h}h_km2"] > 0).mean()) for h in (24, 48, 72)},
          in_scene_share={h: float(si[f"new_{h}h_km2"].sum() / si[f"new_{h}h_km2_total"].sum()) for h in (24, 48, 72)},
          firms_points=int(nf), weather_share_2021_2024=float(si[si.year >= 2021].weather_ok.mean()),
          gfa_landcover=ev.gfa_landcover.value_counts(normalize=True).round(3).head(5).to_dict())
json.dump(st, open(os.path.join(KZ, "kz_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(st, ensure_ascii=False, indent=1, default=float)[:2500])
