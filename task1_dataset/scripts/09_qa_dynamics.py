"""Шаг 9. Контроль динамики: карты времени прихода огня, пример среза 500 м, ERA5 vs GFS, статистика срезов.
Выход: figures/fig4_arrival_examples.png, fig5_slice_example.png, fig6_growth_vs_weather.png, data/qa_dynamics.txt
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

ev = load_events(); arrs = pd.read_csv(os.path.join(DATA, "arrival_summary.csv")); fs = pd.read_csv(os.path.join(DATA, "firms_summary.csv"))
si = pd.read_csv(os.path.join(DATA, "slices_index.csv"), parse_dates=["t_utc"]); sc = json.load(open(os.path.join(DATA, "slices_channels.json")))
out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))

P(f"FIRMS: точек в контурах {int(fs.n_in_poly.sum())}, событий с точками {int((fs.n_in_poly > 0).sum())}/44, медиана пролётов {fs.n_overpasses.median():.0f}")
P(f"  первая термоточка совпадает с FireData (|first_h| <= 1 ч): {int((fs.first_h.abs() <= 1).sum())} событий; раньше FireData > 1 ч: {int((fs.first_h < -1).sum())}")
P(f"Время прихода: доля площади по FIRMS (медиана) {arrs.frac_firms.median():.2f}, по MCD64A1 {arrs.frac_mcd64.median():.2f}, дальний NN {arrs.frac_far.median():.2f}")
P(f"  MCD64A1 обнаруживает (медиана доли контура) {arrs.mcd64_detect_frac.median():.2f}; у {int((arrs.mcd64_detect_frac < 0.1).sum())} событий < 10 %")
P(f"  расхождение FIRMS и MCD64A1, медиана |Δ| {arrs.firms_vs_mcd64_med_abs_days.median():.2f} сут (n={arrs.firms_vs_mcd64_med_abs_days.notna().sum()})")
P(f"  t50 медиана {arrs.t50_h.median():.1f} ч; событий, догоревших за 24 ч (≥ 95 % площади): {int((arrs.burned_by_24h >= .95).sum())}; "
  f"многодневных (t90 > 72 ч): {int((arrs.t90_h > 72).sum())}")
long = arrs[arrs.t50_h > 120].event.tolist()
P(f"  t50 > 5 сут (вероятно, контур объединяет несколько возгораний / долгое тление): {long}")

g = si[si.new_24h_km2 > 0]
P(f"\nСрезы: {len(si)} ({si.event.nunique()} событий), с приростом за 24 ч: {len(g)}; медиана прироста 24/48/72 ч: "
  f"{g.new_24h_km2.median():.1f} / {si.new_48h_km2[si.new_48h_km2 > 0].median():.1f} / {si.new_72h_km2[si.new_72h_km2 > 0].median():.1f} км²")
P(f"  доля нового выгорания за 24 ч среди valid-ячеек сцены (медиана): {(g.new_24h_km2_in_scene / (g.valid_frac * 96 * 96 * 0.25)).median() * 100:.2f} %")
for h in (24, 48, 72):
    q = si[si[f"new_{h}h_km2"] > 0]
    P(f"  прирост внутри сцены 96x96 / весь прирост ({h} ч): медиана по срезам {(q[f'new_{h}h_km2_in_scene'] / q[f'new_{h}h_km2']).median():.3f}, "
      f"по площади {q[f'new_{h}h_km2_in_scene'].sum() / q[f'new_{h}h_km2'].sum():.3f} (крупные пожары не помещаются -> data/slices_full)")
P("  по разбиению split_time:", si.groupby("split_time").agg(slices=("slice", "size"), events=("event", "nunique")).to_dict("index"))
b = si.dropna(subset=["era5_ws_mean_24h", "gfs_ws_mean_24h"])
if len(b):
    for k in ("ws_mean_24h", "T_mean_24h", "RH_min_24h", "P_sum_24h"):
        r = stats.pearsonr(b["era5_" + k], b["gfs_" + k])
        P(f"  ERA5 vs GFS {k:12s}: r={r.statistic:.2f}, смещение GFS−ERA5 {np.mean(b['gfs_' + k] - b['era5_' + k]):+.2f} (n={len(b)})")
for k in ("era5_ws_mean_24h", "era5_VPD_mean_24h", "era5_RH_min_24h", "FWI_day", "era5_P_sum_24h"):
    r = stats.spearmanr(si[k], np.log1p(si.new_24h_km2), nan_policy="omit")
    P(f"  Спирмен {k:18s} vs log(1+прирост 24 ч): ρ={r.statistic:+.2f} (p={r.pvalue:.3f})")

# рис. 4: карты прихода огня
exs = ["AltynDala_2017_07_04", "Korgalzhyn_2020_07_20", "Naurzum_2015_08_03"]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, ex in zip(axs, exs):
    m = json.load(open(os.path.join(DATA, "cubes", ex + ".json"))); a = np.load(os.path.join(DATA, "arrival", ex + ".npz"))
    arr = a["arrival_h"]; t = m["transform"]
    im = ax.imshow(arr / 24, cmap="turbo", vmin=0, vmax=np.nanpercentile(arr, 95) / 24, interpolation="nearest")
    f = read_firms(ex); f = f[f.in_poly & (f.h_since_ign >= -24)]
    ax.scatter((f.x - t[2]) / t[0], (t[5] - f.y) / t[0], c=f.h_since_ign / 24, cmap="turbo", s=6, edgecolor="k", linewidth=.2,
               vmin=0, vmax=np.nanpercentile(arr, 95) / 24)
    ax.plot(m["ign_rc"][1], m["ign_rc"][0], "w*", ms=16, mec="k")
    ax.set_title(f"{ex}\nвремя прихода огня, сут (точки - FIRMS)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=.046)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_arrival_examples.png"), dpi=130); plt.close()

# рис. 5: пример среза 500 м
r = si[(si.event == "AltynDala_2017_07_04") & (si.k_day == 1)].iloc[0]
d = np.load(os.path.join(DATA, "slices", r.event + ".npz")); X = d["X"][r.slice].astype(np.float32); Y = d["Y"][r.slice].astype(np.float32)
C = sc["channels"]
show = [("burned_frac", "Сгорело к t (доля ячейки)", "Reds"), ("af_count_24h", "Термоточки за 24 ч до t", "hot_r"), ("fbfm40", "FBFM40 (из WorldCover)", "tab20"),
        ("pre_NDMI", "NDMI до пожара", "BrBG"), ("prior_burn_frac", "Прошлые гари", "Purples"), ("elev", "Высота, м", "terrain")]
fig, axs = plt.subplots(2, 5, figsize=(19, 7.6))
for ax, (k, tt, cm) in zip(axs.ravel()[:6], show):
    im = ax.imshow(X[C.index(k)], cmap=cm, interpolation="nearest"); ax.set_title(tt, fontsize=10); plt.colorbar(im, ax=ax, fraction=.046)
for i, h in enumerate((24, 48, 72)):
    ax = axs.ravel()[6 + i]; ax.imshow(np.where(d["valid"][r.slice], Y[i], np.nan), cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(X[C.index("burned_frac")] > .5, levels=[.5], colors="r", linewidths=.8); ax.set_title(f"Цель: новое выгорание за {h} ч", fontsize=10)
ax = axs.ravel()[9]; ax.imshow(d["valid"][r.slice], cmap="gray", interpolation="nearest"); ax.set_title("valid (окно куба)", fontsize=10)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Срез 500 м, EPSG:6933, 96×96: {r.event}, t = возгорание + {int(r.t_h)} ч ({r.t_utc:%Y-%m-%d %H:%M} UTC); "
             f"прирост 24/48/72 ч = {r.new_24h_km2:.0f}/{r.new_48h_km2:.0f}/{r.new_72h_km2:.0f} км²", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_slice_example.png"), dpi=120); plt.close()

# рис. 6: прирост vs погода
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, (k, tt) in zip(axs, [("era5_ws_mean_24h", "Ветер, м/с (ERA5, 24 ч)"), ("era5_VPD_mean_24h", "VPD, кПа (ERA5, 24 ч)"), ("FWI_day", "FWI дня t")]):
    ax.scatter(si[k], si.new_24h_km2 + 0.1, c=np.log1p(si.state_km2), cmap="viridis", s=14)
    ax.set_yscale("log"); ax.set_xlabel(tt); ax.set_ylabel("Прирост за 24 ч + 0.1, км²")
    rr = stats.spearmanr(si[k], si.new_24h_km2, nan_policy="omit"); ax.set_title(f"ρ = {rr.statistic:+.2f} (p = {rr.pvalue:.3f})", fontsize=10)
fig.suptitle("Суточный прирост площади по срезам и погода (цвет = log площади к моменту t)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_growth_vs_weather.png"), dpi=130); plt.close()
open(os.path.join(DATA, "qa_dynamics.txt"), "w", encoding="utf8").write("\n".join(out))
