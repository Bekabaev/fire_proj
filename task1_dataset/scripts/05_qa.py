"""Шаг 5. Контроль качества и описательная статистика датасета + рисунки.
Выход: figures/*.png, data/qa_summary.txt
"""
import os, sys, json
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

ev = load_events().merge(pd.read_csv(os.path.join(DATA, "event_weather_features.csv")), on="event")
ci = pd.read_csv(os.path.join(DATA, "cubes_index.csv"))
tab = pd.read_csv(os.path.join(DATA, "pixel_samples.csv.gz"))
out = []
P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
WC = {10: "Деревья", 20: "Кустарник", 30: "Травы (степь)", 40: "Пашня", 50: "Застройка", 60: "Голая почва",
      70: "Снег", 80: "Вода", 90: "Водно-болотные", 95: "Мангры", 100: "Мхи"}

# 1. ветер vs направление распространения
ok = ev[ev.ign_src == "firms_point"]
a = np.deg2rad(ok.az_centroid_deg - ok.downwind_az_24h)
R = np.hypot(np.cos(a).mean(), np.sin(a).mean()); n = len(a)
z = n * R ** 2; p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))   # тест Рэлея
v = np.sqrt(2 * n) * np.cos(a).mean(); p_v = 1 - stats.norm.cdf(v)                              # V-тест (ось = по ветру)
P(f"Ветер (24 ч) vs азимут центроида гари от точки возгорания: n={n}, mean cos={np.cos(a).mean():.2f}, "
  f"доля по ветру={np.mean(np.cos(a) > 0):.2f}, V-тест p={p_v:.4f}, Рэлей p={p_ray:.3f}")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), subplot_kw=dict())
axs[0].remove(); ax = fig.add_subplot(1, 2, 1, projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
d = np.deg2rad((ok.az_centroid_deg - ok.downwind_az_24h) % 360)
ax.hist(d, bins=12, color="#c0504d", alpha=.8); ax.set_title("Направление распространения\nотносительно ветра (0° = по ветру)", fontsize=10)
axs[1].scatter(ok.ws_mean_24h, ok.ros_lb_m_per_h, c=ok.FWI_fire_day, cmap="inferno_r", s=40, edgecolor="k")
axs[1].set_yscale("log"); axs[1].set_xlabel("Средний ветер за 24 ч, м/с"); axs[1].set_ylabel("Нижняя оценка ROS, м/ч")
sc = axs[1].collections[0]; plt.colorbar(sc, ax=axs[1], label="FWI в день пожара")
rho = stats.spearmanr(ok.ws_mean_24h, ok.ros_lb_m_per_h, nan_policy="omit")
axs[1].set_title(f"Ветер и скорость распространения (ρ={rho.statistic:.2f}, p={rho.pvalue:.2f})", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_wind_vs_spread.png"), dpi=150); plt.close()
for k in ("ws_mean_24h", "FWI_fire_day", "VPD_mean_24h", "RH_min_24h", "P_prev_30d", "DC_fire_day"):
    r1 = stats.spearmanr(ok[k], ok.ros_lb_m_per_h, nan_policy="omit"); r2 = stats.spearmanr(ok[k], np.log(ok.area_ha))
    P(f"  Спирмен {k:14s}: с ROS ρ={r1.statistic:+.2f} (p={r1.pvalue:.2f}); с log(площади) ρ={r2.statistic:+.2f} (p={r2.pvalue:.2f})")

# 2. обзор событий
fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
per = gpd.read_file(os.path.join(DATA, "events_spread.gpkg"), layer="perimeter")
for ax, site in zip(axs, ["AltynDala", "Korgalzhyn", "Naurzum", "Mirgorod"]):
    s = per[per.site == site]; e = ev[ev.site == site]
    s.plot(ax=ax, column="year", cmap="viridis", alpha=.55, edgecolor="k", linewidth=.3, legend=False)
    ax.scatter(e.ign_lon, e.ign_lat, c="red", s=12, zorder=3)
    L = 0.06 * max(np.ptp(s.total_bounds[[0, 2]]), 0.05)
    for r in e.itertuples():
        ax.annotate("", (r.ign_lon + L * np.sin(np.deg2rad(r.downwind_az_24h)), r.ign_lat + L * np.cos(np.deg2rad(r.downwind_az_24h))),
                    (r.ign_lon, r.ign_lat), arrowprops=dict(arrowstyle="->", color="blue", lw=1))
    ax.set_title(f"{site}: {len(e)} пожаров", fontsize=10); ax.tick_params(labelsize=7)
fig.suptitle("Контуры пожаров (цвет = год), точки возгорания (красные) и направление ветра за 24 ч (синие стрелки)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_events_overview.png"), dpi=150); plt.close()

# 3. пример куба
ex = ev[ev.gee_pre_ok].sort_values("area_ha").iloc[len(ev[ev.gee_pre_ok]) // 2].event
d = np.load(os.path.join(DATA, "cubes", ex + ".npz")); m = json.load(open(os.path.join(DATA, "cubes", ex + ".json"))); CH = m["channels"]
X = d["x"].astype(np.float32)
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
show = [("pre_NDVI", "NDVI до пожара", "YlGn"), ("pre_NDMI", "NDMI до пожара (влажность)", "BrBG"), ("elev", "Высота, м", "terrain"),
        ("slope_deg", "Уклон, °", "magma"), ("dist_ign_km", "Дистанция до возгорания, км", "cividis"), ("prior_burn_days", "Сутки с прошлой гари", "plasma")]
for ax, (k, t, cm) in zip(axs.ravel(), show):
    im = ax.imshow(X[CH.index(k)], cmap=cm); ax.set_title(t, fontsize=10); plt.colorbar(im, ax=ax, fraction=.046)
from matplotlib.colors import ListedColormap, BoundaryNorm
cols = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0"}
ks = sorted(cols); axs[1, 2].imshow(d["landcover"], cmap=ListedColormap([cols[k] for k in ks]), norm=BoundaryNorm([k - 5 for k in ks] + [ks[-1] + 5], len(ks)), interpolation="nearest")
axs[1, 2].set_title("ESA WorldCover 2021", fontsize=10)
axs[1, 3].imshow(np.where(d["ignore"], 0.5, d["label"]), cmap="Greys"); axs[1, 3].plot(m["ign_rc"][1], m["ign_rc"][0], "r*", ms=14)
e1 = ev[ev.event == ex].iloc[0]; L = 0.15 * min(m["H"], m["W"]); dn = np.deg2rad(e1.downwind_az_24h)
axs[1, 3].arrow(m["ign_rc"][1], m["ign_rc"][0], L * np.sin(dn), -L * np.cos(dn), color="b", width=L / 25)
axs[1, 3].set_title("Цель: контур (чёрн.), ignore (сер.), возгорание ★, ветер →", fontsize=10)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Пример куба 30 м: {ex}  ({m['H']}×{m['W']} пикс.)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_cube_example.png"), dpi=130); plt.close()

# 4. связь пиксельных признаков с целью (взвешенно по окну)
P("\nДоля сгоревших пикселей окна по классам WorldCover (взвешено w):")
g = tab.groupby("landcover").apply(lambda q: pd.Series(dict(burn_rate=np.average(q.label, weights=q.w), w=q.w.sum())), include_groups=False)
g["share_of_window"] = g.w / g.w.sum()
for k, r in g.iterrows():
    P(f"  {k:3d} {WC.get(k, '?'):16s} доля окна {r.share_of_window:.3f}  доля сгоревших {r.burn_rate:.3f}")
pb = tab.assign(cls=pd.cut(tab.prior_burn_days, [-1, 365, 730, 3650], labels=["<1 г", "1-2 г", ">2 г"]).cat.add_categories("не горел").fillna("не горел"))
P("Доля сгоревших по давности прошлой гари:")
for k, q in pb.groupby("cls", observed=True):
    P(f"  {k:9s} n={len(q):7d} доля сгоревших {np.average(q.label, weights=q.w):.3f}")
t2 = tab[tab.pre_NDVI.notna()]
for k in ("pre_NDVI", "pre_NDMI", "pre_NBR", "slope_deg", "along_wind_m"):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(q.label, q[k]) for _, q in (t2 if k.startswith("pre") else tab).dropna(subset=[k]).groupby("event") if q.label.nunique() == 2]
    P(f"  AUC по событиям, признак {k:13s}: медиана {np.median(aucs):.3f} (n={len(aucs)}; <0.5 = чем меньше признак, тем чаще горит)")
qa = tab[tab.qa_dNBR.notna()]
P(f"Контроль разметки: медиана dNBR внутри контура {qa[qa.label == 1].qa_dNBR.median():.3f}, снаружи {qa[qa.label == 0].qa_dNBR.median():.3f}")

# 5. сводка
P("\nСводка: событий", len(ev), "| с GEE до пожара:", int(ev.gee_pre_ok.sum()), "| с GEE после:", int(ev.gee_post_ok.sum()),
  "| кубов:", len(ci), "| пикселей в кубах:", int((ci.H * ci.W).sum()), "| строк в таблице:", len(tab))
P("Покрытие DEM мин", ci.cov_dem.min(), "| WorldCover мин", ci.cov_lc.min(), "| уклон p95 медиана", ci.slope_p95.median(), "°")
P("Окна с прошлыми гарями:", int((ci.prior_burn_frac > 0).sum()), "| с исключёнными одновременными пожарами:", int((ci.ignored_frac > 0).sum()))
open(os.path.join(DATA, "qa_summary.txt"), "w", encoding="utf8").write("\n".join(out))
