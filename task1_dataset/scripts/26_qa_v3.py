"""Шаг 26 (датасет v3). Сводка, нормировка каналов и рисунки контроля.

- data/ext_kz/slices_v3_stats.json: объём по частям, доля срезов с ростом, доля прироста внутри сцены, источники меток,
  покрытие погодой / станцией / прогнозом, крупные пожары (сцена 1 км), сравнение с v2;
- data/ext_kz/slices_v3_norm.json: среднее и СКО каналов по обучающей части (выборка ≤ 3000 срезов, только valid-ячейки), доля NaN;
- figures/fig10_v3_label_sources.png  - источники времени прихода по годам и ошибка времени против экспертных контуров;
- figures/fig11_v3_slice_example.png  - пример среза v3: состояние, термоточки 12 ч, цели 12/24/72 ч, источник метки;
- figures/fig12_v3_weather_vs_station.png - ERA5 (и прогнозы 2024) против станций, ветер за 24 ч после t.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *
from v3_loader import CH, load_chunk

fi = os.path.join(KZ, "slices_v3_index.csv")
si = pd.read_csv(fi if os.path.exists(fi) else os.path.join(KZ, "slices_v3_base.csv"), parse_dates=["t_utc"], low_memory=False)
s1 = pd.read_csv(os.path.join(KZ, "slices_1km_base.csv")) if os.path.exists(os.path.join(KZ, "slices_1km_base.csv")) else pd.DataFrame()
v2 = pd.read_csv(os.path.join(KZ, "slices_index.csv"))
arr = pd.read_csv(os.path.join(KZ, "arrival_summary.csv"))
HZ = (12, 24, 48, 72)

st = dict(slices=len(si), fires=int(si.fire_id.nunique()),
          by_split=si.groupby("split").agg(slices=("fire_id", "size"), fires=("fire_id", "nunique")).to_dict("index"),
          slices_1km=len(s1), fires_1km=int(s1.fire_id.nunique()) if len(s1) else 0,
          v2=dict(slices=len(v2), fires=int(v2.fire_id.nunique()), weather_share=round(float(v2.weather_ok.mean()), 3)))
for h in HZ:
    k = f"new_{h}h_km2"
    st[f"{h}h"] = dict(share_with_growth=round(float((si[k] > 0).mean()), 3),
                       in_scene_area_share=round(float(si[k].sum() / si[f"{k}_total"].sum()), 3),
                       targets_timed_by_af=round(float(np.nansum(si[f"new_{h}h_src_af"] * si[k]) / si[k].sum()), 3),
                       targets_day_only=round(float(np.nansum(si[f"new_{h}h_src_day"] * si[k]) / si[k].sum()), 3),
                       ext_share_of_targets=round(float(1 - si[f"{k}_mcd"].sum() / si[k].sum()), 3))
    if len(s1):
        b = si[si.big_fire]
        st[f"{h}h"]["big_fires_in_scene_500m"] = round(float(b[k].sum() / b[f"{k}_total"].sum()), 3)
        st[f"{h}h"]["big_fires_in_scene_1km"] = round(float(s1[k].sum() / s1[f"{k}_total"].sum()), 3)
for c in ("weather_ok", "station_ok", "forecast_ok"):
    if c in si:
        st[c] = round(float(si[c].mean()), 3)
if "weather_ok" in si:
    st["weather_by_year"] = si.groupby("year").weather_ok.mean().round(2).to_dict()
qa = os.path.join(KZ, "qa_labels_v3.txt")
if os.path.exists(qa):
    st["label_check_vs_expert"] = open(qa, encoding="utf8").read().splitlines()
qw = os.path.join(KZ, "qa_weather_v3.csv")
if os.path.exists(qw):
    st["weather_vs_stations"] = pd.read_csv(qw).to_dict("records")
fr = os.path.join(KZ, "slices_v3_reserves_index.csv")      # экспертные пожары заповедников (шаг 27)
if os.path.exists(fr):
    rs = pd.read_csv(fr, low_memory=False); f1 = os.path.join(KZ, "slices_1km_reserves_base.csv")
    st["reserves"] = dict(slices=len(rs), events=int(rs.event.nunique()), slices_1km=len(pd.read_csv(f1)) if os.path.exists(f1) else 0,
                          by_split_time=rs.groupby("split_time").agg(slices=("event", "size"), events=("event", "nunique")).to_dict("index"),
                          events_in_gfa_train=int(rs[rs.gfa_split.fillna("").str.contains("train")].event.nunique()),
                          events_without_gfa=int(rs[rs.gfa_fire_ids.isna()].event.nunique()),
                          weather_ok=round(float(rs.weather_ok.mean()), 3), station_ok=round(float(rs.station_ok.mean()), 3),
                          **{f"{h}h_in_scene_area_share": round(float(rs[f"new_{h}h_km2"].sum() / rs[f"new_{h}h_km2_total"].sum()), 3) for h in HZ},
                          **{f"{h}h_targets_timed_by_firms": round(float(np.nansum(rs[f"new_{h}h_src_af"] * rs[f"new_{h}h_km2"]) / rs[f"new_{h}h_km2"].sum()), 3) for h in HZ})
if "gfwed_FWI" in si:                                           # шаг 28
    st["fwi_ok"] = round(float(si.gfwed_FWI.notna().mean()), 3)
    st["fwi_median_by_growth_24h"] = {"growth" if k else "no_growth": round(float(v), 1) for k, v in si.groupby(si.new_24h_km2 > 0).gfwed_FWI.median().items()}
if "forecast_gfsa_ok" in si:                                    # шаг 29
    st["forecast_gfsa_by_year"] = si[si.year >= 2022].groupby("year").forecast_gfsa_ok.mean().round(3).to_dict()
qg = os.path.join(KZ, "qa_weather_gfsa.csv")
if os.path.exists(qg):
    st["weather_gfsa_vs_stations"] = pd.read_csv(qg).to_dict("records")
qp = os.path.join(DATA, "qa_perims.txt")                         # шаг 30
if os.path.exists(qp):
    st["perims_check"] = open(qp, encoding="utf8").read().splitlines()
    pq = pd.read_csv(os.path.join(DATA, "qa_perims.csv"))
    st["perims"] = dict(events=int(pq[pq.get("pixels", pd.Series(dtype=float)).notna()].event.nunique()), snapshots=int(pq.pixels.notna().sum()),
                        reliable=int(pq.get("reliable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()))
    for name in ("expert", "expert_img", "mcd64_day", "v3_gfa"):
        mid = pq[(pq.burned_share_img > 0.02) & (pq.burned_share_img < 0.98)]
        st["perims"][f"late_share_{name}"] = round(float(np.average(mid[f"late_share_{name}"].fillna(0), weights=mid.burned_share_img * mid.pixels)), 3)
json.dump(st, open(os.path.join(KZ, "slices_v3_stats.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(st, indent=1, ensure_ascii=False, default=float))

# --- нормировка по обучающей части
tr = si[si.split == "train"].sample(min(3000, (si.split == "train").sum()), random_state=0)
acc = {c: [] for c in CH}; nan = {c: [0, 0] for c in CH}
for f, g in tr.groupby("file"):
    X, Y, V, L = load_chunk(f, g.slice.values)
    for i, c in enumerate(CH):
        x = X[:, i][V]; nan[c][0] += int((~np.isfinite(x)).sum()); nan[c][1] += x.size
        x = x[np.isfinite(x)]
        acc[c].append(np.random.default_rng(0).choice(x, min(len(x), 20000), replace=False) if len(x) else x)
norm = {c: dict(mean=float(np.mean(np.concatenate(acc[c]))), std=float(np.std(np.concatenate(acc[c])) + 1e-6),
                nan_share=round(nan[c][0] / max(nan[c][1], 1), 4)) for c in CH}
json.dump(norm, open(os.path.join(KZ, "slices_v3_norm.json"), "w"), indent=1)
print("\nchannel nan share:", {c: v["nan_share"] for c, v in norm.items() if v["nan_share"] > 0})

# --- рис. 10: источники меток и ошибка времени
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
cols = ["src1", "src2", "src3", "src4", "src5"]; lab = ["VIIRS", "MODIS", "интерполяция", "только дата MCD64A1", "VIIRS вне MCD64A1"]
clr = ["#1b9e77", "#66a61e", "#7570b3", "#d95f02", "#e7298a"]
bot = np.zeros(len(arr))
for c, l, k in zip(cols, lab, clr):
    axs[0].bar(arr.year, arr[c], bottom=bot, color=k, label=l); bot += arr[c].values
axs[0].set_ylabel("доля ячеек пожаров"); axs[0].set_title("Источник времени прихода огня (ячейки 500 м)", fontsize=10)
axs[0].legend(fontsize=8, loc="lower left"); axs[0].set_ylim(0, 1)
if os.path.exists(qa):
    txt = [l for l in open(qa, encoding="utf8").read().splitlines() if l.startswith("  ")]
    nm = [l.split(":")[0].strip() for l in txt]; med = [float(l.split(":")[1].split("/")[0]) for l in txt]
    p75 = [float(l.split(":")[1].split("/")[1]) for l in txt]
    yy = np.arange(len(nm)); axs[1].barh(yy, p75, color="#cccccc", label="75-й процентиль"); axs[1].barh(yy, med, color="#444444", label="медиана")
    axs[1].set_yticks(yy); axs[1].set_yticklabels(nm, fontsize=7.5); axs[1].invert_yaxis()
    axs[1].set_xlabel("|ошибка времени| против экспертного, ч"); axs[1].legend(fontsize=8)
    axs[1].set_title("Проверка на экспертных пожарах заповедников", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig10_v3_label_sources.png"), dpi=150); plt.close()

# --- рис. 11: пример среза
q = si[(si.new_12h_km2 >= 3) & (si.new_72h_km2 >= 20) & (si.af_12h > 5) & (si.state_km2 > 10) & (si.split == "train")]
q = q.sort_values("new_72h_km2").iloc[len(q) // 2]
X, Y, V, L = load_chunk(q.file, [q.slice]); X, Y, V, L = X[0], Y[0], V[0], L[0]
fig, axs = plt.subplots(1, 5, figsize=(17, 3.8))
axs[0].imshow(X[CH.index("burned_frac")], cmap="Greys", vmin=0, vmax=1); axs[0].set_title(f"состояние на t = {q.t_utc:%Y-%m-%d %H} UTC", fontsize=9)
a = X[CH.index("af_count_12h")]; axs[1].imshow(np.where(a > 0, a, np.nan), cmap="hot_r"); axs[1].imshow(X[0], cmap="Greys", alpha=0.25)
axs[1].set_title("термоточки за 12 ч до t", fontsize=9)
grow = np.where(Y[0] > 0, 1, np.where(Y[1] > 0, 2, np.where(Y[3] > 0, 3, 0))).astype(float); grow[grow == 0] = np.nan
axs[2].imshow(X[0], cmap="Greys", vmin=0, vmax=2); axs[2].imshow(grow, cmap=ListedColormap(["#d7191c", "#fdae61", "#2b83ba"]), vmin=1, vmax=3)
axs[2].set_title("новое выгорание: 12 ч / 24 ч / 72 ч", fontsize=9)
ls = np.where(L > 0, L, np.nan).astype(float)
axs[3].imshow(ls, cmap=ListedColormap(clr), vmin=0.5, vmax=5.5); axs[3].set_title("источник метки (цвета как на рис. 10)", fontsize=9)
axs[4].imshow(X[CH.index("ndvi_anom")], cmap="BrBG", vmin=-0.15, vmax=0.15); axs[4].set_title("аномалия NDVI до пожара", fontsize=9)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f"Срез v3: пожар {q.fire_id}, площадь {q.fire_area_ext_km2:.0f} км², сцена 48 × 48 км", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig11_v3_slice_example.png"), dpi=150); plt.close()

# --- рис. 12: погода против станций
if "st_fut_ws_mean_24h" in si and "era5_ws_mean_24h" in si:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (src, c, ttl) in zip(axs, [("era5", "era5_ws_mean_24h", "ERA5 (все годы)"), ("gfs", "fc_gfs_ws_mean_24h", "GFS, 2024, заблаговр. 1–2 сут"),
                                       ("ifs", "fc_ifs_ws_mean_24h", "ECMWF IFS, 2024, заблаговр. 1–2 сут")]):
        if c not in si:
            ax.set_visible(False); continue
        m = si[[c, "st_fut_ws_mean_24h"]].dropna()
        if len(m) < 30:
            ax.set_visible(False); continue
        ax.hexbin(m["st_fut_ws_mean_24h"], m[c], gridsize=40, extent=(0, 12, 0, 12), mincnt=1, cmap="viridis", bins="log")
        ax.plot([0, 12], [0, 12], "w--", lw=1)
        ax.set_xlabel("станция, ветер за 24 ч после t, м/с"); ax.set_ylabel("модель, м/с")
        ax.set_title(f"{ttl}: r = {m[c].corr(m['st_fut_ws_mean_24h']):.2f}, n = {len(m)}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig12_v3_weather_vs_station.png"), dpi=150); plt.close()
# --- рис. 13: промежуточные контуры по снимкам против времени прихода (экспертное по термоточкам и уточнённое)
from scipy.ndimage import uniform_filter
ev13 = "AltynDala_2019_06_30"; fp = os.path.join(DATA, "perims", ev13 + ".npz")
if os.path.exists(fp):
    z = np.load(fp); lab13 = np.load(os.path.join(DATA, "cubes", ev13 + ".npz"))["label"].astype(bool)
    A0 = np.load(os.path.join(DATA, "arrival", ev13 + ".npz"))["arrival_h"]; A1 = np.load(os.path.join(DATA, "arrival_img", ev13 + ".npz"))["arrival_h"]
    rr, cc = np.nonzero(lab13); sl = (slice(max(rr.min() - 30, 0), rr.max() + 30), slice(max(cc.min() - 30, 0), cc.max() + 30))
    ks = [k for k in range(len(z["times_h"])) if (z["burned"][k][lab13] >= 0).mean() > 0.3][:3]
    fig, axs = plt.subplots(1, len(ks) + 1, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.25] * len(ks) + [1]))
    SNAME = {"LE07": "Landsat 7", "LC08": "Landsat 8", "LC09": "Landsat 9", "S2A_": "Sentinel-2A", "S2B_": "Sentinel-2B"}
    cm = ListedColormap(["#f0f0f0", "#1a9641", "#d7191c", "#2b83ba", "#bdbdbd"])
    for ax, k in zip(axs, ks):
        b = z["burned"][k]; t = z["times_h"][k]
        v = uniform_filter((b >= 0).astype(np.float32), 5); p_ = uniform_filter((b == 1).astype(np.float32), 5)
        img = (p_ >= 0.6 * np.maximum(v, 1e-6)) & (v > 0)
        st_ = np.nan_to_num(A0, nan=np.inf) <= t
        code = np.where(~lab13, 0, np.where(b < 0, 4, np.where(img & st_, 1, np.where(img & ~st_, 2, np.where(st_, 3, 0)))))
        ax.imshow(code[sl], cmap=cm, vmin=-0.5, vmax=4.5, interpolation="nearest")
        ax.set_title(f"{SNAME.get(str(z['sensors'][k]).split(':')[1], '')}: {t:.0f} ч после возгорания", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    import matplotlib.patches as mpatches
    axs[0].legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(["#1a9641", "#d7191c", "#2b83ba", "#bdbdbd"],
                  ["гарь на снимке и по метке", "гарь на снимке, метка ещё нет", "по метке сгорело, на снимке не видно", "облака / нет данных"])],
                  fontsize=7, loc="lower left")
    pq = pd.read_csv(os.path.join(DATA, "qa_perims.csv")); mid = pq[(pq.burned_share_img > 0.02) & (pq.burned_share_img < 0.98)]
    nm = [("mcd64_day", "v2: дата MCD64A1"), ("v3_gfa", "v3: GFA + VIIRS"), ("expert", "заповедники: термоточки"), ("expert_img", "заповедники: + снимки")]
    vals = [np.average(mid[f"late_share_{k}"].fillna(0), weights=mid.burned_share_img * mid.pixels) * 100 for k, _ in nm]
    axs[-1].barh(range(len(nm)), vals, color=["#d95f02", "#7570b3", "#1b9e77", "#1a9641"]); axs[-1].set_yticks(range(len(nm)))
    axs[-1].set_yticklabels([l for _, l in nm], fontsize=8); axs[-1].invert_yaxis(); axs[-1].set_xlabel("% видимой гари, которой\nещё нет в метке", fontsize=8)
    axs[-1].set_title(f"Запаздывание меток\n({len(mid)} снимков, {mid.event.nunique()} пожаров)", fontsize=9)
    plt.suptitle(f"Промежуточные контуры по снимкам и время прихода огня по термоточкам: {ev13}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig13_perims_vs_labels.png"), dpi=150); plt.close()
print("figures done")
