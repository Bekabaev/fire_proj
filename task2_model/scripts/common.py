"""Общее для задания 2: пути, разбиение, загрузка срезов, подготовка входов нейросети, метрики."""
import os, json
import numpy as np, pandas as pd

ROOT = r"C:\Users\Ionos Tech\Desktop\burned_area\project_till_nov"
T1 = os.path.join(ROOT, "task1_dataset", "data")
KZ = os.path.join(T1, "ext_kz")
OUT = os.path.join(ROOT, "task2_model", "out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(ROOT, "task2_model", "figures"); os.makedirs(FIG, exist_ok=True)
HZ = (24, 48, 72)

# разбиение как в FireSpread_Report_2: U-Net учится на ранних годах, поправка (GBM) и выбор порога - на следующих,
# сравнительный тест - на последних годах; пожары между частями не пересекаются
def split_of(year, fire_id):
    if year <= 2020:
        return "unet_train"
    if year <= 2022:
        return "calib_fit" if (int(fire_id) * 2654435761 % 2 ** 32) / 2 ** 32 < 0.7 else "calib_sel"   # по пожарам 70/30
    return "test"


def load_index():
    si = pd.read_csv(os.path.join(KZ, "slices_index.csv"), parse_dates=["t_utc"])
    si["part"] = [split_of(y, f) for y, f in zip(si.year, si.fire_id)]
    return si


def load_year(y):
    d = np.load(os.path.join(KZ, "slices", f"{y}.npz"))
    return d["X"], d["Y"], d["valid"]


CH = json.load(open(os.path.join(KZ, "slices_channels.json")))["channels"]
NN_IN = ["burned_frac", "af_count_24h", "af_frp_24h", "af_count_all", "prior_burn", "grass_frac", "crop_frac", "shrub_tree_frac",
         "wetland_frac", "nonburnable_frac", "elev", "slope_deg", "ndvi", "dist_ign_km"]          # 14 карт, как в отчёте


def nn_maps(X, valid):
    """X [S,15,H,W] float16 -> входы сети [S,16,H,W] float32 (14 карт + маска NDVI + valid)."""
    X = X.astype(np.float32); g = lambda k: X[:, CH.index(k)]
    out = []
    for k in NN_IN:
        a = g(k)
        if k in ("af_count_24h", "af_count_all"):
            a = np.log1p(a)
        elif k == "af_frp_24h":
            a = np.log1p(a) / 3
        elif k == "elev":
            a = (np.nan_to_num(a, nan=300.0) - 300.0) / 300.0
        elif k == "slope_deg":
            a = np.nan_to_num(a) / 3.0
        elif k == "ndvi":
            a = np.nan_to_num(a, nan=0.0) * 3
        elif k == "dist_ign_km":
            a = np.minimum(a, 60) / 20.0
        out.append(np.nan_to_num(a))
    out.append(np.isfinite(g("ndvi")).astype(np.float32)); out.append(valid.astype(np.float32))
    return np.stack(out, 1)


def scalars_unet(si):
    """Числовые признаки для FiLM сети (без погоды, как у исходного U-Net отчёта)."""
    d = si.doy.values
    return np.c_[np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365), np.minimum(si.k_day.values, 15) / 5,
                 np.log1p(si.state_km2.values), np.log1p(si.af_24h.values)].astype(np.float32)


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(F1=2 * tp / max(2 * tp + fp + fn, 1), precision=p, recall=r)


def f1_boot(tp, fp, fn, groups, n=1000, seed=0):
    """95 % интервал F1 бутстрэпом по пожарам. tp/fp/fn - массивы по срезам."""
    g = pd.Series(groups).astype("category").cat.codes.values; G = g.max() + 1
    T = np.bincount(g, tp, G); P = np.bincount(g, fp, G); Nn = np.bincount(g, fn, G)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        k = rng.integers(0, G, G)
        out.append(2 * T[k].sum() / max(2 * T[k].sum() + P[k].sum() + Nn[k].sum(), 1))
    return np.percentile(out, [2.5, 97.5])
