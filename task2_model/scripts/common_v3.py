"""Общее для обучения на датасете v3 (task1 шаги 18–30): пути, кэш входов, погода, модель, лосс, метрики.

Срезы v3: 23 канала, сетка 500 м 96 × 96, сроки 00/12 UTC, цели новое выгорание за 12/24/48/72 ч, источник метки lsrc.
Разбиение - как в датасете (split): train 2013–2021, val 2022, test 2023–2024.
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = r"C:\Users\Ionos Tech\Desktop\burned_area\project_till_nov"
sys.path.insert(0, os.path.join(ROOT, "task1_dataset", "scripts"))
from v3_loader import CH, TARGETS, UNC, KZ, load_index        # noqa: E402

T2 = os.path.join(ROOT, "task2_model")
CACHE = os.path.join(T2, "cache_v3"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(T2, "out_v3"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(T2, "figures"); os.makedirs(FIG, exist_ok=True)
HZ = (12, 24, 48, 72)
INVALID = 255                                                   # код lsrc для ячеек вне valid

# --- карты сцены -> входы сети (28 карт) ---
MISS = ("ndvi", "ndvi_anom", "ndvi_peak_prev", "elev", "slope_deg")
TRANS = {"burned_frac": lambda a: a, "burn_age_h": lambda a: a / 240, "af_count_12h": np.log1p,
         "af_frp_12h": lambda a: np.log1p(a) / 3, "af_count_24h": np.log1p, "af_count_all": np.log1p,
         "prior_burn": lambda a: a, "years_since_burn": lambda a: a / 13, "grass_frac": lambda a: a, "crop_frac": lambda a: a,
         "shrub_tree_frac": lambda a: a, "wetland_frac": lambda a: a, "nonburnable_frac": lambda a: a,
         "elev": lambda a: (a - 300) / 300, "slope_deg": lambda a: a / 3, "ndvi": lambda a: a * 3, "ndvi_anom": lambda a: a * 5,
         "ndvi_peak_prev": lambda a: a * 3, "dist_ign_km": lambda a: np.minimum(a, 60) / 20,
         "dist_settlement_km": lambda a: np.minimum(a, 100) / 30, "dist_road_km": lambda a: np.minimum(a, 40) / 10,
         "road_density": lambda a: a}                            # fbfm40 не берётся: категория, дублирует доли WorldCover
MAPS = list(TRANS) + [f"has_{k}" for k in MISS] + ["valid"]


def nn_maps(X, valid):
    """X [n,23,96,96] f16/f32 -> [n,28,96,96] f16."""
    X = X.astype(np.float32); out = []
    for k, f in TRANS.items():
        out.append(np.nan_to_num(f(X[:, CH.index(k)]), nan=0.0, posinf=0.0, neginf=0.0))
    for k in MISS:
        out.append(np.isfinite(X[:, CH.index(k)]).astype(np.float32))
    out.append(valid.astype(np.float32))
    return np.stack(out, 1).astype(np.float16)


# --- числовые признаки ---
FUT = ["T_max", "RH_min", "ws_mean", "gust_max", "wind_u", "wind_v", "P_sum"]
PREV = ["era5_P_sum_prev7d", "era5_VPD_mean_prev7d", "era5_soilm_t", "era5_RH_min_prev24h", "era5_ws_mean_prev24h"]
FWI = ["gfwed_FFMC", "gfwed_DMC", "gfwed_DC", "gfwed_ISI", "gfwed_BUI", "gfwed_FWI", "gfwed_FWI_max3d"]
UV = [(i * len(FUT) + FUT.index("wind_u"), i * len(FUT) + FUT.index("wind_v")) for i in range(len(HZ))]   # позиции ветра в векторе


def base_scalars(si):
    """Без погоды: сезон, срок, номер полусуток пожара, площадь и термоточки на t (6)."""
    d = pd.to_datetime(si.t_utc).dt.dayofyear.values
    return np.c_[np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365), (si.hour_utc.values == 12),
                 np.minimum(si.k_12h.values, 30) / 10, np.log1p(si.state_km2.values), np.log1p(si.af_12h.values)].astype(np.float32)


def raw_weather(si, src="era5"):
    """Сырые погодные признаки: [будущие окна 4 × 7 (src: era5 - «идеальный прогноз», gfsa - прогноз GFS)] + прошлое ERA5 + FWI."""
    pre = {"era5": "era5_", "gfsa": "fc_gfsa_"}[src]
    fut = np.stack([si.get(f"{pre}{v}_{h}h", pd.Series(np.nan, index=si.index)).values.astype(np.float32) for h in HZ for v in FUT], 1)
    prev = np.stack([si.get(c, pd.Series(np.nan, index=si.index)).values.astype(np.float32) for c in PREV], 1)
    fwi = np.stack([si.get(c, pd.Series(np.nan, index=si.index)).values.astype(np.float32) for c in FWI], 1)
    return fut, prev, fwi


def _tf(fut, prev, fwi):
    fut = fut.copy(); prev = prev.copy(); fwi = fwi.copy()
    p = [i for i, v in enumerate(FUT * len(HZ)) if v == "P_sum"]
    fut[:, p] = np.log1p(np.maximum(fut[:, p], 0)); prev[:, 0] = np.log1p(np.maximum(prev[:, 0], 0))
    fwi[:, [1, 2, 4]] = np.log1p(np.maximum(fwi[:, [1, 2, 4]], 0))
    return fut, prev, fwi


def weather_stats(si_train):
    fut, prev, fwi = _tf(*raw_weather(si_train, "era5"))
    A = np.c_[fut, prev, fwi]; mu = np.nanmean(A, 0); sd = np.nanstd(A, 0) + 1e-6
    for iu, iv in UV:                                           # ветер: без центрирования (поворот при аугментации), общий масштаб
        mu[iu] = mu[iv] = 0; sd[iu] = sd[iv] = np.sqrt(np.nanmean(A[:, iu] ** 2 + A[:, iv] ** 2) / 2)
    return dict(mu=mu.tolist(), sd=sd.tolist())


def weather_vec(si, stats, src="era5"):
    """Нормированный вектор погоды (40) + 3 флага пропуска (будущее, прошлое, FWI). Пропуски -> 0."""
    fut, prev, fwi = _tf(*raw_weather(si, src))
    A = (np.c_[fut, prev, fwi] - np.array(stats["mu"], np.float32)) / np.array(stats["sd"], np.float32)
    fl = np.c_[np.isnan(fut).any(1), np.isnan(prev).any(1), np.isnan(fwi).any(1)].astype(np.float32)
    return np.c_[np.nan_to_num(A), fl].astype(np.float32)


# --- модель ---
class FiLMBlock(nn.Module):
    def __init__(s, cin, cout, nz):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, padding=1); s.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.n1 = nn.GroupNorm(4, cout); s.n2 = nn.GroupNorm(4, cout); s.film = nn.Linear(nz, 2 * cout)

    def forward(s, x, z):
        x = F.gelu(s.n1(s.c1(x)))
        g, b = s.film(z).chunk(2, 1)
        return F.gelu(s.n2(s.c2(x)) * (1 + g[:, :, None, None]) + b[:, :, None, None])


class FrontV3(nn.Module):
    """U-Net 4 уровня с FiLM по числовым признакам (сезон, состояние, погода); 4 накопленных выхода 12 ⊂ 24 ⊂ 48 ⊂ 72 ч."""
    def __init__(s, cin, ns, ch=(16, 32, 64, 128), nz=64):
        super().__init__()
        s.zm = nn.Sequential(nn.Linear(ns, 64), nn.GELU(), nn.Linear(64, nz))
        s.enc = nn.ModuleList(); c = cin
        for k in ch:
            s.enc.append(FiLMBlock(c, k, nz)); c = k
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for k in reversed(ch[:-1]):
            s.up.append(nn.ConvTranspose2d(c, k, 2, stride=2)); s.dec.append(FiLMBlock(2 * k, k, nz)); c = k
        s.head = nn.Conv2d(c, len(HZ), 1)

    def forward(s, x, z):
        z = s.zm(z); skips = []
        for i, b in enumerate(s.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = b(x, z); skips.append(x)
        for u, d, sk in zip(s.up, s.dec, reversed(skips[:-1])):
            x = d(torch.cat([u(x), sk], 1), z)
        o = s.head(x); outs = [o[:, 0]]
        for k in range(1, len(HZ)):
            outs.append(outs[-1] + F.softplus(o[:, k]))
        return torch.stack(outs, 1)


# --- веса ячеек (как v3_loader.label_weights, на GPU) ---
def cell_weights(L, Yb, dev):
    """L [B,H,W] uint8 (lsrc, 255 - не valid), Yb [B,4,H,W] bool -> W [B,4,H,W]: фон 1, цель 1/(1+неопр./12 ч);
    на 12 ч цели только с датой MCD64A1 (src 4) не учитываются - ошибка даты больше горизонта."""
    wt = torch.ones(256, device=dev)
    for k, u in UNC.items():
        wt[k] = 1 / (1 + u / 12)
    wt12 = wt.clone(); wt12[4] = 0
    Ll = L.long(); valid = (L != INVALID).float()
    W = []
    for h in range(len(HZ)):
        w = torch.where(Yb[:, h], (wt12 if h == 0 else wt)[Ll], torch.ones_like(valid))
        W.append(w * valid)
    return torch.stack(W, 1)


def loss_fn(logit, y, w):
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(5.0, device=y.device), reduction="none")
    bce = (bce * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y * w).sum((0, 2, 3)) + 1) / ((p * w).sum((0, 2, 3)) + (y * w).sum((0, 2, 3)) + 1)
    return bce + dice.mean()


def rot_wind(z, k, flip, uv_pos):
    """Поворот векторов ветра вместе с картой: rot90 на k (против часовой, север вверху) и отражение восток↔запад."""
    if not uv_pos:
        return z
    z = z.clone(); c, s = [1, 0, -1, 0][k], [0, 1, 0, -1][k]
    for iu, iv in uv_pos:
        u, v = z[:, iu].clone(), z[:, iv].clone()
        z[:, iu] = c * u - s * v; z[:, iv] = s * u + c * v
        if flip:
            z[:, iu] = -z[:, iu]
    return z


# --- метрики ---
NB = 200


def hist_update(H, p, y, m):
    """Гистограммы вероятностей по горизонтам для подбора порога: H [4, 2, NB] (отрицательные / положительные)."""
    b = (p * (NB - 1)).round().long().clamp(0, NB - 1)
    for h in range(p.shape[1]):
        mh = m[:, h]
        H[h, 0] += torch.bincount(b[:, h][mh & ~y[:, h]], minlength=NB).cpu().numpy()
        H[h, 1] += torch.bincount(b[:, h][mh & y[:, h]], minlength=NB).cpu().numpy()


def best_from_hist(H):
    """Лучший F1 и порог по гистограммам (порог = нижняя граница бина)."""
    res = []
    for h in range(H.shape[0]):
        neg = H[h, 0][::-1].cumsum()[::-1]; pos = H[h, 1][::-1].cumsum()[::-1]
        tp = pos; fp = neg; fn = H[h, 1].sum() - pos
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1); i = int(np.argmax(f1[1:]) + 1)
        res.append((float(f1[i]), (i - 0.5) / (NB - 1)))
    return res


def f1_boot(tp, fp, fn, groups, n=1000, seed=0):
    g = pd.Series(groups).astype("category").cat.codes.values; G = g.max() + 1
    T = np.bincount(g, tp, G); P = np.bincount(g, fp, G); Nn = np.bincount(g, fn, G)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        k = rng.integers(0, G, G)
        out.append(2 * T[k].sum() / max(2 * T[k].sum() + P[k].sum() + Nn[k].sum(), 1))
    return np.percentile(out, [2.5, 97.5])


class CatArr:
    """Склейка нескольких memmap по первой оси для выборки по отсортированным индексам (обучение на train + trainx)."""
    def __init__(s, arrs):
        s.arrs = arrs; s.off = np.cumsum([0] + [len(a) for a in arrs]); s.shape = (int(s.off[-1]),) + arrs[0].shape[1:]

    def __len__(s):
        return int(s.off[-1])

    def __getitem__(s, b):
        b = np.asarray(b); out = []
        for a, o0, o1 in zip(s.arrs, s.off[:-1], s.off[1:]):
            k = b[(b >= o0) & (b < o1)] - o0
            if len(k):
                out.append(a[k])
        return np.concatenate(out)


def load_train(load_part):
    """train (+ части из EXTRA="trainx,..."), индексы склеены."""
    parts = ["train"] + [p for p in os.environ.get("EXTRA", "").split(",") if p]
    ps = [load_part(p) for p in parts]
    if len(ps) == 1:
        return ps[0]
    si = pd.concat([p[0] for p in ps], ignore_index=True)
    return (si, *(CatArr([p[k] for p in ps]) for k in (1, 2, 3)))
