"""Задание 2 v3, шаг 9. Парное сравнение вариантов на тесте: разница F1 с 95 % интервалом (бутстрэп по пожарам, одни и те же выборки).
Выход: out_v3/paired_v3.csv."""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
cmp_ = pd.read_csv(os.path.join(OUT, "comparison_v3.csv"))
rows = []
for part, gcol in (("test", "fire_id"), ("reserves", "event")):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    Y = np.load(os.path.join(CACHE, f"{part}_Y.npy"), mmap_mode="r"); L = np.load(os.path.join(CACHE, f"{part}_L.npy"), mmap_mode="r")
    cnt = {}
    for v in ("nowx", "wx_era5", "wx_gfsa"):
        P = np.load(os.path.join(OUT, f"probs_{v}_{part}.npy"), mmap_mode="r")
        th = cmp_[(cmp_.variant == v) & (cmp_.part == part) & (cmp_.subset == "all")].sort_values("horizon_h").threshold.values
        if len(th) < 4:
            continue
        c = np.zeros((len(si), 4, 3))
        with torch.no_grad():
            for i in range(0, len(si), 256):
                y = torch.from_numpy(Y[i:i + 256]).to(dev).bool(); m = cell_weights(torch.from_numpy(L[i:i + 256]).to(dev), y, dev) > 0
                pr = torch.from_numpy(P[i:i + 256]).to(dev).float() / 255 >= torch.tensor(th, device=dev, dtype=torch.float32)[None, :, None, None]
                c[i:i + 256] = torch.stack([(pr & y & m).sum((2, 3)), (pr & ~y & m).sum((2, 3)), (~pr & y & m).sum((2, 3))], -1).cpu().numpy()
        cnt[v] = c
    mask = si.get("forecast_gfsa_ok", pd.Series(True, index=si.index)).fillna(False).values.astype(bool)
    g = pd.Series(si[gcol].values).astype("category").cat.codes.values; G = g.max() + 1
    rng = np.random.default_rng(0); B = [rng.integers(0, G, G) for _ in range(1000)]
    for a, b in (("wx_era5", "nowx"), ("wx_gfsa", "nowx"), ("wx_era5", "wx_gfsa")):
        if a not in cnt or b not in cnt:
            continue
        mk = mask if "wx_gfsa" in (a, b) else np.ones(len(si), bool)
        for h, H_ in enumerate(HZ):
            def agg(c):
                return np.stack([np.bincount(g[mk], c[mk, h, k], G) for k in range(3)], 1)
            A, Bc = agg(cnt[a]), agg(cnt[b])
            f = lambda S: 2 * S[0] / max(2 * S[0] + S[1] + S[2], 1)
            d0 = f(A.sum(0)) - f(Bc.sum(0)); ds = [f(A[k].sum(0)) - f(Bc[k].sum(0)) for k in B]
            lo, hi = np.percentile(ds, [2.5, 97.5])
            rows.append(dict(part=part, a=a, b=b, horizon_h=H_, n_slices=int(mk.sum()), n_groups=int(len(np.unique(g[mk]))),
                             dF1=round(d0, 4), lo=round(lo, 4), hi=round(hi, 4)))
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "paired_v3.csv"), index=False); print(t.to_string())
