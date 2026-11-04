"""Задание 2, шаг 16. Метки v4 (уточнены по Landsat / Sentinel-2, шаг 34 датасета) в кэше обучения без пересборки срезов.

Для каждого среза (fire_id, год, t, окно r0/c0) метки пересчитываются из data/ext_kz/arrival_img/arr_<год>.npz по правилам шага 22;
входные карты не меняются, поэтому «сгорело к t» берётся из burned_frac кэша. Ячейки, которые по новым меткам сгорели до t, а во входе -
нет, исключаются (L = 255). Проверка: для пожаров без снимков метки должны совпасть со старыми.
Выход: cache_v3/<часть>_Yi.npy, <часть>_Li.npy; out_v3/labels_img_stats.csv.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import CACHE, OUT, HZ, MAPS, INVALID, KZ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "task1_dataset", "scripts"))
from kz_common import GH, GW

COV = np.load(os.path.join(KZ, "lc.npz"))["covered"]; IB = MAPS.index("burned_frac"); N = 96
rows = []
for part in [p for p in os.environ.get("PARTS", "train,trainx,val,test").split(",") if os.path.exists(os.path.join(CACHE, f"{p}_Y.npy"))]:
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), low_memory=False)
    M = np.load(os.path.join(CACHE, f"{part}_M.npy"), mmap_mode="r"); Y = np.load(os.path.join(CACHE, f"{part}_Y.npy"), mmap_mode="r")
    L = np.load(os.path.join(CACHE, f"{part}_L.npy"), mmap_mode="r")
    Yi = np.lib.format.open_memmap(os.path.join(CACHE, f"{part}_Yi.npy"), "w+", np.uint8, Y.shape)
    Li = np.lib.format.open_memmap(os.path.join(CACHE, f"{part}_Li.npy"), "w+", np.uint8, L.shape)
    for y, g in si.groupby("year"):
        a = np.load(os.path.join(KZ, "arrival_img", f"arr_{y}.npz"))
        AR = np.full((GH, GW), np.nan, np.float32); FID = np.zeros((GH, GW), np.int32); SRC = np.zeros((GH, GW), np.uint8)
        AR[a["r"], a["c"]] = a["arr_h"]; FID[a["r"], a["c"]] = a["fire_id"]; SRC[a["r"], a["c"]] = a["src"]
        for i in g.index:
            r = si.loc[i]; w = (slice(int(r.r0), int(r.r0) + N), slice(int(r.c0), int(r.c0) + N)); t = r.t_h
            A = AR[w]; F = FID[w]; mine = F == r.fire_id; S0 = np.asarray(M[i, IB]) > 0
            other = (~mine) & (F > 0) & (A > t) & (A <= t + 72)
            V = COV[w] & ~S0 & ~other & ~(mine & (A <= t))
            Yi[i] = np.stack([mine & (A > t) & (A <= t + h) & V for h in HZ]).astype(np.uint8)
            Li[i] = np.where(V, np.where(mine, SRC[w], 0), INVALID).astype(np.uint8)
            yo = Y[i, 3] > 0; yn = Yi[i, 3] > 0
            rows.append(dict(part=part, i=i, fire_id=r.fire_id, new72_old=int(yo.sum()), new72_img=int(yn.sum()), same=bool((yo == yn).all()),
                             src6=int((yn & (Li[i] == 6)).sum()), src7=int((yn & (Li[i] == 7)).sum())))
    Yi.flush(); Li.flush(); del Yi, Li
    d = pd.DataFrame([r for r in rows if r["part"] == part])
    print(f"{part}: slices {len(d)}, labels unchanged {d.same.mean():.3f}, area72 old {d.new72_old.sum() * 0.25:.0f} km² -> img {d.new72_img.sum() * 0.25:.0f} km², "
          f"cells refined (src6) {d.src6.sum()}, added (src7) {d.src7.sum()}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "labels_img_stats.csv"), index=False)
