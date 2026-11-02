"""Задание 2 v3, шаг 6. Кэш входов сети на диске (np.memmap): входные карты, цели, источник метки.

Части: train - не более MAX_TRAIN срезов на пожар (случайно), без пожаров, которые есть среди экспертных пожаров заповедников
(gfa_fire_ids), чтобы тест на заповедниках был честным; val (2022) и test (2023–2024) - все срезы; reserves - экспертные пожары.
Выход: cache_v3/<part>_M.npy f16 [N,28,96,96], _Y.npy u8 [N,4,96,96], _L.npy u8 [N,96,96] (lsrc, 255 - не valid), _idx.csv.
Продолжение после сбоя: готовые файлы срезов перечислены в <part>_done.txt и пропускаются.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *
from v3_loader import DIRS

MAX_TRAIN = int(os.environ.get("MAX_TRAIN", 4))


def parts():
    si = load_index(); rs = load_index(kind="reserves")
    excl = {int(x) for s in rs.gfa_fire_ids.dropna() for x in str(s).split()}
    rng = np.random.default_rng(0); si["r"] = rng.random(len(si))
    tr = si[(si.split == "train") & ~si.fire_id.isin(excl)]
    tr = tr[tr.groupby("fire_id").r.rank(method="first") <= MAX_TRAIN]
    print("excluded reserve fires:", len(excl), "| train slices", len(tr), "fires", tr.fire_id.nunique(), flush=True)
    return {"train": (tr, "500m"), "val": (si[si.split == "val"], "500m"), "test": (si[si.split == "test"], "500m"),
            "reserves": (rs, "reserves")}


def build(part, si, kind):
    si = si.sort_values(["file", "slice"]).reset_index(drop=True); n = len(si)
    fM, fY, fL = (os.path.join(CACHE, f"{part}_{k}.npy") for k in "MYL")
    fdone = os.path.join(CACHE, f"{part}_done.txt"); fidx = os.path.join(CACHE, f"{part}_idx.csv")
    done = set(open(fdone).read().split()) if os.path.exists(fdone) and os.path.exists(fM) else set()
    if not done:
        si.to_csv(fidx, index=False)
        M = np.lib.format.open_memmap(fM, "w+", np.float16, (n, len(MAPS), 96, 96))
        Y = np.lib.format.open_memmap(fY, "w+", np.uint8, (n, 4, 96, 96)); L = np.lib.format.open_memmap(fL, "w+", np.uint8, (n, 96, 96))
        open(fdone, "w").close()
    else:
        M, Y, L = (np.load(f, mmap_mode="r+") for f in (fM, fY, fL))
    for f, g in si.groupby("file"):
        if f in done:
            continue
        d = np.load(os.path.join(KZ, DIRS[kind], f)); k = g.slice.values; pos = g.index.values
        X = d["X"][k]; V = d["valid"][k]
        for i in range(0, len(k), 500):
            M[pos[i:i + 500]] = nn_maps(X[i:i + 500], V[i:i + 500])
        Y[pos] = (d["Y"][k] > 0).astype(np.uint8)
        l = d["lsrc"][k].copy(); l[~V] = INVALID; L[pos] = l
        M.flush(); Y.flush(); L.flush()
        with open(fdone, "a") as fo:
            fo.write(f + "\n")
        print(f"  {part}: {f} {len(k)}", flush=True)
    open(os.path.join(CACHE, f"{part}_ok"), "w").write(str(n))
    print(part, "ready:", n, flush=True)


if __name__ == "__main__":
    for part, (si, kind) in parts().items():
        if os.path.exists(os.path.join(CACHE, f"{part}_ok")):
            print(part, "cached", flush=True); continue
        build(part, si, kind)
