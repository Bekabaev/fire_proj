"""Задание 2, итерации (ITERATIONS.md), шаг 27. Время прихода огня по физике (шаг 14, k_ros 0,05, ERA5) для обучающей выборки -
вход «физический канал» U-Net в итерации 4 (RUN=wxphys, шаг 07 с PHYS=1).

Считается кусками по 1500 срезов в out_v3/phys_T_train_era5.npy (memmap f16); выполненные куски отмечаются в
out_v3/phys_T_train_era5.done (возобновление после отключения питания). PART=trainx - то же для доп. части.
"""
import os, sys, time, importlib.util
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import OUT

_s = importlib.util.spec_from_file_location("ph14", os.path.join(os.path.dirname(__file__), "14_phys_baseline.py"))
ph14 = importlib.util.module_from_spec(_s); sys.modules["ph14"] = ph14; _s.loader.exec_module(ph14)
PART = os.environ.get("PART", "train"); SRC = os.environ.get("SRC", "era5"); CH_ = 1500

if __name__ == "__main__":
    si, M, Y, L = ph14.load(PART); n = len(si)
    f = os.path.join(OUT, f"phys_T_{PART}_{SRC}.npy"); fd = f.replace(".npy", ".done")
    T = np.lib.format.open_memmap(f, mode="r+" if os.path.exists(f) else "w+", dtype=np.float16, shape=(n, 96, 96))
    done = set(int(x) for x in open(fd).read().split()) if os.path.exists(fd) else set()
    t0 = time.time()
    with Pool(int(os.environ.get("PROCS", 9))) as pool:
        for k, i0 in enumerate(range(0, n, CH_)):
            if k in done:
                continue
            idx = np.arange(i0, min(i0 + CH_, n))
            T[idx] = ph14.simulate(M, si, idx, SRC, 0.05, pool); T.flush()
            open(fd, "a").write(f"{k}\n")
            print(f"chunk {k} ({idx[-1] + 1}/{n}) {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)
