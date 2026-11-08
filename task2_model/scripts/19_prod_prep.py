"""Задание 2, шаг 19 (прод). Входы для гибрида «U-Net + физика»: вероятности U-Net wx и время прихода огня по физике на val 2022.

Тест и заповедники уже посчитаны (шаги 12 и 14): out_v3/probs_wx_<src>_<часть>.npy, out_v3/phys_T_<часть>_<src>.npy.
Здесь - то же для val (по нему подбираются веса гибрида в шаге 20). k_ros = 0,05, как в шаге 14
(в гибриде используется время прихода относительно горизонта, масштаб k компенсируется весом).
Выход: out_v3/probs_wx_{era5,gfsa}_val.npy, out_v3/phys_T_val_{era5,gfsa}.npy.
"""
import os, sys, json, importlib.util
import numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *


def _mod(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f)); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m                                       # для pickle функций в пуле процессов (Windows spawn)
    s.loader.exec_module(m); return m


ev12 = _mod("ev12", "12_eval_all.py"); ph14 = _mod("ph14", "14_phys_baseline.py")
K_ROS = 0.05

if __name__ == "__main__":
    stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
    si, M, Y, L = ev12.load_part("val")
    model, ck = ev12.load_model("wx")
    for src in ("era5", "gfsa"):
        f = os.path.join(OUT, f"probs_wx_{src}_val.npy")
        if not os.path.exists(f):
            Z = np.c_[base_scalars(si), weather_vec(si, stats, src)].astype(np.float32)
            np.save(f, ev12.predict(model, M, Z)); print("saved", f, flush=True)
    del model; torch.cuda.empty_cache()
    # REDO=test_gfsa,reserves_era5 - пересчёт файлов шага 14 от 02.11: их перезаписал параллельный запуск старой версии
    # со знаком ветра «откуда» без разворота (огонь шёл против ветра; проверка - 20_prod_hybrid.wind_cos)
    redo = [r.split("_") for r in os.environ.get("REDO", "").split(",") if r]
    with Pool(int(os.environ.get("PROCS", 10))) as pool:
        for part, src in dict.fromkeys([("val", "era5"), ("val", "gfsa")] + [tuple(r) for r in redo]):
            f = os.path.join(OUT, f"phys_T_{part}_{src}.npy")
            if not os.path.exists(f) or [part, src] in redo:
                sp_, Mp = (si, M) if part == "val" else ev12.load_part(part)[:2]
                T = ph14.simulate(Mp, sp_, np.arange(len(sp_)), src, K_ROS, pool); np.save(f, T); print("saved", f, flush=True)
    print("done", flush=True)
