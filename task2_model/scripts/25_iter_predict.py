"""Задание 2, итерации улучшения (ITERATIONS.md), шаг 25. Прогноз U-Net с усреднением по 8 поворотам/отражениям (TTA).

Карта поворачивается на k·90° и отражается восток↔запад, векторы ветра в числовых признаках - вместе с ней (как при обучении,
common_v3.rot_wind); прогноз возвращается в исходную ориентацию, вероятности усредняются. Без TTA (TTA=0) - один проход.
RUNS=wx,wx2 - модели (out_v3/<run>_best.pt). Части: val/test с GFS (как на сайте), reserves с ERA5 (GFS там нет).
Выход: out_v3/probs_<run>[_tta]_<src>_<part>.npy (u8, вероятность × 255). Существующие файлы не пересчитываются (возобновление).
"""
import os, sys, json, importlib.util
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

_s = importlib.util.spec_from_file_location("ev12", os.path.join(os.path.dirname(__file__), "12_eval_all.py"))
ev12 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev12)
dev = ev12.dev
TTA = os.environ.get("TTA", "1") == "1"
JOBS = [("val", "gfsa"), ("test", "gfsa"), ("reserves", "era5")] + ([("val", "era5")] if os.environ.get("VAL_ERA5") else [])


def predict_tta(model, M, Z, uv, T=None, bs=32):
    P = np.zeros((len(M), len(HZ), 96, 96), np.uint8)
    tf = [(k, fl) for k in range(4) for fl in (False, True)] if TTA else [(0, False)]
    with torch.no_grad():
        for i in range(0, len(M), bs):
            xa = M[i:i + bs].astype(np.float32)
            if T is not None:                                   # модель с физическим каналом (итерация 4)
                xa = np.concatenate([xa, phys_maps(T[i:i + bs])], 1)
            x0 = torch.from_numpy(xa).to(dev); z0 = torch.from_numpy(Z[i:i + bs]).to(dev); acc = 0
            for k, fl in tf:
                x = torch.rot90(x0, k, (2, 3))
                if fl:
                    x = x.flip(3)
                p = torch.sigmoid(model(x, rot_wind(z0, k, fl, uv)))
                if fl:
                    p = p.flip(3)
                acc = acc + torch.rot90(p, -k, (2, 3))
            P[i:i + bs] = (acc / len(tf) * 255).round().byte().cpu().numpy()
    return P


if __name__ == "__main__":
    stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
    for run in os.environ.get("RUNS", "wx").split(","):
        model, ck = ev12.load_model(run); wx = ck.get("wx")
        uv = [(6 + iu, 6 + iv) for iu, iv in UV] if wx else []
        for part, src in JOBS:
            f = os.path.join(OUT, f"probs_{run}{'_tta' if TTA else ''}_{src}_{part}.npy")
            if os.path.exists(f):
                print("exists", f, flush=True); continue
            si, M, Y, L = ev12.load_part(part)
            Z = (np.c_[base_scalars(si), weather_vec(si, stats, src)] if wx else base_scalars(si)).astype(np.float32)
            T = np.load(os.path.join(OUT, f"phys_T_{part}_{src}.npy"), mmap_mode="r") if ck.get("phys") else None
            np.save(f + ".tmp.npy", predict_tta(model, M, Z, uv, T)); os.replace(f + ".tmp.npy", f)
            print("saved", f, flush=True)
        del model; torch.cuda.empty_cache()
    print("done", flush=True)
