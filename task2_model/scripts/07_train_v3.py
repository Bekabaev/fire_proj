"""Задание 2 v3, шаг 7. Обучение U-Net (FrontV3) на срезах v3.

RUN=nowx - без погоды (6 числовых признаков: сезон, срок, состояние пожара), как U-Net v1;
RUN=wx   - + погода через FiLM: окна 12/24/48/72 ч ERA5 («идеальный прогноз»), прошлые 7 сут ERA5, FWI GFWED (49 признаков);
           при аугментации векторы ветра поворачиваются вместе с картой.
Лосс: BCE (pos_weight 5) + Dice с весами ячеек по источнику метки (cell_weights). Кропы 80 × 80, повороты/отражения.
Выбор эпохи - по среднему F1 четырёх горизонтов на val 2022 (погода ERA5).
Чекпоинт после каждой эпохи: out_v3/<RUN>_last.pt (модель, оптимизатор, расписание, журнал); при перезапуске обучение
продолжается со следующей эпохи. Итог: out_v3/<RUN>_best.pt, out_v3/<RUN>_log.csv.
"""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

RUN = os.environ.get("RUN", "wx"); WX = RUN.startswith("wx")
EPOCHS = int(os.environ.get("EPOCHS", 24)); BS = 48; CROP = 80; LR = 2e-3
dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", 0)); PHYS = os.environ.get("PHYS") == "1"     # итерации: SEED - другая инициализация, PHYS - физический канал
AUG_GFS = os.environ.get("AUG_GFS") == "1"   # it6+: к погоде ERA5 добавляются ошибки прогноза GFS (шаг 28, val 2022), p = 0,5
EDGE = float(os.environ.get("EDGE", 1.0))    # it8: вес фона в 1 клетке от метки (< 1 - мягче штраф за промах на клетку у края)
torch.manual_seed(SEED); np.random.seed(SEED)


def load_part(part):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), parse_dates=["t_utc"], low_memory=False)
    M, Y, L = (np.load(os.path.join(CACHE, f"{part}_{k}{LAB if k != 'M' and os.path.exists(os.path.join(CACHE, f'{part}_{k}{LAB}.npy')) else ''}.npy"), mmap_mode="r") for k in "MYL")
    if PHYS:                                                    # время прихода по физике (ERA5, шаги 19/27) -> +4 канала
        return si, PhysM(M, np.load(os.path.join(OUT, f"phys_T_{part}_era5.npy"), mmap_mode="r")), Y, L
    return si, M, Y, L


class PhysM:
    """Карты сцены + 4 физических канала log(T/h), выборка по индексам как у memmap."""
    def __init__(s, M, T):
        s.M, s.T = M, T; s.shape = (M.shape[0], M.shape[1] + len(HZ)) + M.shape[2:]

    def __len__(s):
        return len(s.M)

    def __getitem__(s, b):
        return np.concatenate([np.asarray(s.M[b], np.float32), phys_maps(s.T[b])], 1)


def scalars(si, stats, src="era5"):
    z = base_scalars(si)
    return np.c_[z, weather_vec(si, stats, src)].astype(np.float32) if WX else z


def batch(M, Y, L, Z, b):
    return (torch.from_numpy(M[b].astype(np.float32)), torch.from_numpy(Y[b]), torch.from_numpy(L[b]), torch.from_numpy(Z[b]))


def evaluate(model, M, Y, L, Z, bs=96):
    """Гистограммы вероятностей по горизонтам на ячейках с весом > 0."""
    model.eval(); H = np.zeros((len(HZ), 2, NB))
    with torch.no_grad(), ThreadPoolExecutor(1) as ex:
        idx = [np.arange(i, min(i + bs, len(M))) for i in range(0, len(M), bs)]
        fut = ex.submit(batch, M, Y, L, Z, idx[0])
        for j in range(len(idx)):
            x, y, l, z = fut.result()
            if j + 1 < len(idx):
                fut = ex.submit(batch, M, Y, L, Z, idx[j + 1])
            x, y, l, z = x.to(dev), y.to(dev).bool(), l.to(dev), z.to(dev)
            p = torch.sigmoid(model(x, z)); w = cell_weights(l, y, dev)
            hist_update(H, p, y, w > 0)
    return H


if __name__ == "__main__":
    si_tr, Mtr, Ytr, Ltr = load_train(load_part); si_va, Mva, Yva, Lva = load_part("val")
    fst = os.path.join(OUT, "weather_stats.json")
    if not os.path.exists(fst):
        json.dump(weather_stats(si_tr), open(fst, "w"))
    stats = json.load(open(fst))
    Ztr = scalars(si_tr, stats); Zva = scalars(si_va, stats, os.environ.get("VAL_SRC", "era5"))   # it6+: VAL_SRC=gfsa - выбор эпохи по val с GFS
    uv = [(6 + iu, 6 + iv) for iu, iv in UV] if WX else []
    RES = torch.from_numpy(np.load(os.path.join(OUT, "gfs_resid.npy"))).to(dev) if AUG_GFS else None
    print("AUG_GFS", AUG_GFS, "EDGE", EDGE, flush=True)
    print(RUN, "| train", len(si_tr), "| val", len(si_va), "| maps", Mtr.shape[1], "| scalars", Ztr.shape[1], flush=True)
    model = FrontV3(Mtr.shape[1], Ztr.shape[1]).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    nb = int(np.ceil(len(si_tr) / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb, pct_start=0.1)
    flast = os.path.join(OUT, f"{RUN}_last.pt"); fbest = os.path.join(OUT, f"{RUN}_best.pt")
    ep0, log, best = 0, [], -1.0
    if os.path.exists(flast):
        ck = torch.load(flast, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
        ep0, log, best = ck["epoch"] + 1, ck["log"], ck["best"]
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"].cpu())
        print(f"resume from epoch {ep0}, best {best:.4f}", flush=True)
    for ep in range(ep0, EPOCHS):
        model.train(); t0 = time.time(); perm = np.random.permutation(len(si_tr)); Ls = []
        idx = [np.sort(perm[i:i + BS]) for i in range(0, len(perm), BS)]
        with ThreadPoolExecutor(2) as ex:
            futs = [ex.submit(batch, Mtr, Ytr, Ltr, Ztr, idx[j]) for j in range(min(3, len(idx)))]
            for j in range(len(idx)):
                x, y, l, z = futs[j].result(); futs[j] = None
                if j + 3 < len(idx):
                    futs.append(ex.submit(batch, Mtr, Ytr, Ltr, Ztr, idx[j + 3]))
                x, y, l, z = x.to(dev), y.to(dev).float(), l.to(dev), z.to(dev)
                if RES is not None:                             # ошибка прогноза: случайная строка R половине срезов батча
                    m = (torch.rand(len(z), 1, device=dev) < 0.5).float()
                    z = torch.cat([z[:, :6], z[:, 6:] + m * RES[torch.randint(len(RES), (len(z),), device=dev)]], 1)
                k = np.random.randint(4); fl = np.random.rand() < 0.5
                x, y, l = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3)), torch.rot90(l, k, (1, 2))
                if fl:
                    x, y, l = x.flip(3), y.flip(3), l.flip(2)
                z = rot_wind(z, k, fl, uv)
                r, c = np.random.randint(0, 96 - CROP + 1, 2)
                x, y, l = x[:, :, r:r + CROP, c:c + CROP], y[:, :, r:r + CROP, c:c + CROP], l[:, r:r + CROP, c:c + CROP]
                w = cell_weights(l, y.bool(), dev)
                if EDGE != 1.0:                                 # фон рядом с меткой (1 клетка) - вес EDGE
                    near = (torch.nn.functional.max_pool2d(y, 3, 1, 1) > 0) & (y == 0)
                    w = torch.where(near, w * EDGE, w)
                loss = loss_fn(model(x, z), y, w)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); Ls.append(loss.item())
                if j % 200 == 0:
                    print(f"  ep {ep} batch {j}/{len(idx)} loss {np.mean(Ls[-200:]):.4f} ({time.time() - t0:.0f} s)", flush=True)
        res = best_from_hist(evaluate(model, Mva, Yva, Lva, Zva)); f = float(np.mean([r[0] for r in res]))
        row = dict(epoch=ep, loss=float(np.mean(Ls)), val_F1_mean=f, sec=time.time() - t0,
                   **{f"val_F1_{h}h": r[0] for h, r in zip(HZ, res)}, **{f"th_{h}h": r[1] for h, r in zip(HZ, res)})
        log.append(row)
        print(f"epoch {ep:2d} loss {row['loss']:.4f} val F1 " + " ".join(f"{r[0]:.3f}" for r in res) + f" mean {f:.4f} ({row['sec']:.0f} s)", flush=True)
        if f > best:
            best = f; torch.save(dict(state=model.state_dict(), cin=Mtr.shape[1], ns=Ztr.shape[1], wx=WX, epoch=ep, phys=PHYS, seed=SEED, aug_gfs=AUG_GFS, edge=EDGE,
                                      th=[r[1] for r in res]), fbest)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), epoch=ep, log=log, best=best,
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), flast + ".tmp")
        os.replace(flast + ".tmp", flast)                      # атомарная запись: при отключении питания остаётся прошлая эпоха
        pd.DataFrame(log).to_csv(os.path.join(OUT, f"{RUN}_log.csv"), index=False)
    open(os.path.join(OUT, f"{RUN}_done"), "w").write(str(best))
    print("best val F1 mean", round(best, 4), flush=True)
