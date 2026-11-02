"""Задание 2 v3, шаг 8. Оценка на тесте 2023–2024 и на экспертных пожарах заповедников.

Варианты: ring - расширение текущей гари на r ячеек (r подбирается на val), nowx - U-Net без погоды,
wx_era5 - U-Net с погодой ERA5 («идеальный прогноз»), wx_gfsa - тот же U-Net с настоящим прогнозом GFS (архив AWS, прогон t − 12 ч).
Порог каждого варианта и горизонта - по F1 на val 2022 (для wx_gfsa - тоже с GFS); на заповедниках пороги те же (без подгонки).
Ячейки с весом 0 (вне valid; на 12 ч - цели только с датой MCD64A1) не оцениваются.
Выход: out_v3/comparison_v3.csv/.json (F1, precision, recall, 95 % интервал бутстрэпом по пожарам),
out_v3/probs_<вариант>_<часть>.npy (u8, вероятность × 255) для задания 3.
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common_v3 import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
RADII = (1, 2, 3, 5, 8)


def load_part(part):
    si = pd.read_csv(os.path.join(CACHE, f"{part}_idx.csv"), parse_dates=["t_utc"], low_memory=False)
    return si, *(np.load(os.path.join(CACHE, f"{part}_{k}.npy"), mmap_mode="r") for k in "MYL")


def load_model(run):
    ck = torch.load(os.path.join(OUT, f"{run}_best.pt"), map_location=dev, weights_only=False)
    m = FrontV3(ck["cin"], ck["ns"]).to(dev); m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def run_variant(name, si, M, Y, L, model=None, Z=None, th=None, save=None, bs=96):
    """th=None -> гистограммы для подбора порога (или радиуса); иначе - счётчики tp/fp/fn по срезам [S,4,3]."""
    n = len(si); H = np.zeros((len(HZ), 2, NB)) if model is not None else np.zeros((len(RADII), len(HZ), 3))
    cnt = np.zeros((n, len(HZ), 3)); P = np.zeros((n, len(HZ), 96, 96), np.uint8) if save else None
    with torch.no_grad():
        for i in range(0, n, bs):
            x = torch.from_numpy(M[i:i + bs].astype(np.float32)).to(dev); y = torch.from_numpy(Y[i:i + bs]).to(dev).bool()
            l = torch.from_numpy(L[i:i + bs]).to(dev); m = cell_weights(l, y, dev) > 0
            if model is None:                                   # расширение текущей гари
                b = (x[:, MAPS.index("burned_frac")] > 0).float()[:, None]
                for ri, r in enumerate(RADII):
                    pr = (F.max_pool2d(b, 2 * r + 1, 1, r) > 0) & (b == 0)
                    pr = pr.expand(-1, len(HZ), -1, -1)
                    tp = (pr & y & m).sum((2, 3)); fp = (pr & ~y & m).sum((2, 3)); fn = (~pr & y & m).sum((2, 3))
                    if th is None:
                        H[ri] += torch.stack([tp, fp, fn], -1).sum(0).cpu().numpy()
                    else:
                        sel = torch.tensor([th[h] == r for h in range(len(HZ))], device=dev)
                        cnt[i:i + bs] += (torch.stack([tp, fp, fn], -1) * sel[None, :, None]).cpu().numpy()
                continue
            p = torch.sigmoid(model(x, torch.from_numpy(Z[i:i + bs]).to(dev)))
            if save:
                P[i:i + bs] = (p * 255).round().byte().cpu().numpy()
            if th is None:
                hist_update(H, p, y, m)
            else:
                pr = p >= torch.tensor(th, device=dev)[None, :, None, None]
                cnt[i:i + bs] = torch.stack([(pr & y & m).sum((2, 3)), (pr & ~y & m).sum((2, 3)), (~pr & y & m).sum((2, 3))], -1).cpu().numpy()
    if save:
        np.save(save, P)
    return H if th is None else cnt


def pick(H, model):
    if model is not None:
        return [r[1] for r in best_from_hist(H)]
    f1 = 2 * H[..., 0] / np.maximum(2 * H[..., 0] + H[..., 1] + H[..., 2], 1)      # [radius, horizon]
    return [RADII[int(np.argmax(f1[:, h]))] for h in range(len(HZ))]


def summarize(cnt, groups, mask=None):
    if mask is not None:
        cnt, groups = cnt[mask], np.asarray(groups)[mask]
    out = {}
    for h, H_ in enumerate(HZ):
        tp, fp, fn = cnt[:, h, 0], cnt[:, h, 1], cnt[:, h, 2]
        T, P_, N_ = tp.sum(), fp.sum(), fn.sum(); lo, hi = f1_boot(tp, fp, fn, groups)
        out[H_] = dict(F1=2 * T / max(2 * T + P_ + N_, 1), precision=T / max(T + P_, 1), recall=T / max(T + N_, 1), lo=lo, hi=hi)
    return out


if __name__ == "__main__":
    stats = json.load(open(os.path.join(OUT, "weather_stats.json")))
    va = load_part("val"); te = load_part("test"); rs = load_part("reserves")
    models = {r: load_model(r) for r in ("nowx", "wx") if os.path.exists(os.path.join(OUT, f"{r}_best.pt"))}
    variants = [("ring", None, None)]
    if "nowx" in models:
        variants.append(("nowx", "nowx", None))
    if "wx" in models:
        variants += [("wx_era5", "wx", "era5"), ("wx_gfsa", "wx", "gfsa")]

    def Zof(si, run, src):
        z = base_scalars(si)
        return np.c_[z, weather_vec(si, stats, src)].astype(np.float32) if models[run][1].get("wx") else z

    rows, res = [], {}
    for name, run, src in variants:
        model = models[run][0] if run else None
        th = pick(run_variant(name, *va, model=model, Z=Zof(va[0], run, src) if run else None), model)
        print(name, "threshold/radius by horizon:", th, flush=True)
        for part, (si, M, Y, L) in (("test", te), ("reserves", rs)):
            Z = Zof(si, run, src) if run else None
            save = os.path.join(OUT, f"probs_{name}_{part}.npy") if run else None
            cnt = run_variant(name, si, M, Y, L, model=model, Z=Z, th=th, save=save)
            groups = si.fire_id.values if part == "test" else si.event.values
            masks = {"all": None}
            if src == "gfsa":                                   # прогноз есть не у всех срезов заповедников (только 2022)
                masks = {"all": si.get("forecast_gfsa_ok", pd.Series(False, index=si.index)).fillna(False).values.astype(bool)}
                if not masks["all"].any():
                    continue
            if part == "test":
                for y in (2023, 2024):
                    masks[str(y)] = (si.year.values == y) & (masks["all"] if masks["all"] is not None else True)
            for sub, mk in masks.items():
                s = summarize(cnt, groups, mk)
                for H_, v in s.items():
                    rows.append(dict(variant=name, part=part, subset=sub, horizon_h=H_, n_slices=int(len(si) if mk is None else mk.sum()),
                                     threshold=th[HZ.index(H_)], **{k: round(float(x), 4) for k, x in v.items()}))
                print(f"  {part:8s} {sub:4s} " + " ".join(f"{H_}h {v['F1']:.3f} [{v['lo']:.3f}-{v['hi']:.3f}]" for H_, v in s.items()), flush=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "comparison_v3.csv"), index=False)
    t.to_json(os.path.join(OUT, "comparison_v3.json"), orient="records", force_ascii=False, indent=1)
    print(t[t.subset == "all"].pivot_table(index=["part", "variant"], columns="horizon_h", values="F1").round(3).to_string())
