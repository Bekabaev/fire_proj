"""Задание 2, шаг 1. Компактный U-Net (аналог Compact Front из FireSpread_Report_2), без погоды.

Вход: 14 карт сцены (+ маска NDVI, valid), 5 числовых признаков через FiLM (сезон, сутки от начала, площадь, термоточки).
Четыре уровня 12/24/48/96 каналов, связи между уровнями, FiLM на каждом уровне. Выход - накопленные по времени логиты
нового выгорания за 24/48/72 ч (z48 ≥ z24, z72 ≥ z48 -> вложенность областей).
Обучение: годы ≤ 2020 (unet_train), случайные кропы 80×80, повороты/отражения; выбор эпохи по F1 на calib_sel.
Выход: out/unet.pt, out/unet_probs_<год>.npy (uint8, вероятность ×255, [S,3,96,96]), out/unet_train_log.csv
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 30)); BS = 64; CROP = 80


class FiLMBlock(nn.Module):
    def __init__(s, cin, cout, ns):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, padding=1); s.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.n1 = nn.GroupNorm(4, cout); s.n2 = nn.GroupNorm(4, cout); s.film = nn.Linear(ns, 2 * cout)

    def forward(s, x, z):
        x = F.gelu(s.n1(s.c1(x)))
        g, b = s.film(z).chunk(2, 1)
        x = s.n2(s.c2(x)) * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return F.gelu(x)


class CompactFront(nn.Module):
    def __init__(s, cin=16, ns=5, ch=(12, 24, 48, 96)):
        super().__init__()
        s.zm = nn.Sequential(nn.Linear(ns, 32), nn.GELU(), nn.Linear(32, 32))
        s.enc = nn.ModuleList(); c = cin
        for k in ch:
            s.enc.append(FiLMBlock(c, k, 32)); c = k
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for k in reversed(ch[:-1]):
            s.up.append(nn.ConvTranspose2d(c, k, 2, stride=2)); s.dec.append(FiLMBlock(2 * k, k, 32)); c = k
        s.head = nn.Conv2d(c, 3, 1)

    def forward(s, x, z):
        z = s.zm(z); skips = []
        for i, b in enumerate(s.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = b(x, z); skips.append(x)
        for u, d, sk in zip(s.up, s.dec, reversed(skips[:-1])):
            x = d(torch.cat([u(x), sk], 1), z)
        o = s.head(x)
        z1 = o[:, 0]; z2 = z1 + F.softplus(o[:, 1]); z3 = z2 + F.softplus(o[:, 2])
        return torch.stack([z1, z2, z3], 1)


def loss_fn(logit, y, v):
    v = v[:, None].expand_as(y)
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(5.0, device=y.device), reduction="none")
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logit) * v
    dice = 1 - (2 * (p * y).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + (y * v).sum((0, 2, 3)) + 1)
    return bce + dice.mean()


def augment(x, y, v):
    k = np.random.randint(4)
    x, y, v = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3)), torch.rot90(v, k, (1, 2))
    if np.random.rand() < 0.5:
        x, y, v = x.flip(3), y.flip(3), v.flip(2)
    r, c = np.random.randint(0, 96 - CROP + 1, 2)
    return x[:, :, r:r + CROP, c:c + CROP], y[:, :, r:r + CROP, c:c + CROP], v[:, r:r + CROP, c:c + CROP]


def predict(model, M, Z, bs=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(M), bs):
            x = torch.from_numpy(M[i:i + bs].astype(np.float32)).to(dev); z = torch.from_numpy(Z[i:i + bs]).to(dev)
            out.append(torch.sigmoid(model(x, z)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3, 96, 96), np.float32)


def best_f1(P, Y, V):
    ths = np.linspace(0.05, 0.95, 37); best = (0, 0.5)
    for t in ths:
        pr = (P >= t) & V[:, None]; yy = (Y > 0) & V[:, None]
        tp = (pr & yy).sum(); fp = (pr & ~yy).sum(); fn = (~pr & yy).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best[0]:
            best = (f, t)
    return best


if __name__ == "__main__":
    si = load_index(); years = sorted(si.year.unique())
    # обучение: не более 3 случайных срезов на пожар (память и баланс между крупными и мелкими пожарами)
    rng = np.random.default_rng(0); si["r"] = rng.random(len(si))
    si["keep"] = (si.part != "unet_train") | (si.groupby("fire_id").r.rank(method="first") <= 3)
    M, Yt, Vt, Z, PART, YR, SL = [], [], [], [], [], [], []
    for y in years:
        X, Y, V = load_year(y); s = si[(si.year == y) & si.keep].sort_values("slice"); k = s.slice.values
        M.append(nn_maps(X[k], V[k]).astype(np.float16)); Yt.append(Y[k]); Vt.append(V[k]); Z.append(scalars_unet(s))
        PART += list(s.part); YR += [y] * len(s); SL += list(k); del X, Y, V
        print("loaded", y, len(s), flush=True)
    M = np.concatenate(M); Yt = np.concatenate(Yt); Vt = np.concatenate(Vt); Z = np.concatenate(Z); PART = np.array(PART); YR = np.array(YR)
    tr = np.flatnonzero(PART == "unet_train"); sel = np.flatnonzero(PART == "calib_sel")
    print("train slices", len(tr), "| sel slices", len(sel), "| maps", M.shape, flush=True)
    model = CompactFront(cin=M.shape[1], ns=Z.shape[1]).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr) / BS)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    log, best = [], (-1, None)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); perm = np.random.permutation(tr); L = []
        for i in range(0, len(perm), BS):
            b = np.sort(perm[i:i + BS])
            x = torch.from_numpy(M[b].astype(np.float32)).to(dev); y = torch.from_numpy(Yt[b].astype(np.float32)).to(dev)
            v = torch.from_numpy(Vt[b].astype(np.float32)).to(dev); z = torch.from_numpy(Z[b]).to(dev)
            x, y, v = augment(x, y, v)
            loss = loss_fn(model(x, z), y, v)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); L.append(loss.item())
        P = predict(model, M[sel], Z[sel]); f, th = best_f1(P, Yt[sel], Vt[sel])
        log.append(dict(epoch=ep, loss=np.mean(L), sel_F1=f, sel_th=th, sec=time.time() - t0))
        print(f"epoch {ep:2d} loss {np.mean(L):.4f} sel F1 {f:.4f} @ {th:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if f > best[0]:
            best = (f, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); torch.save(dict(state=best[1], cin=M.shape[1], ns=Z.shape[1]), os.path.join(OUT, "unet.pt"))
    pd.DataFrame(log).to_csv(os.path.join(OUT, "unet_train_log.csv"), index=False)
    SL = np.array(SL)
    for y in years:   # вероятности для всех срезов года (индекс = номер среза в файле года); для обучающих лет - по всем срезам
        s = si[si.year == y].sort_values("slice")
        if (s.part == "unet_train").all():
            X, Y, V = load_year(y); P = []
            for i in range(0, len(s), 2000):
                P.append(predict(model, nn_maps(X[i:i + 2000], V[i:i + 2000]).astype(np.float16), scalars_unet(s.iloc[i:i + 2000])))
            P = np.concatenate(P); del X, Y, V
        else:
            k = np.flatnonzero(YR == y); P = np.zeros((len(s), 3, 96, 96), np.float32); P[SL[k]] = predict(model, M[k], Z[k])
        np.save(os.path.join(OUT, f"unet_probs_{y}.npy"), (P * 255).round().astype(np.uint8))
    print("best sel F1", round(best[0], 4))
