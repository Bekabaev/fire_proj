"""Гибрид «сеть → скорость → решатель»: U-Net предсказывает скорость огня в 16 направлениях и задержку старта на кромке,
дифференцируемый решатель (кратчайший путь, мягкий минимум по 16 соседям, K итераций) считает время прихода огня T в каждой ячейке,
вероятность выгорания к горизонту h = sigmoid((h − T) / σ). Учится целиком по тем же меткам и тому же лоссу, что FrontV3.

Скорость: log S_k(x) = A(x) + C_k(x) + B(x) · (e_k · w), где e_k - единичный вектор направления k, w - ветер окна 24 ч
(вектор поворачивается при аугментации вместе с картой, поэтому ветровая часть равноправна для всех поворотов).
Без погоды (nowx) ветрового члена нет. hard=True (RUN=swxh) - жёсткий минимум вместо мягкого (мягкий не имеет неподвижной точки и занижает T).
RES=1 - добавочный свободный выход сети к логиту (решатель + поправка).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common_v3 import FiLMBlock, HZ, MAPS

OFFS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1),
        (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2)]      # (dr, dc): строки вниз = на юг
DIST = torch.tensor([0.5 * math.hypot(r, c) for r, c in OFFS])                        # км
EV = torch.tensor([[c / math.hypot(r, c), -r / math.hypot(r, c)] for r, c in OFFS])  # (восток, север)
BIG = 200.0                                                                           # ч: «не дошёл»


class SolverNet(nn.Module):
    def __init__(s, cin, ns, ch=(16, 32, 64, 128), nz=64, K=64, tau=0.5, res=False, wind_pos=None, hard=False):
        super().__init__()
        s.K, s.tau, s.res, s.wind_pos, s.hard = K, tau, res, wind_pos, hard
        s.zm = nn.Sequential(nn.Linear(ns, 64), nn.GELU(), nn.Linear(64, nz))
        s.enc = nn.ModuleList(); c = cin
        for k in ch:
            s.enc.append(FiLMBlock(c, k, nz)); c = k
        s.up = nn.ModuleList(); s.dec = nn.ModuleList()
        for k in reversed(ch[:-1]):
            s.up.append(nn.ConvTranspose2d(c, k, 2, stride=2)); s.dec.append(FiLMBlock(2 * k, k, nz)); c = k
        s.head = nn.Conv2d(c, 1 + 16 + 1 + 1 + (len(HZ) if res else 0), 1)      # A, C_k, B, задержка, [поправка]
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)
        with torch.no_grad():
            s.head.bias[0] = math.log(0.1); s.head.bias[18] = 1.0                 # старт: 0,1 км/ч, задержка ~1,3 ч
        s.log_sigma = nn.Parameter(torch.tensor(math.log(4.0)))                   # разброс времени прихода, ч
        s.register_buffer("dist", DIST.clone()); s.register_buffer("ev", EV.clone())
        s.ib = MAPS.index("burned_frac")

    def features(s, x, z):
        zz = s.zm(z); skips = []
        for i, b in enumerate(s.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = b(x, zz); skips.append(x)
        for u, d, sk in zip(s.up, s.dec, reversed(skips[:-1])):
            x = d(torch.cat([u(x), sk], 1), zz)
        return s.head(x)

    def speeds(s, o, z):
        logS = o[:, :1] + 0.5 * torch.tanh(o[:, 1:17])
        if s.wind_pos is not None:                                                # ветер окна 24 ч (нормированный)
            w = torch.stack([z[:, s.wind_pos[0]], z[:, s.wind_pos[1]]], 1)        # [B,2]
            proj = w @ s.ev.T                                                     # [B,16]
            logS = logS + o[:, 17:18] * proj[:, :, None, None]
        return torch.exp(logS.clamp(-7, 3))                                       # 0,001 … 20 км/ч

    def _steps(s, T, cost, n):
        H, W = T.shape[-2:]
        for _ in range(n):
            V = F.pad(T[:, None] + cost, (2, 2, 2, 2), value=BIG)                 # [B,16,H+4,W+4]: время прихода в соседа через x
            cand = [V[:, k, 2 - r:2 - r + H, 2 - c:2 - c + W] for k, (r, c) in enumerate(OFFS)]
            cand = torch.stack(cand + [T], 1)
            if s.hard:                                                            # жёсткий минимум: неподвижная точка, градиент по выбранному пути
                T = cand.min(1).values
            else:                                                                 # мягкий минимум (первая версия): смещает T вниз до tau·ln17 за итерацию
                T = -s.tau * torch.logsumexp(-cand / s.tau, 1)
            T = T.clamp(max=BIG)
        return T

    def arrival(s, x, z, o=None):
        o = s.features(x, z) if o is None else o
        S = s.speeds(o, z)                                                        # [B,16,H,W]
        cost = s.dist[None, :, None, None] / S                                    # ч на переход из x в соседа k
        burned = x[:, s.ib] > 0
        T = torch.where(burned, F.softplus(o[:, 18]), torch.full_like(o[:, 18], BIG))
        blk = 8
        for i in range(0, s.K, blk):
            if s.training:
                T = checkpoint(s._steps, T, cost, blk, use_reentrant=False)
            else:
                T = s._steps(T, cost, blk)
        return T, o

    def forward(s, x, z):
        T, o = s.arrival(x, z)
        sig = torch.exp(s.log_sigma)
        hz = torch.tensor(HZ, device=x.device, dtype=x.dtype)[None, :, None, None]
        logit = (hz - T[:, None]) / sig
        if s.res:
            logit = logit + o[:, 19:19 + len(HZ)]
        return logit
