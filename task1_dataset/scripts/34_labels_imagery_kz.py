"""Шаг 34 (датасет v4). Уточнение меток GFA по снимкам Landsat / Sentinel-2 (шаг 33).

1) Порог гари по ячейкам с известным состоянием (по всем снимкам): «сгорело» - ячейки пожара со временем по VIIRS (src 1/5)
   ≥ 24 ч до снимка; «не сгорело» - ячейки окна вне пожаров года (буфер 3 ячейки). Порог dNBR - максимум F1 разделения,
   «не сгорело» - dNBR < половины порога; ячейка считается при безоблачности ≥ 50 %.
2) Надёжность снимка - распознано ≥ 70 % заведомо сгоревших ячеек (≥ 20); «не сгорело» используется только на надёжных снимках.
3) Границы времени прихода: верхняя - первый снимок «сгорело», нижняя - последний надёжный «не сгорело» (как шаг 30).
   Время позже верхней границы более чем на 12 ч сдвигается раньше (для любого источника); позже не сдвигается - «нет гари» ненадёжно.
   Источник = 6 «уточнено по снимку».
4) Пропущенная гарь: ячейки вне всех пожаров года в буфере 3 ячейки вокруг пожара, сгоревшие на снимке не позже конца пожара + 48 ч,
   с надёжным «не сгорело» до этого и интервалом ≤ 96 ч - добавляются в пожар со временем посередине интервала, источник = 7.
Выход: data/ext_kz/arrival_img/arr_<год>.npz (формат шага 18 + src 6/7), data/ext_kz/qa_labels_img.csv / .txt.
"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.dirname(__file__))
from kz_common import KZ, GH, GW

ID = os.path.join(KZ, "imagery"); OD = os.path.join(KZ, "arrival_img"); os.makedirs(OD, exist_ok=True)
ev = pd.read_csv(os.path.join(KZ, "events_kz.csv")).set_index("fire_id")
files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(ID, "*.npz"))}
print("imagery files", len(files), flush=True)


def grids(y):
    a = np.load(os.path.join(KZ, "arrival", f"arr_{y}.npz"))
    AR = np.full((GH, GW), np.nan, np.float32); FID = np.zeros((GH, GW), np.int32); SRC = np.zeros((GH, GW), np.uint8)
    AR[a["r"], a["c"]] = a["arr_h"]; FID[a["r"], a["c"]] = a["fire_id"]; SRC[a["r"], a["c"]] = a["src"]
    return a, AR, FID, SRC


def load(fid):
    try:
        z = np.load(files[fid])
    except Exception:
        return None
    if "dn" not in z.files or len(z["times_h"]) == 0:
        return None
    return {k: z[k] for k in z.files}


# ---------- 1. порог ----------
pos, neg = [], []
for y in sorted(ev.loc[[f for f in files if f in ev.index]].year.unique()):
    a, AR, FID, SRC = grids(y)
    for fid in [f for f in files if f in ev.index and ev.loc[f].year == y]:
        z = load(fid)
        if z is None:
            continue
        K, h, w = z["dn"].shape; ws = (slice(z["r0"], z["r0"] + h), slice(z["c0"], z["c0"] + w))
        mine = FID[ws] == fid; A = AR[ws]; S = SRC[ws]; bgc = ~binary_dilation(FID[ws] > 0, iterations=3)
        for k in range(K):
            ok = z["cf"][k] >= 50; dn = z["dn"][k] / 1000
            kn = ok & mine & np.isin(S, (1, 5)) & (A < z["times_h"][k] - 24)
            pos.append(dn[kn]); neg.append(dn[ok & bgc][::5])
pos = np.concatenate(pos); neg = np.concatenate(neg)
grid = np.linspace(0.02, 0.4, 39); f1 = []
for t in grid:
    tp = (pos >= t).sum(); fp = (neg >= t).sum() * 5; fn = (pos < t).sum(); f1.append(2 * tp / max(2 * tp + fp + fn, 1))
THR = float(grid[int(np.argmax(f1))])
print(f"known burned {len(pos)}, background {len(neg) * 5}; dNBR threshold {THR:.3f}, F1 {max(f1):.3f}, "
      f"recall {(pos >= THR).mean():.3f}, false-positive rate {(neg >= THR).mean():.4f}", flush=True)

# ---------- 2–4. метки ----------
qa = []
for y in sorted(ev.year.unique()):
    if not os.path.exists(os.path.join(KZ, "arrival", f"arr_{y}.npz")):
        continue
    a, AR, FID, SRC = grids(y); AR2, FID2, SRC2 = AR.copy(), FID.copy(), SRC.copy()
    for fid in [f for f in files if f in ev.index and ev.loc[f].year == y]:
        z = load(fid)
        if z is None:
            continue
        K, h, w = z["dn"].shape; ws = (slice(z["r0"], z["r0"] + h), slice(z["c0"], z["c0"] + w)); times = z["times_h"]
        mine = FID[ws] == fid; A = AR[ws]; S = SRC[ws]
        ok = z["cf"] >= 50; dn = z["dn"] / 1000
        st = np.where(ok, np.where(dn >= THR, 1, np.where(dn < THR / 2, 0, -1)), -1)
        kn = mine & np.isin(S, (1, 5))
        rel = []
        for k in range(K):
            m = kn & (A < times[k] - 24) & (st[k] >= 0)
            rel.append(m.sum() >= 20 and (st[k][m] == 1).mean() >= 0.7)
        lo = np.full((h, w), -np.inf, np.float32); hi = np.full((h, w), np.inf, np.float32)
        for k in range(K):
            hi = np.where(st[k] == 1, np.minimum(hi, times[k]), hi)
            if rel[k]:
                lo = np.where(st[k] == 0, np.maximum(lo, times[k]), lo)
        cons = mine & (lo < hi)
        # позже не сдвигаем: «нет гари» на снимке ненадёжно (сигнал степной гари слабый и быстро пропадает при отрастании травы)
        late = cons & (A > hi + 12); early = np.zeros_like(late)
        A2 = A.copy(); S2 = S.copy()
        A2[late] = np.where(np.isfinite(lo[late]), (np.maximum(lo[late], A[late] - 240) + hi[late]) / 2, hi[late] - 6)
        A2[early] = np.where(np.isfinite(hi[early]), (lo[early] + hi[early]) / 2, lo[early] + 6)
        S2[late | early] = 6
        t1 = np.nanmax(A[mine]); t0 = np.nanmin(A[mine])
        near = binary_dilation(mine, iterations=3) & (FID[ws] == 0)
        add = near & np.isfinite(hi) & (hi <= t1 + 48) & np.isfinite(lo) & (hi - lo <= 96) & (lo < hi) & (hi >= t0)
        A2[add] = (np.maximum(lo[add], t0) + hi[add]) / 2; S2[add] = 7
        mm = mine | add
        AR2[ws] = np.where(mm, A2, AR2[ws]); SRC2[ws] = np.where(mm, S2, SRC2[ws]); FID2[ws] = np.where(add, fid, FID2[ws])
        qa.append(dict(fire_id=fid, year=y, split=ev.loc[fid].split, snapshots=K, reliable=int(sum(rel)), cells=int(mine.sum()),
                       bounded=int((mine & (np.isfinite(lo) | np.isfinite(hi))).sum()), moved_earlier=int(late.sum()), moved_later=int(early.sum()),
                       moved_date_only=int(((late | early) & (S == 4)).sum()), added=int(add.sum())))
    r, c = np.nonzero(FID2 > 0)
    np.savez_compressed(os.path.join(OD, f"arr_{y}.npz"), r=r.astype(np.int32), c=c.astype(np.int32), fire_id=FID2[r, c],
                        arr_h=AR2[r, c], src=SRC2[r, c], offset_h=a["offset_h"])
    print(y, "done", flush=True)
q = pd.DataFrame(qa); q.to_csv(os.path.join(KZ, "qa_labels_img.csv"), index=False)
s = q.groupby("split")[["snapshots", "reliable", "cells", "bounded", "moved_earlier", "moved_later", "moved_date_only", "added"]].sum()
txt = (f"dNBR threshold {THR:.3f} (F1 {max(f1):.3f}, recall {(pos >= THR).mean():.3f}, FPR {(neg >= THR).mean():.4f})\n"
       f"fires with imagery {len(q)}, with ≥1 reliable snapshot {(q.reliable > 0).sum()}\n" + s.to_string())
open(os.path.join(KZ, "qa_labels_img.txt"), "w", encoding="utf8").write(txt + "\n"); print(txt)
