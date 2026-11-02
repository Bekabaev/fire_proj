"""Задание 2, шаг 2. Физический расчёт распространения для каждого среза (аналог ELMFIRE-ветки FireSpread_Report_2).

Ротермел (physics.py) по FBFM40 ячейки, влажность мёртвого топлива - равновесная по T и RH (RH средняя из VPD и T),
+ поправка на осадки; живой травы - по NDVI. Ветер 10 м -> средняя высота пламени ×0.4 (открытая степь).
Прошлые гари снижают запас топлива (×0.3 - гарь этого/прошлого года, ×0.7 - двухлетняя).
Источники - сгоревшие ячейки и термоточки за последние сутки (как начальная маска ELMFIRE в отчёте).
Для горизонта h погода = средняя за [t, t+h). Время прихода считается при множителе скорости 1:
при множителе m время = T/m, поэтому калибровка m не требует пересчёта.
Выход: out/phys_T_<год>.npy (float16, часы до прихода огня при m=1, [S,3,96,96]; inf - недостижимо)
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from physics import ros_rothermel, emc, length_to_breadth, arrival_time

WAF = 0.4
NB_BINS = np.linspace(0.05, 0.75, 15)


def mherb_of_ndvi(nd):
    return np.clip(0.3 + (nd - 0.15) / 0.45 * 1.7, 0.3, 2.0)


def slice_physics(X, row):
    g = lambda k: X[CH.index(k)].astype(np.float32)
    fb = g("fbfm40"); nd = g("ndvi"); nd = np.where(np.isfinite(nd), nd, np.nanmedian(nd) if np.isfinite(nd).any() else 0.3)
    prior = g("prior_burn"); state = g("burned_frac") > 0.5; af = g("af_count_24h") > 0
    src = state | af
    burnable = np.isin(fb, list((101, 102, 104, 121, 161))) & (g("nonburnable_frac") < 0.7)
    fuel_k = np.where(prior >= 1, 0.3, np.where(prior > 0, 0.7, 1.0))
    nbin = np.digitize(nd, NB_BINS)
    out = np.full((3, 96, 96), np.inf, np.float32)
    for i, h in enumerate(HZ):
        T = row.get(f"T_mean_{h}h"); vpd = row.get(f"VPD_mean_{h}h"); ws = row.get(f"ws_mean_{h}h")
        u, v, P = row.get(f"wind_u_{h}h"), row.get(f"wind_v_{h}h"), row.get(f"P_sum_{h}h")
        if not np.isfinite([T, vpd, ws, u, v]).all():
            continue
        es = 0.6108 * np.exp(17.27 * T / (T + 237.3)); RH = float(np.clip(100 * (1 - vpd / es), 5, 100))
        m1 = float(emc(T, RH)) + (0.08 if P > 2 else 0.0)
        Umid = WAF * ws
        ros = np.zeros((96, 96), np.float32)
        for fm in (101, 102, 104, 121, 161):
            mk = burnable & (fb == fm)
            if not mk.any():
                continue
            for b in np.unique(nbin[mk]):
                mb = mk & (nbin == b)
                ndv = NB_BINS[min(b, len(NB_BINS) - 1)]
                ros[mb] = ros_rothermel(fm, m1, m1 + 0.01, m1 + 0.02, float(mherb_of_ndvi(ndv)), 1.0, Umid)
        ros *= fuel_k
        if not (src.any() and (ros > 0).any()):
            continue
        wind_to = np.arctan2(u, v)          # азимут, куда дует (u - на восток, v - на север)
        a = arrival_time(ros, burnable | src, src, wind_to, length_to_breadth(Umid))
        out[i] = a / 60.0
    return out


if __name__ == "__main__":
    si = load_index()
    for y in sorted(si.year.unique()):
        fo = os.path.join(OUT, f"phys_T_{y}.npy")
        s = si[si.year == y].sort_values("slice")
        if os.path.exists(fo) or not (s.part != "unet_train").any():
            continue
        X, Y, V = load_year(y); t0 = time.time()
        res = np.stack([slice_physics(X[r.slice], r._asdict()) for r in s.itertuples()])
        np.save(fo, res.astype(np.float16))
        print(y, len(s), f"{time.time() - t0:.0f} s, reachable share {np.isfinite(res).mean():.3f}", flush=True)
