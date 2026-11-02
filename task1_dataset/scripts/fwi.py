"""Канадская система индексов пожарной опасности (FWI), Van Wagner (1987), формулы как в cffdrs.
Входы - значения в местный полдень: T (°C), RH (%), ветер (км/ч), осадки за 24 ч (мм)."""
import numpy as np

EL = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]   # длина дня для DMC, широты > 30° N
FL = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]   # поправка длины дня для DC


def ffmc(t, rh, ws, rain, f0):
    rh = min(rh, 100.0)
    mo = 147.2 * (101 - f0) / (59.5 + f0)
    if rain > 0.5:
        rf = rain - 0.5
        add = 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
        mo = mo + add + (0.0015 * (mo - 150) ** 2 * np.sqrt(rf) if mo > 150 else 0)
        mo = min(mo, 250.0)
    ed = 0.942 * rh ** 0.679 + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - t) * (1 - np.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8)
        m = ed + (mo - ed) * 10 ** (-ko * 0.581 * np.exp(0.0365 * t))
    else:
        ew = 0.618 * rh ** 0.753 + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - t) * (1 - np.exp(-0.115 * rh))
        if mo < ew:
            k1 = 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) / 100) ** 8)
            m = ew - (ew - mo) * 10 ** (-k1 * 0.581 * np.exp(0.0365 * t))
        else:
            m = mo
    return float(np.clip(59.5 * (250 - m) / (147.2 + m), 0, 101))


def dmc(t, rh, rain, d0, month):
    t = max(t, -1.1)
    rk = 1.894 * (t + 1.1) * (100 - rh) * EL[month - 1] * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20 + 280 / np.exp(0.023 * d0)
        b = 100 / (0.5 + 0.3 * d0) if d0 <= 33 else (14 - 1.3 * np.log(d0) if d0 <= 65 else 6.2 * np.log(d0) - 17.2)
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - np.log(wmr - 20))
    else:
        pr = d0
    return float(max(max(pr, 0) + rk, 0))


def dc(t, rain, d0, month):
    t = max(t, -2.8)
    pe = max((0.36 * (t + 2.8) + FL[month - 1]) / 2, 0)
    if rain > 2.8:
        rw = 0.83 * rain - 1.27
        smi = 800 * np.exp(-d0 / 400)
        dr = max(d0 - 400 * np.log(1 + 3.937 * rw / smi), 0)
    else:
        dr = d0
    return float(dr + pe)


def isi(ws, f):
    fm = 147.2 * (101 - f) / (59.5 + f)
    sf = 19.115 * np.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
    return float(sf * np.exp(0.05039 * ws))


def bui(p, d):
    if p <= 0 and d <= 0:
        return 0.0
    if p <= 0.4 * d:
        u = 0.8 * p * d / (p + 0.4 * d)
    else:
        u = p - (1 - 0.8 * d / (p + 0.4 * d)) * (0.92 + (0.0114 * p) ** 1.7)
    return float(max(u, 0))


def fwi(i, u):
    bb = 0.1 * i * (0.626 * u ** 0.809 + 2) if u <= 80 else 0.1 * i * (1000 / (25 + 108.64 * np.exp(-0.023 * u)))
    return float(bb if bb <= 1 else np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647))


def run(daily, f0=85.0, p0=6.0, d0=15.0):
    """daily: DataFrame с колонками date, t, rh, ws_kmh, rain24. Возвращает FFMC, DMC, DC, ISI, BUI, FWI по дням."""
    out = []
    for r in daily.itertuples():
        m = r.date.month
        f0 = ffmc(r.t, r.rh, r.ws_kmh, r.rain24, f0)
        p0 = dmc(r.t, r.rh, r.rain24, p0, m)
        d0 = dc(r.t, r.rain24, d0, m)
        i = isi(r.ws_kmh, f0); u = bui(p0, d0)
        out.append(dict(FFMC=f0, DMC=p0, DC=d0, ISI=i, BUI=u, FWI=fwi(i, u)))
    return out
