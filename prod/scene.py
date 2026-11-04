"""Сцена 96 × 96 (500 м) в формате обучающего среза v3: 23 канала X, valid и строка индекса (сезон, состояние пожара, погода).

Состояние пожара задаётся маской гари и временем выгорания ячеек (часы до t), термоточки - списком (строка, столбец, часов до t, FRP)
в координатах окна. Если термоточек нет (сценарий симулятора), они ставятся на кромку, выгоревшую за последние 12 ч, -
так выглядит активный пожар в данных FIRMS.
"""
import numpy as np, pandas as pd
from scipy.ndimage import binary_erosion

from .config import CH, CELL, N

FRP_DEFAULT = 15.0                                          # МВт, типичная мощность степной термоточки VIIRS


def synthetic_hotspots(burned, age_h):
    """Термоточки на кромке гари, выгоревшей ≤ 12 ч назад."""
    edge = burned & ~binary_erosion(burned, border_value=0)
    fresh = edge & (age_h <= 12)
    if not fresh.any():
        fresh = edge
    r, c = np.nonzero(fresh)
    return [(int(i), int(j), float(min(age_h[i, j], 11.0)), FRP_DEFAULT) for i, j in zip(r, c)]


def build(static, covered, burned, age_h, t_utc, ign_rc, k_12h=0, hotspots=None, weather=None):
    """-> X [1,23,96,96] f32, valid [1,96,96] bool, row (dict для base_scalars / weather_vec).

    static - каналы Landscape.static; burned - bool [96,96]; age_h - часы с выгорания (0..240) для burned;
    ign_rc - точка возгорания в координатах окна; hotspots - [(r, c, часов до t, frp)] или None (синтетические на кромке)."""
    n = burned.shape[0]
    age = np.where(burned, np.clip(age_h, 0, 240), 0).astype(np.float32)
    hs = synthetic_hotspots(burned, age) if hotspots is None else hotspots
    af12 = np.zeros((n, n), np.float32); fp12 = np.zeros((n, n), np.float32); af24 = np.zeros((n, n), np.float32); afa = np.zeros((n, n), np.float32)
    for r, c, dt, frp in hs:
        if 0 <= r < n and 0 <= c < n:
            afa[r, c] += 1
            if dt < 24:
                af24[r, c] += 1
            if dt < 12:
                af12[r, c] += 1; fp12[r, c] += frp
    gr, gc = np.mgrid[0:n, 0:n]
    L = dict(static, burned_frac=burned.astype(np.float32), burn_age_h=age, af_count_12h=af12, af_frp_12h=fp12, af_count_24h=af24,
             af_count_all=afa, dist_ign_km=np.hypot(gr - ign_rc[0], gc - ign_rc[1]) * CELL / 1000)
    X = np.stack([L[c] for c in CH]).astype(np.float32)[None]
    valid = (covered & ~burned)[None]
    t = pd.Timestamp(t_utc)
    row = dict(t_utc=t, hour_utc=t.hour, k_12h=int(k_12h), state_km2=float(burned.sum() * (CELL / 1000) ** 2), af_12h=int(af12.sum()))
    row.update(weather or {})
    return X, valid, row


def disk(shape, rc, radius_cells):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (yy - rc[0]) ** 2 + (xx - rc[1]) ** 2 <= radius_cells ** 2
