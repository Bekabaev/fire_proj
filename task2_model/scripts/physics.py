"""Физическая часть гибрида: скорость распространения по Ротермелу (1972, с поправками Albini 1976 и динамической
травяной нагрузкой Scott & Burgan 2005), эллипс по ветру (Anderson 1983 / Alexander 1985), время прихода огня на сетке
кратчайшим путём (аналог level set / minimum travel time, как у ELMFIRE/FARSITE в упрощённом виде).

Погода внутри сцены однородна (как в FireSpread_Report_2). Влажность мёртвого топлива - равновесная влажность по T и RH
(Simard 1968) с поправкой на осадки; влажность живой травы - по NDVI (чем зеленее, тем влажнее).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# --- модели топлива Scott & Burgan (2005), единицы: нагрузка т/акр -> lb/ft², SAV 1/ft, глубина ft, Mx доля
# (1h, 10h, 100h, live herb, live woody) нагрузка т/акр; SAV (1h, herb, woody); depth; Mx_dead; heat 8000 BTU/lb
FM = {
    101: dict(load=(0.10, 0.0, 0.0, 0.30, 0.0), sav=(2200, 2000, 1500), depth=0.4, mx=0.15, dyn=True),   # GR1
    102: dict(load=(0.10, 0.0, 0.0, 1.00, 0.0), sav=(2000, 1800, 1500), depth=1.0, mx=0.15, dyn=True),   # GR2
    104: dict(load=(0.25, 0.0, 0.0, 1.90, 0.0), sav=(2000, 1800, 1500), depth=2.0, mx=0.15, dyn=True),   # GR4
    121: dict(load=(0.20, 0.0, 0.0, 0.50, 0.65), sav=(2000, 1800, 1800), depth=0.9, mx=0.15, dyn=True),  # GS1
    161: dict(load=(0.20, 0.90, 1.50, 0.0, 0.20), sav=(2000, 1800, 1600), depth=0.6, mx=0.20, dyn=True), # TU1
}
T_AC = 2000.0 / 43560.0     # т/акр -> lb/ft²


def emc(T, RH):
    """Равновесная влажность мёртвого топлива, доля (Simard 1968; T °C, RH %)."""
    Tf = T * 9 / 5 + 32
    m = np.where(RH < 10, 0.03229 + 0.281073 * RH - 0.000578 * RH * Tf,
                 np.where(RH < 50, 2.22749 + 0.160107 * RH - 0.01478 * Tf,
                          21.0606 + 0.005565 * RH ** 2 - 0.00035 * RH * Tf - 0.483199 * RH))
    return np.clip(m / 100, 0.02, 0.35)


def ros_rothermel(fm, m1, m10, m100, mherb, mwood, U_mid_ms, slope_deg=0.0):
    """Скорость головного фронта, м/мин. fm - код FBFM40; влажности - доли; U_mid_ms - ветер на средней высоте пламени."""
    if fm not in FM:
        return 0.0
    p = FM[fm]
    w = np.array(p["load"]) * T_AC
    s1, sh, sw = p["sav"]
    sav = np.array([s1, 109.0, 30.0, sh, sw])
    M = np.array([m1, m10, m100, mherb, mwood], float)
    if p["dyn"] and w[3] > 0:      # перенос травяной нагрузки в мёртвую по влажности (Scott & Burgan 2005)
        T = np.clip((1.2 - mherb) / (1.2 - 0.3), 0, 1)
        w = w.copy(); dead_h = w[3] * T; w[3] -= dead_h
        # мёртвая трава - отдельный класс с SAV травы и влажностью 1h
        w = np.r_[w, dead_h]; sav = np.r_[sav, sh]; M = np.r_[M, m1]
        live = np.array([False, False, False, True, True, False])
    else:
        live = np.array([False, False, False, True, True])
    ok = w > 0
    if not ok.any():
        return 0.0
    rho_p, h, St, Se = 32.0, 8000.0, 0.0555, 0.010
    a = sav * w / rho_p                                    # площадь поверхности частиц
    A_d, A_l = a[~live & ok].sum(), a[live & ok].sum()
    A_t = A_d + A_l
    f = np.where(live, a / max(A_l, 1e-12), a / max(A_d, 1e-12)) * ok
    fd, fl = A_d / A_t, A_l / A_t
    wn = w * (1 - St)
    sig_d = (f * sav)[~live].sum(); sig_l = (f * sav)[live].sum() if A_l > 0 else 0.0
    sig = fd * sig_d + fl * sig_l
    wn_d = (f * wn)[~live].sum(); wn_l = (f * wn)[live].sum() if A_l > 0 else 0.0
    Md = (f * M)[~live].sum()
    # влажность угасания живого топлива (Albini 1976)
    if A_l > 0:
        ex = np.exp(-138.0 / np.maximum(sav, 1e-6))
        W = (w * ex)[~live].sum() / max((w * np.exp(-500.0 / np.maximum(sav, 1e-6)))[live].sum(), 1e-12)
        Mf_dead = (w * ex * M)[~live].sum() / max((w * ex)[~live].sum(), 1e-12)
        Mx_l = max(2.9 * W * (1 - Mf_dead / p["mx"]) - 0.226, p["mx"])
        Ml = (f * M)[live].sum()
    else:
        Mx_l, Ml = p["mx"], 0.0
    def eta_m(Mf, Mx):
        r = min(Mf / Mx, 1.0)
        return max(1 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3, 0.0)
    eta_s = min(0.174 * Se ** -0.19, 1.0)
    w_load = w.sum(); beta = w_load / p["depth"] / rho_p
    beta_op = 3.348 * sig ** -0.8189; rb = beta / beta_op
    A = 133.0 * sig ** -0.7913
    Gmax = sig ** 1.5 / (495 + 0.0594 * sig ** 1.5)
    G = Gmax * rb ** A * np.exp(A * (1 - rb))
    IR = G * h * (wn_d * eta_m(Md, p["mx"]) * eta_s + (wn_l * eta_m(Ml, Mx_l) * eta_s if A_l > 0 else 0.0))   # BTU/ft²/min
    xi = np.exp((0.792 + 0.681 * sig ** 0.5) * (beta + 0.1)) / (192 + 0.2595 * sig)
    U = U_mid_ms * 196.85                                     # м/с -> ft/min
    C = 7.47 * np.exp(-0.133 * sig ** 0.55); B = 0.02526 * sig ** 0.54; E = 0.715 * np.exp(-3.59e-4 * sig)
    U = min(U, 0.9 * IR)                                      # предел ветра
    phi_w = C * U ** B * rb ** -E
    phi_s = 5.275 * beta ** -0.3 * np.tan(np.deg2rad(slope_deg)) ** 2
    rho_b = w_load / p["depth"]
    eps = np.where(live, 0.0, np.exp(-138.0 / sav))
    Qig = 250 + 1116 * M
    heat_sink = rho_b * ((f * eps * Qig)[~live].sum() * fd + (f * np.exp(-138.0 / sav) * Qig)[live].sum() * fl)
    if heat_sink <= 0 or IR <= 0:
        return 0.0
    R = IR * xi * (1 + phi_w + phi_s) / heat_sink             # ft/min
    return float(R * 0.3048)                                  # м/мин


def length_to_breadth(Umid_ms):
    """Отношение осей эллипса (Anderson 1983; ветер на средней высоте пламени, миль/ч)."""
    u = Umid_ms * 2.237
    return min(0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397, 8.0)


def dir_ros_factor(theta, LB):
    """Скорость в направлении theta (рад, 0 = по ветру) / скорость головы для эллипса с отношением LB (точечный очаг)."""
    e = np.sqrt(max(1 - 1 / LB ** 2, 0.0))
    return (1 - e) / (1 - e * np.cos(theta))


# 16 соседей (ход коня + 8 соседей) - меньше анизотропия сетки
NB = [(dr, dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2) if (dr, dc) != (0, 0) and np.gcd(abs(dr), abs(dc)) == 1]


def arrival_time(ros_head, burnable, sources, wind_to_rad, LB, cell_m=500.0, t0=None):
    """Время прихода огня (мин) от источников. ros_head [H,W] м/мин; wind_to_rad - куда дует ветер (азимут, рад от севера по часовой).
    t0 [H,W] - стартовое время для источников (для пошагового расчёта по суткам)."""
    H, W = ros_head.shape; n = H * W
    idx = np.arange(n).reshape(H, W)
    rows, cols, wts = [], [], []
    for dr, dc in NB:
        az = np.arctan2(dc, -dr)                               # азимут шага (север = −r)
        fac = dir_ros_factor(az - wind_to_rad, LB)
        dist = cell_m * np.hypot(dr, dc)
        r0, r1 = max(0, -dr), min(H, H - dr); c0, c1 = max(0, -dc), min(W, W - dc)
        a = idx[r0:r1, c0:c1]; b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        # скорость на ребре - среднее гармоническое скоростей двух ячеек (время = сумма половин пути)
        ra = ros_head[r0:r1, c0:c1] * fac; rb = ros_head[r0 + dr:r1 + dr, c0 + dc:c1 + dc] * fac
        ok = burnable[r0 + dr:r1 + dr, c0 + dc:c1 + dc] & (ra > 0) & (rb > 0)
        t = 0.5 * dist / np.maximum(ra, 1e-6) + 0.5 * dist / np.maximum(rb, 1e-6)
        rows.append(a[ok]); cols.append(b[ok]); wts.append(t[ok])
    # виртуальный узел-источник n со стартовыми временами
    src = np.flatnonzero(sources)
    st = np.zeros(len(src)) if t0 is None else t0.ravel()[src]
    rows.append(np.full(len(src), n)); cols.append(src); wts.append(st + 1e-6)
    g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 1, n + 1))
    d = dijkstra(g, directed=True, indices=n)
    return d[:n].reshape(H, W)
