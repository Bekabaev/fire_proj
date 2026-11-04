"""Физический симулятор распространения травяного пожара на сетке 500 м (эталон для синтетических сценариев и физический бейзлайн).

Скорость - канадская система FBP, тип топлива O-1b (открытая степь, сплошная трава), Forestry Canada 1992 (ST-X-3):
  ISI = 0,208 · f(W) · f(F),  f(W) = exp(0,05039·WS),  WS - ветер 10 м, км/ч
  f(F) = 91,9 · exp(−0,1386·m) · (1 + m^5,31 / 4,93e7),  m = 147,2·(101 − FFMC) / (59,5 + FFMC)
  ROS_head = 250 · (1 − exp(−0,035·ISI))^1,7 · CF  (м/мин),   CF - поправка на степень высыхания травы C (%):
  CF = 0,005·(exp(0,061·C) − 1) при C < 58,8, иначе 0,176 + 0,02·(C − 58,8)
  ROS_back - та же формула с f(W) = exp(−0,05039·WS);  вытянутость эллипса LB = 1 + 8,729·(1 − exp(−0,030·WS))^2,155
  склон: множитель exp(3,533 · tan(φ)^1,2) для подъёма по направлению шага (FBP SF).
Форма - эллипс (Ван Вагнер 1969 / Alexander 1985): точка возгорания в фокусе-смещении c = a − ROS_back, a = (ROS_head + ROS_back)/2,
b = a / LB; скорость в направлении θ от головы - расстояние от точки возгорания до эллипса за единицу времени.
Топливо ячейки: трава 1, пашня 0,6, кусты/лес 0,4, болото 0,2; негорючее > 50 % - не горит. Высыхание C - по NDVI (C = 100 − 150·(NDVI − 0,2)).
Распространение - кратчайшее время прихода по 16 направлениям (как в SolverNet), ветер меняется по окнам 0–12, 12–24, 24–48, 48–72 ч.
k_ros - общий множитель скорости (калибруется на val: FBP рассчитан на непрерывную траву днём, в степи Казахстана огонь медленнее).
"""
import math
import numpy as np

OFFS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1),
        (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2)]
DIST = np.array([0.5 * math.hypot(r, c) for r, c in OFFS])                 # км
EV = np.array([[c / math.hypot(r, c), -r / math.hypot(r, c)] for r, c in OFFS])   # (восток, север)
EDGES = (0, 12, 24, 48, 72)
BIG = 1e4
LAT_TS = 30.0                                                             # стандартная параллель EPSG:6933 (сетка датасета)


def geometry(lat=None):
    """Расстояния (км) и направления (восток, север) 16 соседей НА МЕСТНОСТИ. Сетка EPSG:6933 равновеликая: на широте φ ячейка
    «500 м» - это 500·cos φ / cos 30° м по В–З и 500·cos 30° / cos φ м по С–Ю (на 47,5° с. ш. ≈ 390 × 640 м). lat=None - квадрат 500 м."""
    if lat is None or not np.isfinite(lat):
        return DIST, EV
    sx = math.cos(math.radians(lat)) / math.cos(math.radians(LAT_TS)); sy = 1 / sx
    d = np.array([0.5 * math.hypot(r * sy, c * sx) for r, c in OFFS])
    ev = np.array([[c * sx, -r * sy] for r, c in OFFS]) / (d[:, None] / 0.5)
    return d, ev


def isi(ws_kmh, ffmc, back=False):
    m = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)
    return 0.208 * np.exp((-1 if back else 1) * 0.05039 * ws_kmh) * ff


def curing_factor(C):
    C = np.clip(C, 0, 100)
    return np.where(C < 58.8, 0.005 * (np.exp(0.061 * C) - 1), 0.176 + 0.02 * (C - 58.8))


def ros_o1b(isi_):
    return 250 * (1 - np.exp(-0.035 * isi_)) ** 1.7                       # м/мин без поправки на высыхание


def ellipse_speed(head, back, lb, cos_t):
    """Скорость фронта в направлении под углом θ к голове (cos θ), эллипс с точкой возгорания в смещённом фокусе."""
    a = (head + back) / 2; b = a / lb; c = a - back
    sin2 = 1 - cos_t ** 2
    # (r·cosθ − c)^2 / a^2 + r^2·sin^2θ / b^2 = 1 (центр эллипса - на c впереди точки возгорания, направление головы +)
    A = cos_t ** 2 / a ** 2 + sin2 / b ** 2; B = -2 * c * cos_t / a ** 2; Cc = c ** 2 / a ** 2 - 1
    return (-B + np.sqrt(np.maximum(B ** 2 - 4 * A * Cc, 0))) / (2 * A)


def fwi_from(isi_, bui):
    fd = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    bb = 0.1 * isi_ * fd
    return np.where(bb > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1e-9))) ** 0.647), bb)


def fuel_factor(grass, crop, shrub, wet, nonb):
    f = grass + 0.6 * crop + 0.4 * shrub + 0.2 * wet
    return np.where(nonb > 0.5, 0.0, f)


def speeds(fuel, ndvi, wind_uv, ffmc, k_ros, ev=EV):
    """Скорости [16, H, W] км/ч для одного окна погоды. wind_uv - (u, v) м/с, куда дует."""
    u, v = wind_uv; ws = math.hypot(u, v) * 3.6
    cf = curing_factor(100 - 150 * (np.nan_to_num(ndvi, nan=0.3) - 0.2))
    h0 = float(ros_o1b(isi(ws, ffmc))); b0 = float(ros_o1b(isi(ws, ffmc, back=True)))
    head = h0 * cf * 0.06 * k_ros                                         # м/мин -> км/ч
    lb = 1 + 8.729 * (1 - math.exp(-0.030 * ws)) ** 2.155
    hd = np.array([u, v]) / (math.hypot(u, v) + 1e-9) if ws > 0.1 else np.array([1.0, 0.0])
    cos_t = ev @ hd                                                       # [16]
    rel = np.array([ellipse_speed(1.0, b0 / max(h0, 1e-9), lb, c) for c in cos_t])   # форма эллипса не зависит от высыхания
    S = rel[:, None, None] * (head * fuel)[None]
    return S


def arrival(burned, fuel, ndvi, elev, winds, ffmc, k_ros=1.0, iters=400, lat=None):
    """Время прихода огня, ч (0 - уже горит). winds - 4 вектора (u, v) для окон EDGES. Плоский ландшафт при elev=None."""
    H, W = burned.shape
    dist, ev = geometry(lat)
    S = [speeds(fuel, ndvi, w, ffmc, k_ros, ev) for w in winds]
    slope_f = np.ones((16, H, W))
    if elev is not None:
        e = np.pad(elev, 2, mode="edge")
        for k, (r, c) in enumerate(OFFS):
            rise = (e[2 + r:2 + r + H, 2 + c:2 + c + W] - elev) / (dist[k] * 1000)
            slope_f[k] = np.exp(3.533 * np.maximum(rise, 0) ** 1.2)
    T = np.where(burned, 0.0, BIG)
    for _ in range(iters):
        Tp = np.pad(T, 2, constant_values=BIG); new = T.copy()
        wi = np.searchsorted(EDGES[1:-1], T, side="right")                 # окно погоды по времени выхода из ячейки
        for k, (r, c) in enumerate(OFFS):
            # переход из соседа (x − off) в x: скорость по направлению k, среднее гармоническое скоростей двух ячеек
            src_T = Tp[2 - r:2 - r + H, 2 - c:2 - c + W]
            wsrc = np.pad(wi, 2, mode="edge")[2 - r:2 - r + H, 2 - c:2 - c + W]
            Ssrc = np.choose(wsrc, [np.pad(s[k], 2)[2 - r:2 - r + H, 2 - c:2 - c + W] for s in S])
            Sdst = np.choose(wsrc, [s[k] for s in S])
            sf = np.pad(slope_f[k], 2, constant_values=1)[2 - r:2 - r + H, 2 - c:2 - c + W]
            v = 2 / (1 / np.maximum(Ssrc * sf, 1e-6) + 1 / np.maximum(Sdst * sf, 1e-6))
            cand = src_T + dist[k] / v
            new = np.minimum(new, np.where((Ssrc > 0) & (Sdst > 0), cand, BIG))
        if np.allclose(new, T):
            break
        T = new
    return np.minimum(T, BIG)
