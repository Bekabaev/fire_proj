"""Картинки для слайдов о данных и моделях -> presentation/site_shots/*.png (+ svg графика обучения в train_curve.svg).

  ../../.venv/Scripts/python.exe -X utf8 -W ignore tools/make_figs.py     (из папки presentation)

1. Эксперт и спутник: пожар Алтын-Дала 30.06.2019 (куб 30 м): снимок Sentinel-2 после пожара (NBR) + контур эксперта,
   суточная метка MCD64A1, время прихода огня по термоточкам (наш датасет).
2. Один срез теста (демо-случай 4, пожар 3.04.2023): входные карты, ответ (новая гарь 12/24/48/72 ч),
   физика (время прихода T), U-Net (вероятность), гибрид — против факта.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # presentation/
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
OUT = os.path.join(HERE, "site_shots")
CASE = 4

INK, PAPER, ASH, RED, BLUE, OVER = "#1B1A18", "#F2EFE9", "#3A3A3A", "#E4572E", "#2E86AB", "#B07566"
HOURS = LinearSegmentedColormap.from_list("hours", ["#7A1E05", "#C2501E", "#E2622B", "#F2A541", "#F7D98C", "#FBF1CF"])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100, facecolor=fig.get_facecolor()); plt.close(fig); print(name)


def panel(a, w=6.0, h=6.0, bg=PAPER):
    """Картинка без осей, ровно по массиву."""
    fig = plt.figure(figsize=(w, h), facecolor=bg); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_facecolor(bg)
    return fig, ax


# ---------------- 1. эксперт и спутник ----------------
def expert():
    ev = "AltynDala_2019_06_30"; D = os.path.join(PROJ, "task1_dataset", "data")
    cube = np.load(os.path.join(D, "cubes", ev + ".npz"), allow_pickle=True)
    ch = json.load(open(os.path.join(D, "cubes", ev + ".json")))["channels"]
    arr = np.load(os.path.join(D, "arrival", ev + ".npz"))
    s = 2                                                        # 30 м -> 60 м для картинки
    lab = cube["label"][::s, ::s] > 0
    nbr = cube["x"][ch.index("post_NBR")][::s, ::s].astype(np.float32)
    a_h = arr["arrival_h"][::s, ::s]; m_h = arr["mcd64_h"][::s, ::s]
    H, W = lab.shape; asp = W / H
    from scipy import ndimage
    edge = lab & ~ndimage.binary_erosion(lab, iterations=2)
    # а) снимок после пожара: NBR в оттенках серого (гарь тёмная) + контур эксперта
    fig, ax = panel(None, 6 * asp, 6)
    lo, hi = np.nanpercentile(nbr, [2, 98]); g = np.clip((nbr - lo) / (hi - lo), 0, 1)
    ax.imshow(np.nan_to_num(g, nan=0.85), cmap="gray", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap(["#F2A541"]), interpolation="nearest")
    save(fig, "exp_image.png")
    vmax = float(np.nanpercentile(a_h[lab], 99))
    for name, v, miss in (("exp_mcd64.png", m_h, True), ("exp_ours.png", a_h, False)):
        fig, ax = panel(None, 6 * asp, 6)
        base = np.where(lab, 1, 0)
        ax.imshow(np.ma.masked_where(base == 0, base), cmap=ListedColormap(["#D9D2C5"]), interpolation="nearest")  # контур эксперта: серый
        vv = np.where(lab & np.isfinite(v), v, np.nan)
        ax.imshow(np.ma.masked_invalid(vv), cmap=HOURS, vmin=0, vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap([INK]), interpolation="nearest")
        save(fig, name)
    share = float((np.isfinite(m_h) & lab).sum() / lab.sum())
    # шкала часов
    fig = plt.figure(figsize=(8, 0.5), facecolor=PAPER); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.linspace(0, 1, 512)[None], cmap=HOURS, aspect="auto"); save(fig, "exp_scale.png")
    return dict(mcd_share=round(share, 2), vmax_h=round(vmax))


# ---------------- 2. один срез теста ----------------
def case():
    from prod.config import CH, HZ
    from prod.engine import FireEngine
    from prod.demo import Demo, counts, f1
    E = FireEngine(); Dm = Demo(E); i = CASE
    X = Dm.X[i:i + 1].astype(np.float32); Y = Dm.Y[i:i + 1]; v = Dm.valid[i:i + 1]; l = Dm.lsrc[i:i + 1]; row = Dm.rows[i]["row"]
    o = E.predict_scene(X, v, row)
    burned = X[0, CH.index("burned_frac")] > 0.5
    y = Y[0] > 0
    T = o["arrival_h"]; U = o["unet"]; Hm = o["mask"]
    phys_m = np.stack([T <= h for h in HZ]) & ~burned
    unet_m = (U >= 0.5) & ~burned[None]
    F = dict(hybrid=f1(counts(Hm, Y, v, l)).tolist(), unet=f1(counts(unet_m, Y, v, l)).tolist(), phys=f1(counts(phys_m, Y, v, l)).tolist())
    # окно картинки: квадрат вокруг гари, факта и прогнозов
    any_ = burned | y.any(0) | Hm[3] | unet_m[3] | phys_m[3]
    rr, cc = np.nonzero(any_); pad = 6
    r0, r1, c0, c1 = rr.min() - pad, rr.max() + pad, cc.min() - pad, cc.max() + pad
    n = max(r1 - r0, c1 - c0); rc, cm = (r0 + r1) // 2, (c0 + c1) // 2
    r0, c0 = max(0, rc - n // 2), max(0, cm - n // 2); r0, c0 = min(r0, 96 - n), min(c0, 96 - n)
    win = (slice(r0, r0 + n), slice(c0, c0 + n))
    # входные карты
    hot = LinearSegmentedColormap.from_list("hot", ["#F2A541", "#C2501E", "#7A1E05"])
    maps = [("burned_frac", ListedColormap(["#FFFDF8", ASH]), "гарь сейчас"), ("burn_age_h", hot, "возраст гари, ч"),
            ("af_count_24h", hot, "термоточки за 24 ч"), ("grass_frac", "YlGn", "доля травы"),
            ("ndvi", "Greens", "NDVI"), ("elev", "terrain", "высота"), ("prior_burn", "Purples", "прошлые гари"),
            ("dist_road_km", "Blues_r", "расстояние до дорог")]
    for k, cmap, _ in maps:
        a = X[0, CH.index(k)][win]
        fig, ax = panel(None, 2.4, 2.4)
        if k in ("af_count_24h", "burn_age_h"):
            a = np.ma.masked_where(a <= 0, a); ax.imshow(np.zeros_like(a.data), cmap=ListedColormap(["#FFFDF8"]))
        ax.imshow(a, cmap=cmap, interpolation="nearest"); save(fig, f"in_{k}.png")
    # ответ: первый горизонт, к которому клетка сгорела
    first = np.full(burned.shape, -1)
    for k in range(len(HZ))[::-1]:
        first[y[k]] = k
    cols = ["#7A1E05", "#C2501E", "#F2A541", "#F7D98C"]
    fig, ax = panel(None, 4, 4, "#FFFDF8")
    ax.imshow(np.ma.masked_where(first[win] < 0, first[win]), cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~burned[win], burned[win]), cmap=ListedColormap([ASH]), interpolation="nearest")
    save(fig, "target.png")

    def vs(m, name):                                               # как на сайте: факт красный, прогноз синий, совпало - бурый
        img = np.zeros(burned.shape + (3,)); img[:] = matplotlib.colors.to_rgb("#FFFDF8")
        for mask, c in ((y[3] & ~m, RED), (m & ~y[3], "#A9CCE0"), (m & y[3], OVER), (burned, ASH)):
            img[mask] = matplotlib.colors.to_rgb(c)
        fig, ax = panel(None, 4, 4, "#FFFDF8"); ax.imshow(img[win], interpolation="nearest"); save(fig, name)
    vs(phys_m[3], "cmp_phys.png"); vs(unet_m[3], "cmp_unet.png"); vs(Hm[3] & ~burned, "cmp_hybrid.png")
    # физика: карта времени прихода T
    fig, ax = panel(None, 5, 5, "#FFFDF8")
    from scipy import ndimage
    t = np.where(burned | (T >= 96), np.nan, T)                 # дальше 96 ч - прозрачно
    ax.imshow(np.ma.masked_invalid(t)[win], cmap=HOURS, vmin=0, vmax=96, interpolation="nearest")
    tt = np.where(burned, 0, np.minimum(T, 200))[win]
    ax.contour(ndimage.zoom(tt, 4, order=1), levels=[24, 72], colors=[INK], linewidths=[1.2, 2.2],
               extent=(-0.5, n - 0.5, n - 0.5, -0.5), origin="upper")
    ax.imshow(np.ma.masked_where(~burned[win], burned[win]), cmap=ListedColormap([ASH]), interpolation="nearest")
    truth = ndimage.binary_closing(y[3] | burned, iterations=2)[win].astype(float)
    ax.contour(ndimage.zoom(truth, 4, order=1), levels=[0.5], colors=[BLUE], linewidths=3, linestyles="--",
               extent=(-0.5, n - 0.5, n - 0.5, -0.5), origin="upper")
    save(fig, "phys_T.png")
    # U-Net: вероятность сгореть за 72 ч
    fig, ax = panel(None, 5, 5, "#FFFDF8")
    p = np.where(burned, np.nan, U[3])
    ax.imshow(np.ma.masked_where(~(p >= 0.05), p)[win], cmap=LinearSegmentedColormap.from_list("p", ["#E3EEF4", "#2E86AB", "#123A52"]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.contour(ndimage.zoom(np.nan_to_num(p)[win], 4, order=1), levels=[0.5], colors=[INK], linewidths=1.5,
               extent=(-0.5, n - 0.5, n - 0.5, -0.5), origin="upper")
    ax.imshow(np.ma.masked_where(~burned[win], burned[win]), cmap=ListedColormap([ASH]), interpolation="nearest")
    ax.contour(ndimage.zoom(truth, 4, order=1), levels=[0.5], colors=[RED], linewidths=3, linestyles="--",
               extent=(-0.5, n - 0.5, n - 0.5, -0.5), origin="upper")
    save(fig, "unet_P.png")
    # погода и состояние для таблицы
    u, vv = row["fc_gfsa_wind_u_24h"], row["fc_gfsa_wind_v_24h"]
    info = dict(F1=F, win_km=n * 0.5, wind_from_deg=round(float(np.degrees(np.arctan2(u, vv)) % 360)),
                ws=round(row["fc_gfsa_ws_mean_24h"], 1), gust=round(row["fc_gfsa_gust_max_24h"], 1), T=round(row["fc_gfsa_T_max_24h"], 1),
                RH=round(row["fc_gfsa_RH_min_24h"]), P=round(row["fc_gfsa_P_sum_24h"], 1), FFMC=round(row["gfwed_FFMC"]), FWI=round(row["gfwed_FWI"]),
                state_km2=row["state_km2"], k_12h=row["k_12h"], t_utc=row["t_utc"],
                true_km2=[float(y[k].sum()) * 0.25 for k in range(4)], hyb_km2=[float((Hm[k] & ~burned).sum()) * 0.25 for k in range(4)],
                phys_km2=[float(phys_m[k].sum()) * 0.25 for k in range(4)], unet_km2=[float(unet_m[k].sum()) * 0.25 for k in range(4)])
    return info


# ---------------- 3. график обучения (svg для слайда) ----------------
def curve():
    import pandas as pd
    d = pd.read_csv(os.path.join(PROJ, "task2_model", "out_v3", "wx_log.csv"))
    W, H, L, B, R, T = 680, 380, 70, 44, 20, 24
    x = lambda e: L + (W - L - R) * e / (len(d) - 1)
    f1 = d.val_F1_mean.values; lo = d.loss.values
    yF = lambda v: T + (H - T - B) * (1 - (v - 0.1) / 0.25)                        # шкала F1 0,10 … 0,35
    yL = lambda v: T + (H - T - B) * (1 - (v - lo.min() * 0.95) / (lo.max() - lo.min() * 0.95))
    pF = " ".join(f"{x(e):.1f},{yF(v):.1f}" for e, v in enumerate(f1)); pL = " ".join(f"{x(e):.1f},{yL(v):.1f}" for e, v in enumerate(lo))
    eb = int(np.argmax(f1))
    g = "".join(f'<line x1="{L}" x2="{W - R}" y1="{yF(t):.1f}" y2="{yF(t):.1f}" stroke="#E4DCCF" stroke-width="1"/>'
                f'<text x="{L - 10}" y="{yF(t) + 7:.1f}" font-size="20" fill="#8C8377" text-anchor="end">' + f"{t:.2f}".replace(".", ",") + "</text>"
                for t in (0.15, 0.2, 0.25, 0.3))
    ticks = "".join(f'<text x="{x(e):.1f}" y="{H - 14}" font-size="20" fill="#8C8377" text-anchor="middle">{e + 1}</text>' for e in (0, 5, 11, 17, 23))
    svg = (f'<svg aria-label="Обучение U-Net: ошибка падает, F1 на проверочном 2022 году растёт и выходит на плато к 17-й эпохе" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">{g}{ticks}'
           f'<polyline points="{pL}" fill="none" stroke="#CFC6B8" stroke-width="4"/>'
           f'<polyline points="{pF}" fill="none" stroke="#E2622B" stroke-width="5"/>'
           f'<circle cx="{x(eb):.1f}" cy="{yF(f1[eb]):.1f}" r="9" fill="#E2622B"/></svg>')
    open(os.path.join(OUT, "train_curve.svg"), "w", encoding="utf-8").write(svg)
    return dict(best_epoch=eb + 1, best_f1=round(float(f1[eb]), 3), f1_ep1=round(float(f1[0]), 3), epochs=len(d), sec=round(float(d.sec.mean())))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    info = dict(expert=expert(), case=case(), curve=curve())
    json.dump(info, open(os.path.join(OUT, "figs_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(info, ensure_ascii=False, default=float))
