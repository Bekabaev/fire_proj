"""Приёмочные проверки прод-движка на сценариях сайта (запуск: python -X utf8 -m prod.checks из project_till_nov).

1) Возгорание в точке на реальном ландшафте (40 точек степи, лето 2024), ветер 8 м/с с С / В / Ю / З:
   площадь > 0, доля прогнозов со смещением по ветру (cos > 0,707; случайно 25 %), средний cos.
2) Сила ветра 0 / 3 / 6 / 10 м/с: медианная площадь растёт, вытянутость по ветру растёт.
3) Негорючие ячейки (вода, застройка > 50 %) не выгорают.
4) Сверка с офлайн-оценкой: движок на сырых срезах теста 2023–24 (прогноз GFS из индекса) - F1 как в task2 шаг 20.
Выход: prod/checks_report.json + печать.
"""
import json, os, sys, time
import numpy as np, pandas as pd

from .config import CH, HZ, ROOT, SCRIPTS_T1
from .engine import FireEngine
from . import scene as sc, weather as wx

sys.path.insert(0, SCRIPTS_T1)
yy, xx = np.mgrid[0:96, 0:96]


def shift_cos(mask, burned, wind_to):
    a = mask & ~burned
    if a.sum() == 0:
        return np.nan
    d = np.array([xx[a].mean() - xx[burned].mean(), -(yy[a].mean() - yy[burned].mean())])
    return float(d @ wind_to / (np.linalg.norm(d) * np.linalg.norm(wind_to) + 1e-9))


def ignition_points(E, n, rng):
    pts = []
    while len(pts) < n:
        lon, lat = rng.uniform(52, 82), rng.uniform(46, 54)
        r, c = E.land.rc(lon, lat)
        if not (50 <= r < 2729 and 50 <= c < 8045):
            continue
        fr = E.land.FR[:, r - 3:r + 4, c - 3:c + 4] / 100
        if E.land.COV[r, c] and fr[0].mean() > 0.7 and fr[4].mean() < 0.05:
            pts.append((lon, lat))
    return pts


def scen(E, lon, lat, t, ws, wd, radius_km=1.0, mode="active"):
    r, c = E.land.rc(lon, lat); r0, c0 = E.land.window_origin(r, c)
    static, cov, _ = E.land.static(r0, c0, t)
    b = sc.disk((96, 96), (r - r0, c - c0), radius_km / 0.5) & cov
    row = wx.scenario_row((ws, wd), T=30.0, RH=20.0)
    X, valid, row = sc.build(static, cov, b, np.full((96, 96), 6.0), t, (r - r0, c - c0), 0, None, row)
    row["lat"] = lat
    return E.predict_scene(X, valid, row, mode), b, X


def main():
    E = FireEngine(); rng = np.random.default_rng(1); t = pd.Timestamp("2024-08-10 06:00"); rep = {"variant": E.variant}
    pts = ignition_points(E, 40, rng); t0 = time.time()
    # 1) направление ветра
    rows = []
    for lon, lat in pts:
        for wd, to in ((0, (0, -1)), (90, (-1, 0)), (180, (0, 1)), (270, (1, 0))):     # «откуда» -> вектор «куда» (восток, север)
            for mode in ("active", "expected"):
                out, b, _ = scen(E, lon, lat, t, 8.0, wd, mode=mode)
                for k, h in enumerate(HZ):
                    rows.append(dict(mode=mode, lon=lon, lat=lat, wd=wd, h=h, area=out["area_km2"][k],
                                     cos=shift_cos(out["mask"][k], b, np.array(to, float)),
                                     cos_unet=shift_cos(out["unet"][k] >= 0.5, b, np.array(to, float))))
    d = pd.DataFrame(rows)
    g = d.groupby(["mode", "h"]).agg(area_med=("area", "median"), grows=("area", lambda a: float((a > 0).mean())), cos=("cos", "mean"),
                           downwind=("cos", lambda a: float((a > 0.707).mean())), cos_unet=("cos_unet", "mean"))
    rep["ignition_wind8"] = g.round(3).reset_index().to_dict("records"); print("1) ignition, wind 8 m/s\n", g.round(3), flush=True)
    # 2) сила ветра
    rows = []
    for lon, lat in pts[:20]:
        for ws in (0.0, 3.0, 6.0, 10.0):
            out, b, _ = scen(E, lon, lat, t, ws, 270)
            rows.append(dict(ws=ws, area72=out["area_km2"][3], area24=out["area_km2"][1],
                             cos72=shift_cos(out["mask"][3], b, np.array((1.0, 0.0)))))
    d2 = pd.DataFrame(rows).groupby("ws").median().round(3)
    rep["wind_speed"] = d2.reset_index().to_dict("records"); print("2) wind speed (west wind)\n", d2, flush=True)
    # 3) негорючее
    burnt_nonb = []
    for lon, lat in pts[:20]:
        out, b, X = scen(E, lon, lat, t, 8.0, 270)
        nb = X[0, CH.index("nonburnable_frac")] > 0.5
        burnt_nonb.append(float((out["mask"][3] & nb).sum() / max(nb.sum(), 1)))
    rep["nonburnable_burnt_share"] = float(np.mean(burnt_nonb)); print("3) nonburnable burnt share", rep["nonburnable_burnt_share"], flush=True)
    # 4) сверка с офлайн-оценкой на сырых срезах теста
    from v3_loader import load_chunk
    ci = pd.read_csv(os.path.join(ROOT, "task2_model", "cache_v3", "test_idx.csv"), low_memory=False)
    OUTD = os.path.join(ROOT, "task2_model", "out_v3")
    Pof = np.load(os.path.join(OUTD, "probs_wx_gfsa_test.npy"), mmap_mode="r"); Tof = np.load(os.path.join(OUTD, "phys_T_test_gfsa.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(ci), 400, replace=False))
    st = {k: np.zeros((3, 4)) for k in ("engine", "offline", "engine_active_on_growing")}; H = E.heads["expected"]
    for i in idx:
        r = ci.iloc[i]; X, Y, valid, lsrc = load_chunk(r.file, [r.slice]); row = r.to_dict()
        o = E.predict_scene(X, valid, row); oa = E.predict_scene(X, valid, row, "active")
        t = np.asarray(Tof[i], np.float32); far = t >= 1000; lt = np.log(np.clip(np.where(far, 1000, t), 0.5, 1000)[None] / np.array(HZ, float)[:, None, None])
        pu = np.clip(Pof[i].astype(np.float32) / 255, 0.5 / 255, 254.5 / 255); lp = np.log(pu / (1 - pu))
        for k, h in enumerate(HZ):
            m = valid[0] & (((lsrc[0] != 4) | (Y[0, 0] == 0)) if k == 0 else True); y = (Y[0, k] > 0) & m
            W = H["W"][k]; z = W[0] * lp[k] + W[1] * lt[k] + W[2] * far + W[3] * lt[k] * float(row[f"fc_gfsa_ws_mean_{h}h"]) + H["b"][k]
            pof = (np.round(255 / (1 + np.exp(-z))) >= H["level"][k]) & m
            st["engine"][:, k] += [(o["mask"][k] & m & y).sum(), (o["mask"][k] & m & ~y).sum(), (~o["mask"][k] & y).sum()]
            st["offline"][:, k] += [(pof & y).sum(), (pof & ~y).sum(), (~pof & y).sum()]
            if r[f"new_{h}h_km2"] >= 1:
                pa = oa["mask"][k] & m
                st["engine_active_on_growing"][:, k] += [(pa & y).sum(), (pa & ~y).sum(), (~pa & y).sum()]
    for n, v in st.items():
        f1 = (2 * v[0] / np.maximum(2 * v[0] + v[1] + v[2], 1)).round(3).tolist(); rep[f"test400_F1_{n}"] = dict(zip([f"{h}h" for h in HZ], f1))
        print(f"4) {n}: F1", f1, flush=True)
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(ROOT, "prod", "checks_report.json"), "w"), indent=1, ensure_ascii=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
