"""Демо «известные пожары»: срезы теста 2023–24 с полными входами модели (как в офлайн-оценке и prod/checks.py, шаг 4).
Движок считает тот же prod-гибрид на тех же входах (реальная гарь с возрастом, термоточки VIIRS, точка возгорания, прогноз GFS),
поэтому F1 на сайте = офлайн F1 этих срезов.

Сборка набора (один раз):  ../.venv/Scripts/python.exe -X utf8 -m prod.demo   -> prod/demo_cases.npz, prod/demo_cases.json
"""
import json, os, sys
import numpy as np, pandas as pd

from .config import CH, HZ, ROOT, SCRIPTS_T1, CELL

HERE = os.path.dirname(os.path.abspath(__file__))
F_NPZ, F_JSON = os.path.join(HERE, "demo_cases.npz"), os.path.join(HERE, "demo_cases.json")
N_CASES = 40


def score_mask(Y, valid, lsrc, k):
    """Маска оценки как в офлайн-тесте: на 12 ч не считаем ячейки, датированные только по дню MCD64A1."""
    return valid[0] & (((lsrc[0] != 4) | (Y[0, 0] == 0)) if k == 0 else True)


def counts(mask, Y, valid, lsrc):
    out = []
    for k in range(len(HZ)):
        m = score_mask(Y, valid, lsrc, k); y = (Y[0, k] > 0) & m; p = mask[k] & m
        out.append([int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())])
    return np.array(out)


def f1(c):
    c = np.asarray(c, float); return (2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1)).round(3)


class Demo:
    def __init__(self, engine):
        self.E = engine; z = np.load(F_NPZ, allow_pickle=False)
        self.X, self.Y, self.valid, self.lsrc = z["X"], z["Y"], z["valid"], z["lsrc"]
        self.meta = json.load(open(F_JSON, encoding="utf-8"))
        self.rows = self.meta["cases"]

    def run(self, i):
        c = self.rows[i]; E = self.E
        X = self.X[i:i + 1].astype(np.float32); Y, valid, lsrc = self.Y[i:i + 1], self.valid[i:i + 1], self.lsrc[i:i + 1]
        o = E.predict_scene(X, valid, c["row"])
        r0, c0 = c["r0"], c["c0"]; ib = CH.index("burned_frac"); burned = X[0, ib] > 0.5
        cnt = counts(o["mask"], Y, valid, lsrc)
        hs = np.argwhere(X[0, CH.index("af_count_24h")] > 0)
        lon, lat = E.land.lonlat(r0 + hs[:, 0], c0 + hs[:, 1]) if len(hs) else (np.array([]), np.array([]))
        return dict(id=i, title=c["title"], t_utc=c["t_utc"],
                    start=dict(type="FeatureCollection", features=E.geojson(burned, r0, c0)),
                    hotspots=np.c_[lon, lat].round(5).tolist(),
                    horizons={f"{h}h": dict(pred=dict(type="FeatureCollection", features=E.geojson(o["mask"][k] & ~burned, r0, c0)),
                                            true=dict(type="FeatureCollection", features=E.geojson((Y[0, k] > 0) & ~burned, r0, c0)),
                                            F1=float(f1(cnt[k])), pred_km2=round(float((o["mask"][k] & ~burned).sum()) * (CELL / 1000) ** 2, 2),
                                            true_km2=round(float(((Y[0, k] > 0) & ~burned).sum()) * (CELL / 1000) ** 2, 2))
                              for k, h in enumerate(HZ)})


def build():
    sys.path.insert(0, SCRIPTS_T1)
    from v3_loader import load_chunk
    from .engine import FireEngine
    E = FireEngine()
    ci = pd.read_csv(os.path.join(ROOT, "task2_model", "cache_v3", "test_idx.csv"), low_memory=False)
    ok = (ci.new_72h_km2 >= 5) & (ci.state_km2 >= 1) & (ci.valid_frac > 0.95)
    pick = ci[ok].sample(frac=1, random_state=0).drop_duplicates("fire_id").head(N_CASES).sort_values("t_utc")
    Xs, Ys, Vs, Ls, cases, tot = [], [], [], [], [], np.zeros((len(HZ), 3))
    for _, r in pick.iterrows():
        X, Y, valid, lsrc = load_chunk(r.file, [r.slice]); row = r.to_dict()
        o = E.predict_scene(X, valid, r.to_dict()); cnt = counts(o["mask"], Y, valid, lsrc); tot += cnt
        lon, lat = (float(v) for v in E.land.lonlat(int(r.r0) + 48, int(r.c0) + 48))
        cases.append(dict(title=f"{str(r.t_utc)[:10]} · {lat:.2f}° с. ш., {lon:.2f}° в. д. · гарь {r.state_km2:.0f} км²", fire_id=int(r.fire_id),
                          t_utc=str(r.t_utc), lon=round(lon, 4), lat=round(lat, 4), r0=int(r.r0), c0=int(r.c0),
                          state_km2=float(r.state_km2), F1=f1(cnt).tolist(), row=row))
        Xs.append(X[0]); Ys.append(Y[0]); Vs.append(valid[0]); Ls.append(lsrc[0])
        print(cases[-1]["title"], cases[-1]["F1"], flush=True)
    np.savez_compressed(F_NPZ, X=np.stack(Xs), Y=np.stack(Ys), valid=np.stack(Vs), lsrc=np.stack(Ls))
    meta = dict(pooled_F1=dict(zip([f"{h}h" for h in HZ], f1(tot).tolist())), n=len(cases), cases=cases)
    json.dump(meta, open(F_JSON, "w", encoding="utf-8"), ensure_ascii=False, default=str)
    print("pooled F1", meta["pooled_F1"])


if __name__ == "__main__":
    build()
