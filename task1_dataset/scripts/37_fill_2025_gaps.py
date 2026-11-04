"""Шаг 37 (прод, 04.11.2026). Заполнение пробелов MCD64A1 2025 (нет в каталоге Planetary Computer: май, август–октябрь 2025 -
сбой загрузки всех MODIS-продуктов каталога за июль–октябрь 2025) гарью по термоточкам FIRMS (архив SP, шаг 10, ONLY_YEARS=2025).

Правило «ячейка 500 м выгорела в день первой термоточки» подбирается на 2024 (там есть и FIRMS, и GFA) для тех же месяцев:
варианты - VIIRS (S-NPP + NOAA-20) / VIIRS + MODIS, минимальный размер кластера термоточек (ячеек, 8-связность после расширения на 1),
морфологическое закрытие 3 × 3; критерий - IoU с GFA при отношении площадей 0,7–1,5. Одиночные термоточки (палы на полях, факелы)
GFA гарью не считает: «все термоточки» дают IoU 0,08 и площадь ×2,7, кластеры ≥ 20 ячеек с закрытием - IoU 0,17, площадь ×1,1.
Ячейки, уже выгоревшие по MCD64A1 в доступные месяцы, не меняются; чистый MCD64A1 сохраняется в doy_2025_mcd64.npy.
Выход: doy_2025.npy (дополнен), mcd64_months_2025.json (+ месяцы из FIRMS), qa_firms_fill_2024.csv; затем перезапустить
`36_update_2025_26.py fuel` (last_burn_2026 зависит от гарей 2025).
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.ndimage import binary_dilation
from rasterio.warp import transform as warp_transform
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

FD = os.path.join(DATA, "ext_cache", "firms_kz")
GAP = {5, 8, 9, 10}                                            # месяцы 2025 без MCD64A1 в каталоге


def hotspots(y):
    a = pd.read_csv(os.path.join(FD, f"firms_kz_{y}.csv.gz"), parse_dates=["time_utc"])
    a = a[a["type"].fillna(0) == 0]                            # 0 - предполагаемый пожар растительности
    x, yv = warp_transform("EPSG:4326", CRS, a.lon.tolist(), a.lat.tolist())
    a["r"] = ((Y1 - np.array(yv)) // CELL).astype(int); a["c"] = ((np.array(x) - X0) // CELL).astype(int)
    a = a[(a.r >= 0) & (a.r < GH) & (a.c >= 0) & (a.c < GW)].copy()
    a["doy"] = a.time_utc.dt.dayofyear.astype(np.int16); a["month"] = a.time_utc.dt.month
    return a


def clusters(p0, k, close):
    """Оставить кластеры ≥ k ячеек (8-связность после расширения на 1); close - закрытие 3 × 3 (сплошная полоса гари)."""
    from scipy import ndimage
    lab, n = ndimage.label(ndimage.binary_dilation(p0, np.ones((3, 3), bool)), np.ones((3, 3), bool))
    size = np.bincount(lab[p0], minlength=n + 1); keep = np.isin(lab, np.nonzero(size >= k)[0]) & (lab > 0) & p0
    return (ndimage.binary_closing(keep, np.ones((3, 3), bool)) | keep) if close else keep


def raster(a, min_n=1, dilate=False):
    """Первая дата термоточки в ячейке (int16 doy, 0 - нет); min_n - минимум термоточек в ячейке."""
    g = a.groupby(["r", "c"]).agg(doy=("doy", "min"), n=("doy", "size")).reset_index()
    g = g[g.n >= min_n]; out = np.zeros((GH, GW), np.int16); out[g.r.values, g.c.values] = g.doy.values
    if dilate:
        m = binary_dilation(out > 0, np.ones((3, 3), bool)); add = m & (out == 0)
        from scipy.ndimage import grey_dilation
        big = np.where(out > 0, out, 999).astype(np.int16)
        out[add] = -grey_dilation(-big, size=3)[add]           # ближайшая (минимальная) дата соседей
    return out


if __name__ == "__main__":
    # --- подбор правила на 2024 для месяцев-пробелов
    g24 = np.asarray(load_doy(2024)); mon24 = np.zeros_like(g24, np.int8)
    ok = g24 > 0; mon24[ok] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(g24[ok] - 1, "D")).month
    truth = np.isin(mon24, list(GAP))
    a24 = hotspots(2024); a24 = a24[a24.month.isin(GAP)]
    rows = []
    for srcs, name in ((("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"), "viirs"), (("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"), "viirs+modis")):
        p0 = raster(a24[a24.source.isin(srcs)]) > 0
        for k in (1, 5, 10, 20, 40):
            for close in (False, True):
                p = clusters(p0, k, close)
                inter = (p & truth).sum(); uni = (p | truth).sum()
                rows.append(dict(src=name, min_cluster=k, close=close, IoU=inter / max(uni, 1), recall=inter / max(truth.sum(), 1),
                                 precision=inter / max(p.sum(), 1), area_ratio=p.sum() / max(truth.sum(), 1)))
    q = pd.DataFrame(rows).sort_values("IoU", ascending=False); q.round(3).to_csv(os.path.join(KZ, "qa_firms_fill_2024.csv"), index=False)
    print(q.round(3).to_string(index=False), flush=True)
    best = q[q.area_ratio.between(0.7, 1.5)].iloc[0]; srcs = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP") + (("MODIS_SP",) if best.src == "viirs+modis" else ())
    print("rule:", best.src, "min_cluster", best.min_cluster, "close", best.close, flush=True)
    # --- заполнение 2025 (от чистого MCD64A1)
    fb = os.path.join(KZ, "doy_2025_mcd64.npy")
    if not os.path.exists(fb):
        np.save(fb, np.load(doy_path(2025)))
    d25 = np.load(fb); a25 = hotspots(2025); a25 = a25[a25.month.isin(GAP) & a25.source.isin(srcs)]
    r = raster(a25); keep = clusters(r > 0, int(best.min_cluster), bool(best.close))
    from scipy.ndimage import grey_dilation
    big = np.where(r > 0, r, 999).astype(np.int16); near = -grey_dilation(-big, size=3)     # дата ячеек, добавленных закрытием - ближайшая
    f = np.where(keep, np.where(r > 0, r, near), 0).astype(np.int16); f[f >= 999] = 0
    add = (f > 0) & (d25 == 0)
    d25[add] = f[add]; np.save(doy_path(2025), d25)
    fm = os.path.join(KZ, "mcd64_months_2025.json"); meta = json.load(open(fm))
    meta.update(months=sorted(set(meta["months"]) | {f"2025-{m:02d}" for m in GAP}),
                filled_from_firms=sorted(f"2025-{m:02d}" for m in GAP),
                firms_rule=dict(src=best.src, min_cluster=int(best.min_cluster), close=bool(best.close), IoU_2024=round(float(best.IoU), 3),
                                area_ratio_2024=round(float(best.area_ratio), 3)))
    json.dump(meta, open(fm, "w"))
    print("2025 filled cells", int(add.sum()), "km²", float(add.sum() * 0.25), "| total burned km²", float((d25 > 0).sum() * 0.25), flush=True)
    print("done", flush=True)
