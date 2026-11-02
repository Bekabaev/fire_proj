"""Шаг 29 (датасет v3). Настоящие прогнозы GFS для срезов 2022–2024 (val + test) из архива NOAA на AWS - проверка «на прогнозе»
не только на 2024 (шаг 21, Open-Meteo Previous Runs), но и на валидационном 2022 и тестовом 2023.

Источник: s3://noaa-gfs-bdp-pds (без ключа, с 2021 г.), файлы gfs.tCCz.pgrb2.0p50.fLLL; по .idx скачиваются только нужные поля
(диапазоны байт): TMP 2 м, RH 2 м, UGRD/VGRD 10 м, GUST, APCP (накопление с начала прогона).
Для среза t берётся прогон t − 12 ч (00/12 UTC; выпускается через ~4 ч, т. е. доступен на момент t), заблаговременности 12…84 ч
через 6 ч -> окна [t, t + 12/24/48/72 ч] с отсчётами каждые 6 ч (T max и RH min по 6-часовым отсчётам грубее, чем по часовым ERA5).
Признаки fc_gfsa_{T_max,RH_min,ws_mean,gust_max,wind_u,wind_v,P_sum}_{H}h в ближайшей ячейке 0,5°.
Выход: data/ext_cache/gfs_aws/<YYYYmmddHH>.npz (вырезка по Казахстану [13 сроков, 6 полей, h, w] f16), колонки fc_gfsa_*
в slices_v3_index.csv и slices_v3_reserves_index.csv, data/ext_kz/qa_weather_gfsa.csv (против станций).
"""
import os, sys, io, time
import numpy as np, pandas as pd, requests, rasterio
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *
from kz_common import *

OD = os.path.join(CACHE, "gfs_aws"); os.makedirs(OD, exist_ok=True)
URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p50.f{l:03d}"
LEADS = list(range(12, 85, 6)); FIELDS = ["TMP", "RH", "UGRD", "VGRD", "GUST", "APCP"]
WANT = {"TMP:2 m above ground": "TMP", "RH:2 m above ground": "RH", "UGRD:10 m above ground": "UGRD",
        "VGRD:10 m above ground": "VGRD", "GUST:surface": "GUST"}
LON0, LON1, LAT0, LAT1 = 45.0, 89.0, 39.0, 57.0
HZ = (12, 24, 48, 72); YEARS_FC = (2022, 2023, 2024)
IDX = [os.path.join(KZ, "slices_v3_index.csv"), os.path.join(KZ, "slices_v3_reserves_index.csv")]
S = requests.Session(); S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def get(url, **kw):
    for k in range(6):
        try:
            r = S.get(url, timeout=120, **kw)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status(); return r
        except Exception as e:
            time.sleep(5 * (k + 1)); err = e
    print("  fail", url[-40:], str(err)[:80], flush=True)
    return None


def lead(c, l, tmp):
    """Одна заблаговременность: поля (6, h, w) по Казахстану или None."""
    u = URL.format(d=c, l=l); r = get(u + ".idx")
    if r is None:
        return None
    L = [x.split(":") for x in r.text.splitlines()]
    parts = {}
    for i, x in enumerate(L):
        k = x[3] + ":" + x[4]; name = WANT.get(k)
        if k == "APCP:surface" and x[5].startswith("0-"):
            name = "APCP"
        if name and name not in parts:
            end = int(L[i + 1][1]) - 1 if i + 1 < len(L) else ""
            b = get(u, headers={"Range": f"bytes={x[1]}-{end}"})
            if b is None:
                return None
            parts[name] = b.content
    f = os.path.join(tmp, f"{c:%Y%m%d%H}_{l}.grib2"); out = []
    for name in FIELDS:
        if name not in parts:
            out.append(None); continue
        open(f, "wb").write(parts[name])
        with rasterio.open(f) as s:
            a = s.read(1).astype(np.float32); tr = s.transform
            lon = tr.c + (np.arange(s.width) + 0.5) * tr.a; lat = tr.f + (np.arange(s.height) + 0.5) * tr.e
            ci = (lon >= LON0) & (lon <= LON1); ri = (lat >= LAT0) & (lat <= LAT1)
            out.append(a[np.ix_(ri, ci)])
    os.remove(f)
    shp = next(o.shape for o in out if o is not None)
    return np.stack([o if o is not None else np.full(shp, np.nan, np.float32) for o in out]), lat[ri], lon[ci]


def cycle(c):
    fo = os.path.join(OD, f"{c:%Y%m%d%H}.npz")
    if os.path.exists(fo):
        return c, True
    tmp = os.path.join(OD, "tmp"); os.makedirs(tmp, exist_ok=True)
    with ThreadPoolExecutor(13) as ex:
        res = list(ex.map(lambda l: lead(c, l, tmp), LEADS))
    if all(r is None for r in res):
        return c, False
    ref = next(r for r in res if r is not None)
    data = np.stack([r[0] if r is not None else np.full(ref[0].shape, np.nan, np.float32) for r in res]).astype(np.float16)
    np.savez_compressed(fo, data=data, leads=np.array(LEADS), fields=np.array(FIELDS), lat=ref[1], lon=ref[2])
    return c, True


tabs = [pd.read_csv(f, parse_dates=["t_utc"], low_memory=False) for f in IDX]
cyc = pd.concat([t.loc[t.year.isin(YEARS_FC), "t_utc"] for t in tabs]) - pd.Timedelta(hours=12)
cyc = sorted(cyc.drop_duplicates())
print("cycles needed:", len(cyc), flush=True)
t0 = time.time(); bad = []
with ThreadPoolExecutor(3) as ex:
    for i, (c, ok) in enumerate(ex.map(cycle, cyc)):
        if not ok:
            bad.append(c)
        if i % 50 == 0:
            print(f"  {i}/{len(cyc)}  {time.time() - t0:.0f} s, missing {len(bad)}", flush=True)
print("missing cycles:", len(bad), [str(b) for b in bad[:10]], flush=True)


def feats(z, la, lo):
    lat, lon = z["lat"], z["lon"]; i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
    d = z["data"][:, :, i, j].astype(np.float32)           # [lead, field]
    o = {}
    T, RH, U, V, G, P = (d[:, k] for k in range(6))
    for H in HZ:
        m = np.array(LEADS) <= 12 + H                      # отсчёты с временем действия в [t, t + H]
        if np.isnan(U[m]).mean() > 0.2:
            continue
        ws = np.hypot(U[m], V[m])
        # ветер «откуда» как в ERA5/Open-Meteo: u, v - средние компоненты вектора направления «откуда» × скорость
        o.update({f"fc_gfsa_T_max_{H}h": np.nanmax(T[m]), f"fc_gfsa_RH_min_{H}h": np.nanmin(RH[m]), f"fc_gfsa_ws_mean_{H}h": np.nanmean(ws),
                  f"fc_gfsa_gust_max_{H}h": np.nanmax(G[m]), f"fc_gfsa_wind_u_{H}h": float(np.nanmean(-U[m])),
                  f"fc_gfsa_wind_v_{H}h": float(np.nanmean(-V[m])),
                  f"fc_gfsa_P_sum_{H}h": float(max(P[LEADS.index(12 + H)] - P[0], 0)) if np.isfinite(P[[0, LEADS.index(12 + H)]]).all() else np.nan})
    return o


for f in IDX:      # индекс перечитывается: за время скачивания его мог дополнить шаг 28
    t = pd.read_csv(f, parse_dates=["t_utc"], low_memory=False)
    t = t.drop(columns=[c for c in t if c.startswith("fc_gfsa_")] + (["forecast_gfsa_ok"] if "forecast_gfsa_ok" in t else []))
    rows, cache = [], {}
    for r in t.itertuples():
        o = {}
        if r.year in YEARS_FC:
            c = r.t_utc - pd.Timedelta(hours=12); fo = os.path.join(OD, f"{c:%Y%m%d%H}.npz")
            if c not in cache:
                cache = {c: dict(np.load(fo)) if os.path.exists(fo) else None}
            if cache[c] is not None:
                o = feats(cache[c], r.lat, r.lon)
        rows.append(o)
    t = pd.concat([t, pd.DataFrame(rows, index=t.index)], axis=1)
    t["forecast_gfsa_ok"] = t.get("fc_gfsa_ws_mean_24h", pd.Series(np.nan, index=t.index)).notna()
    t.to_csv(f, index=False)
    print(os.path.basename(f), "| forecast GFS-AWS by year:", t[t.year.isin(YEARS_FC)].groupby("year").forecast_gfsa_ok.mean().round(3).to_dict(), flush=True)
    if "reserves" in f:
        continue
    rows = []
    for src, fmt in [("era5", "era5_{}_24h"), ("gfs_aws", "fc_gfsa_{}_24h"), ("gfs_openmeteo", "fc_gfs_{}_24h")]:
        for yrs in ((2022, 2023), (2024,)):
            for v in ("ws_mean", "T_max", "RH_min"):
                a, b = fmt.format(v), f"st_fut_{v}_24h"
                if a not in t or b not in t:
                    continue
                m = t.loc[t.year.isin(yrs) & t.forecast_gfsa_ok, [a, b]].dropna()
                if len(m) >= 30:
                    rows.append(dict(source=src, years="-".join(map(str, yrs)), var=v, n=len(m), bias=round(float((m[a] - m[b]).mean()), 2),
                                     mae=round(float((m[a] - m[b]).abs().mean()), 2), r=round(float(m[a].corr(m[b])), 3)))
    q = pd.DataFrame(rows); q.to_csv(os.path.join(KZ, "qa_weather_gfsa.csv"), index=False)
    print("\nvs stations (24 h after t):\n", q.to_string(), flush=True)
