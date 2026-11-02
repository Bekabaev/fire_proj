"""Шаг 3. Растровые кубы событий на единой сетке 30 м (UTM зоны полигона), окно = bbox полигона + 3 км.

Каналы (float, NaN = нет данных):
  топливо до пожара (GEE, медиана 45 сут до пожара, только где композит годен): pre_B2..pre_B12, pre_NDVI, pre_NDMI, pre_NBR
  рельеф (Copernicus DEM GLO-90, билинейно на 30 м): elev, slope_deg, aspect_sin, aspect_cos
  дистанция до точки возгорания: dist_ign_km (км; float16 не вмещает метры > 65 км)
  прошлые гари из инвентаря заповедника: prior_burn_days (сутки с предыдущего пожара в пикселе, NaN = не горел)
  после пожара - только для контроля разметки, НЕ вход модели: post_NBR, dNBR
Отдельно: landcover (ESA WorldCover 2021, uint8, мода), label (контур события), ignore (другие пожары того же
периода: дата в [дата пожара, дата снимка]), ign_rc (пиксель возгорания).
Выход: data/cubes/<event>.npz, data/cubes_index.csv
"""
import os, sys, json, math
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
sys.path.insert(0, os.path.dirname(__file__))
from ds_common import *

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
# Copernicus DEM GLO-90 (тайлы скачаны в data/ext_cache/dem90; 30-м бакет AWS отсюда слишком медленный, степь плоская)
DEM_URL = os.path.join(DATA, "ext_cache", "dem90", "Copernicus_DSM_COG_30_N{la:02d}_00_E{lo:03d}_00_DEM.tif")
WC_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N{la:02d}E{lo:03d}_Map.tif"
CH = ["pre_B2", "pre_B3", "pre_B4", "pre_B8A", "pre_B11", "pre_B12", "pre_NDVI", "pre_NDMI", "pre_NBR",
      "elev", "slope_deg", "aspect_sin", "aspect_cos", "dist_ign_km", "prior_burn_days", "post_NBR", "dNBR"]

ev = load_events(); gm = pd.read_csv(GEE_MAP).set_index("event")
allpoly = {e.event: read_poly(e.site, e.event) for e in ev.itertuples()}


def warp_into(src_path, dst_shape, tr, crs, bands, resampling, dst_dtype=np.float32, nodata=np.nan, window_ll=None):
    out = np.full((len(bands),) + dst_shape, nodata, dst_dtype)
    with rasterio.open(src_path) as s:
        if window_ll is not None:   # читаем только нужное окно удалённого COG
            b = transform_bounds(crs, s.crs, *window_ll, densify_pts=21)
            w = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
            w = w.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
            src = s.read(bands, window=w); stf = s.window_transform(w)
        else:
            src = s.read(bands); stf = s.transform
        src_nd = s.nodata
        for i in range(len(bands)):
            reproject(src[i], out[i], src_transform=stf, src_crs=s.crs, dst_transform=tr, dst_crs=crs,
                      resampling=resampling, src_nodata=src_nd, dst_nodata=nodata, init_dest_nodata=False)
    return out


def tiles(bounds_ll, step):
    lo0, la0, lo1, la1 = bounds_ll
    return [(la, lo) for la in range(int(math.floor(la0 / step) * step), int(math.floor(la1 / step) * step) + 1, step)
            for lo in range(int(math.floor(lo0 / step) * step), int(math.floor(lo1 / step) * step) + 1, step)]


REBUILD = "--rebuild" in sys.argv   # перезаписать уже готовые кубы
idx = []
for e in ev.itertuples():
    f_out = os.path.join(DATA, "cubes", e.event + ".npz")
    g = allpoly[e.event]; crs = g.crs
    b = g.total_bounds
    x0 = math.floor((b[0] - BUFFER_M) / GRID_M) * GRID_M; y1 = math.ceil((b[3] + BUFFER_M) / GRID_M) * GRID_M
    x1 = math.ceil((b[2] + BUFFER_M) / GRID_M) * GRID_M; y0 = math.floor((b[1] - BUFFER_M) / GRID_M) * GRID_M
    W = int(round((x1 - x0) / GRID_M)); H = int(round((y1 - y0) / GRID_M))
    tr = from_origin(x0, y1, GRID_M, GRID_M); shp = (H, W); bb = (x0, y0, x1, y1)
    if os.path.exists(f_out) and not REBUILD:
        idx.append(json.load(open(f_out.replace(".npz", ".json")))); print("skip", e.event); continue
    X = np.full((len(CH), H, W), np.nan, np.float32)
    # --- метки
    label = rasterize([(geom, 1) for geom in g.geometry], shp, transform=tr, fill=0, dtype="uint8")
    fd, img = e.ign_time_utc, e.image_date + pd.Timedelta(hours=6)
    same_site = ev[(ev.site == e.site) & (ev.event != e.event)]
    conc = same_site[(same_site.ign_time_utc >= fd) & (same_site.ign_time_utc <= img)].event.tolist()
    prior = same_site[same_site.ign_time_utc < fd]
    ignore = np.zeros(shp, bool)
    if conc:
        ignore = rasterize([(geom, 1) for c in conc for geom in allpoly[c].to_crs(crs).geometry], shp, transform=tr,
                           fill=0, dtype="uint8").astype(bool) & (label == 0)
    pb = np.full(shp, np.nan, np.float32)
    for p in prior.sort_values("ign_time_utc").itertuples():   # позднейший пожар перезаписывает -> минимум суток
        m = rasterize([(geom, 1) for geom in allpoly[p.event].to_crs(crs).geometry], shp, transform=tr, fill=0, dtype="uint8").astype(bool)
        pb[m] = (fd - p.ign_time_utc).total_seconds() / 86400
    X[CH.index("prior_burn_days")] = pb
    # --- точка возгорания
    rr, cc = np.mgrid[0:H, 0:W]
    xs = x0 + (cc + 0.5) * GRID_M; ys = y1 - (rr + 0.5) * GRID_M
    X[CH.index("dist_ign_km")] = np.hypot(xs - e.ign_x, ys - e.ign_y) / 1000
    ign_rc = [int((y1 - e.ign_y) // GRID_M), int((e.ign_x - x0) // GRID_M)]
    del rr, cc, xs, ys
    # --- рельеф
    bll = transform_bounds(crs, "EPSG:4326", *bb, densify_pts=21)
    dem = np.full(shp, np.nan, np.float32)
    for la, lo in tiles(bll, 1):
        try:
            dem = np.where(np.isnan(dem), warp_into(DEM_URL.format(la=la, lo=lo), shp, tr, crs, [1], Resampling.bilinear, window_ll=bb)[0], dem)
        except Exception as ex:
            print("  DEM tile missing", la, lo, str(ex)[:80])
    gy, gx = np.gradient(dem, GRID_M)           # gy по строкам (на юг), gx на восток
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)                 # направление склона вниз, от севера по часовой
    X[CH.index("elev")] = dem; X[CH.index("slope_deg")] = slope
    X[CH.index("aspect_sin")] = np.sin(aspect); X[CH.index("aspect_cos")] = np.cos(aspect)
    # --- растительный покров
    lc = np.zeros(shp, np.uint8)
    for la, lo in tiles(bll, 3):
        try:
            t = warp_into(WC_URL.format(la=la, lo=lo), shp, tr, crs, [1], Resampling.mode, np.uint8, 0, window_ll=bb)[0]
            lc = np.where(lc == 0, t, lc)
        except Exception as ex:
            print("  WorldCover tile missing", la, lo, str(ex)[:80])
    # --- GEE: до пожара (топливо) и после (контроль)
    m = gm.loc[e.event] if e.event in gm.index else None
    if e.gee_pre_ok and isinstance(m["pre"], str):
        pre = warp_into(os.path.join(GEE_DIR, m["pre"]), shp, tr, crs, [1, 2, 3, 4, 5, 6], Resampling.bilinear)
        pre[(pre <= 0) | (pre > 1.5)] = np.nan
        X[0:6] = pre
        nd = lambda a, c: (a - c) / (a + c + 1e-9)
        X[CH.index("pre_NDVI")] = nd(pre[3], pre[2]); X[CH.index("pre_NDMI")] = nd(pre[3], pre[4]); X[CH.index("pre_NBR")] = nd(pre[3], pre[5])
    if e.gee_post_ok and isinstance(m["post"], str):
        post = warp_into(os.path.join(GEE_DIR, m["post"]), shp, tr, crs, [4, 6], Resampling.bilinear)
        post[(post <= 0) | (post > 1.5)] = np.nan
        X[CH.index("post_NBR")] = (post[0] - post[1]) / (post[0] + post[1] + 1e-9)
        X[CH.index("dNBR")] = X[CH.index("pre_NBR")] - X[CH.index("post_NBR")]
    np.savez_compressed(f_out, x=X.astype(np.float16), landcover=lc, label=label, ignore=ignore)
    cov = lambda k: round(float(np.isfinite(X[CH.index(k)]).mean()), 3)
    meta = dict(event=e.event, site=e.site, crs=f"EPSG:{crs.to_epsg()}", transform=list(tr)[:6], H=H, W=W, px_m=GRID_M,
                channels=CH, ign_rc=ign_rc, concurrent_ignored=";".join(conc), n_prior_events=len(prior),
                burned_px=int(label.sum()), burned_frac=round(float(label.mean()), 4), ignored_frac=round(float(ignore.mean()), 4),
                prior_burn_frac=round(float(np.isfinite(pb).mean()), 4), cov_pre=cov("pre_NBR"), cov_post=cov("post_NBR"),
                cov_dem=cov("elev"), cov_lc=round(float((lc > 0).mean()), 3),
                lc_in_burn=json.dumps({int(k): round(float(v), 3) for k, v in zip(*np.unique(lc[label == 1], return_counts=True))}
                                      if label.any() else {}),
                elev_mean=round(float(np.nanmean(dem)), 1), slope_p95=round(float(np.nanpercentile(slope, 95)), 2))
    if meta["lc_in_burn"] != "{}":   # доли классов внутри гари
        d = json.loads(meta["lc_in_burn"]); s = sum(d.values()); meta["lc_in_burn"] = json.dumps({k: round(v / s, 3) for k, v in d.items()})
    json.dump(meta, open(f_out.replace(".npz", ".json"), "w"), ensure_ascii=False)
    idx.append(meta)
    print(f"{e.event:24s} {H}x{W} burned={meta['burned_frac']:.3f} ign={meta['ignored_frac']:.3f} prior={meta['prior_burn_frac']:.3f} "
          f"pre={meta['cov_pre']:.2f} post={meta['cov_post']:.2f} dem={meta['cov_dem']:.2f} lc={meta['cov_lc']:.2f} "
          f"elev={meta['elev_mean']} slope95={meta['slope_p95']} lc_burn={meta['lc_in_burn']}", flush=True)
    del X
pd.DataFrame(idx).drop(columns=["channels"]).to_csv(os.path.join(DATA, "cubes_index.csv"), index=False)
print("cubes:", len(idx))
