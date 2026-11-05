"""Zagruzki iz GEE.  python acquire.py probe [lon lat] | stack [w s e n] | vrt | s2dem | s1 | prod | esri | all"""
import os, sys, math, time, subprocess
import ee, geemap
import config as cfg

SEASON = ("2022-04-01", "2022-10-31")
S1_WIN = ("2022-05-01", "2022-09-30")
STEP = 2.0

def _tiled(name, build, scale, dtype):
    tdir = os.path.join(cfg.DATA_DIR, f"{name}_tiles_{cfg.AOI_NAME}")
    os.makedirs(tdir, exist_ok=True)
    w, s, e, n = cfg.AOI_BBOX
    xs = [w + i*STEP for i in range(math.ceil((e-w)/STEP))]
    ys = [s + j*STEP for j in range(math.ceil((n-s)/STEP))]
    total = len(xs)*len(ys); done = 0; tiles = []
    for j, y0 in enumerate(ys):
        for i, x0 in enumerate(xs):
            done += 1
            x1 = min(x0+STEP, e); y1 = min(y0+STEP, n)
            path = os.path.join(tdir, f"{name}_r{j:02d}_c{i:02d}.tif")
            tiles.append(path)
            if os.path.exists(path) and os.path.getsize(path) > 100000:
                print(f"[{done}/{total}] propusk {os.path.basename(path)}"); continue
            sub = ee.Geometry.Rectangle([x0, y0, x1, y1])
            print(f"[{done}/{total}] kachayu {os.path.basename(path)}")
            try:
                geemap.download_ee_image(image=build(sub).clip(sub), filename=path, region=sub,
                    scale=scale, crs="EPSG:4326", dtype=dtype, max_tile_size=2,
                    num_threads=1, overwrite=True)
            except Exception as ex:
                print("   !! upal:", ex)
                if os.path.exists(path): os.remove(path)
    vrt = os.path.join(cfg.DATA_DIR, f"{name}_{cfg.AOI_NAME}.vrt")
    ok = [t for t in tiles if os.path.exists(t) and os.path.getsize(t) > 100000]
    subprocess.run(["gdalbuildvrt", vrt] + ok, check=True)
    print(f"Gotovo {name}: {len(ok)}/{total} -> {vrt}")

def _clouds(img):
    qa = img.select("QA60")
    return img.updateMask(qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0)))

def s2dem():
    aoi = ee.Geometry.Rectangle(cfg.AOI_BBOX)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(cfg.DATE_START, cfg.DATE_END)
          .filterBounds(aoi).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.CLOUD_MAX))
          .map(_clouds).select(["B4","B8","B11","B12"]))
    print("S2 ->", cfg.S2_RAW)
    geemap.download_ee_image(s2.median().clip(aoi), cfg.S2_RAW, region=aoi, scale=cfg.SCALE,
        crs="EPSG:4326", dtype="int16", max_tile_size=8, overwrite=True)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().clip(aoi)
    print("DEM ->", cfg.DEM_RAW)
    geemap.download_ee_image(dem, cfg.DEM_RAW, region=aoi, scale=30, crs="EPSG:4326",
        dtype="int16", max_tile_size=8, overwrite=True)

def esri():
    aoi = ee.Geometry.Rectangle(cfg.AOI_BBOX)
    lc = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
          .filterDate("2022-01-01", "2023-01-01").mosaic().clip(aoi))
    out = f"{cfg.DATA_DIR}/ESRI_LC_2022_{cfg.AOI_NAME}.tif"
    print("ESRI ->", out)
    geemap.download_ee_image(lc, out, region=aoi, scale=cfg.SCALE, crs="EPSG:4326",
        dtype="uint8", resampling="near", max_tile_size=8, overwrite=True)

def s1():
    def build(aoi):
        base = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(*S1_WIN)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filterBounds(aoi).select(["VV", "VH"]))
        vv = base.select("VV").median().rename("VV")
        vh = base.select("VH").median().rename("VH")
        return vv.addBands(vh)
    _tiled("S1_VVVH", build, 20, "float32")

def prod():
    def build(aoi):
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(*SEASON)
               .filterBounds(aoi).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.CLOUD_MAX)).map(_clouds))
        def veg(img):
            nir = img.select("B8").multiply(0.0001); red = img.select("B4").multiply(0.0001)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            return img.addBands(ndvi).addBands(ndvi.multiply(nir).rename("NIRv"))
        col = col.map(veg); nd = col.select("NDVI")
        return (nd.mean().rename("INDVI")
                .addBands(col.select("NIRv").max().rename("NIRv"))
                .addBands(nd.max().subtract(nd.min()).rename("NDVI_amp")))
    _tiled("productivity_20m", build, 30, "float32")

# ---------------- ves Kazahstan: stek priznakov po taylam ----------------
# QA60 dlya snimkov posle 25.01.2022 ne zapolnen v GEE -> oblaka cherez Cloud Score+
CSP = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_MIN = 0.6

def _kz():
    return (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
            .filter(ee.Filter.eq("country_na", "Kazakhstan")).geometry())

def _s2col(aoi, start, end):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.CLOUD_MAX))
            .linkCollection(ee.ImageCollection(CSP), ["cs_cdf"])
            .map(lambda i: i.updateMask(i.select("cs_cdf").gte(CS_MIN))))

def _stack(aoi):
    # S2 leto: mediana kanalov -> indeksy (kak features.indices)
    b = _s2col(aoi, cfg.DATE_START, cfg.DATE_END).select(["B4", "B8", "B11", "B12"]).median().multiply(0.0001)
    ndvi = b.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = b.normalizedDifference(["B8", "B11"]).rename("NDMI")
    savi = b.expression("1.5*(N-R)/(N+R+0.5)", {"N": b.select("B8"), "R": b.select("B4")}).rename("SAVI")
    nbr = b.normalizedDifference(["B8", "B12"]).rename("NBR")
    # rel'ef: ee.Terrain.slope schitaet v metrah, proekciya ne nuzhna
    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(aoi).select("DEM").mosaic()
           .setDefaultProjection("EPSG:4326", None, 30))
    elev = dem.resample("bilinear").rename("elevation")
    slope = ee.Terrain.slope(dem).resample("bilinear").rename("slope")
    # S1: dva sezona (2021-2022), v 2022 tol'ko S1A -> propuski; pustaya kollekciya -> nodata
    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
           .filter(ee.Filter.calendarRange(2021, 2022, "year")).filter(ee.Filter.calendarRange(5, 9, "month"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
    s1 = ee.Image(ee.Algorithms.If(s1c.size().gt(0), s1c.median(),
                                   ee.Image.constant([0, 0]).rename(["VV", "VH"]).toFloat().updateMask(0)))
    # produktivnost' za sezon (kak prod)
    col = _s2col(aoi, *SEASON)
    def veg(img):
        nir = img.select("B8").multiply(0.0001); red = img.select("B4").multiply(0.0001)
        nd = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        return nd.addBands(nd.multiply(nir).rename("NIRv"))
    col = col.map(veg); nd = col.select("NDVI")
    prod = (nd.mean().rename("INDVI").addBands(col.select("NIRv").max().rename("NIRv"))
            .addBands(nd.max().subtract(nd.min()).rename("NDVI_amp")))
    # vysota pologa (ETH 2020, 10 m) i ESRI LULC 2022
    # u ETH net znacheniy tam, gde net rastitel'nosti (zastroyka, voda, pustyni) -> 0 m, inache pikseli vypadut
    ch = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").unmask(0).rename("canopy_h")
    # v gorah ETH "vidit" polog na lugah, skalah i snegu (test almaty: >=6 m u 50-60% pastbishch vyshe 1500 m);
    # vyshe 1000 m vysotu ostavlyaem tol'ko tam, gde derev'ya/kustarnik vidyat Hansen ili WorldCover
    tc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").select("treecover2000")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    woody = tc.gte(10).Or(wc.eq(10)).Or(wc.eq(20))
    ch = ch.where(elev.gte(1000).And(woody.Not()), 0)
    lc = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
          .filterDate("2022-01-01", "2023-01-01").filterBounds(aoi).mosaic().rename("lc"))
    img = ee.Image.cat([ndvi, ndmi, savi, nbr, elev, slope, s1, prod, ch, lc]).toFloat()
    names = list(cfg.STACK_BANDS)
    return (img.select(names).multiply(ee.Image.constant(list(cfg.STACK_BANDS.values())).rename(names))
            .round().clip(_kz()).unmask(cfg.NODATA_I16).toInt16())

def _kz_tiles(bbox=None):
    w, s, e, n = cfg.KZ_GRID; T = cfg.TILE_DEG
    cells = [(w + i*T, s + j*T) for j in range(int((n-s)/T)) for i in range(int((e-w)/T))]
    if bbox:
        bw, bs, be, bn = bbox
        cells = [(x, y) for x, y in cells if x < be and x + T > bw and y < bn and y + T > bs]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([x, y, x+T, y+T]), {"x": x, "y": y}) for x, y in cells])
    hit = fc.filterBounds(_kz())
    return list(zip(hit.aggregate_array("x").getInfo(), hit.aggregate_array("y").getInfo()))

class RestrictedMode(Exception):
    pass

def _dl_tile(x, y, part, tries=5):
    """odin 2° tayl steka; 429 -> pauza i povtor, memory limit -> men'she kusok, Restricted Mode -> RestrictedMode"""
    T = cfg.TILE_DEG; tile_mb = 8; wait = 60
    for a in range(1, tries + 1):
        try:
            geemap.download_ee_image(_stack(ee.Geometry.Rectangle([x, y, x+T, y+T])), part, crs="EPSG:4326",
                crs_transform=[cfg.RES, 0, x, 0, -cfg.RES, y + T], shape=(cfg.TILE_PX, cfg.TILE_PX),
                dtype="int16", max_tile_size=tile_mb, num_threads=2, overwrite=True)
            return
        except Exception as ex:
            msg = str(ex)
            if "Restricted Mode" in msg: raise RestrictedMode(msg) from ex
            if a == tries: raise
            if "memory limit" in msg.lower() and tile_mb > 2:
                tile_mb //= 2; print(f"   memory limit -> max_tile_size={tile_mb}, povtor {a}/{tries - 1}", flush=True)
            elif "429" in msg or "Too Many Requests" in msg or "concurrency" in msg.lower():
                print(f"   429 -> pauza {wait} s, povtor {a}/{tries - 1}", flush=True); time.sleep(wait); wait *= 2
            else:
                raise
            if os.path.exists(part): os.remove(part)

def stack(bbox=None):
    """python acquire.py stack [zapad yug vostok sever] - bez bbox ves Kazahstan"""
    tiles = _kz_tiles(bbox); T = cfg.TILE_DEG; failed = []; k = 0
    print(f"Taylov: {len(tiles)} -> {cfg.STACK_DIR}")
    for k, (x, y) in enumerate(tiles, 1):
        path = os.path.join(cfg.STACK_DIR, f"stack_{x:03.0f}E_{y:02.0f}N.tif")
        if os.path.exists(path):
            print(f"[{k}/{len(tiles)}] propusk {os.path.basename(path)}"); continue
        part = path + ".part.tif"
        print(f"[{k}/{len(tiles)}] kachayu {os.path.basename(path)}", flush=True)
        t0 = time.time()
        try:
            _dl_tile(x, y, part)
            subprocess.run(["gdal_translate", "-q", "-a_nodata", str(cfg.NODATA_I16),
                            "-co", "COMPRESS=ZSTD", "-co", "ZSTD_LEVEL=9", "-co", "PREDICTOR=2",
                            "-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
                            "-co", "BIGTIFF=YES", part, path], check=True)
            # obzory - chtoby 100+ taylov bystro otkryvalis' v QGIS i chitalis' s prorezhivaniem
            subprocess.run(["gdaladdo", "-q", "-r", "nearest", "--config", "COMPRESS_OVERVIEW", "ZSTD",
                            "--config", "PREDICTOR_OVERVIEW", "2", path, "4", "16", "64"], check=True)
            print(f"   gotovo za {(time.time() - t0) / 60:.1f} min, {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
        except RestrictedMode as ex:
            # kvota proekta ischerpana: ostal'nye tayly upadut tak zhe, ne tratim vremya
            print("   !! STOP: proekt GEE v Restricted Mode, ostanavlivayus'.\n  ", ex, flush=True)
            failed.append(os.path.basename(path))
            if os.path.exists(path): os.remove(path)
            break
        except Exception as ex:
            print("   !! upal:", ex, flush=True); failed.append(os.path.basename(path))
            if os.path.exists(path): os.remove(path)
        finally:
            if os.path.exists(part): os.remove(part)
    left = len(tiles) - k
    if left: print(f"Ne nachaty {left} taylov (ostanovka po Restricted Mode)")
    if failed: print(f"UPALO {len(failed)}/{len(tiles)}: {' '.join(failed)}\n-> zapustite tu zhe komandu eshche raz, gotovye tayly propustyatsya")
    vrt()

def probe(x=77.0, y=43.4, n=280):
    """bystraya proverka steka na kusochke n x n pikseley (~5 km) + spisok taylov KZ"""
    print("Taylov v Kazahstane:", len(_kz_tiles()))
    path = os.path.join(cfg.STACK_DIR, "probe.tif")
    geemap.download_ee_image(_stack(ee.Geometry.Rectangle([x, y - n*cfg.RES, x + n*cfg.RES, y])), path,
        crs="EPSG:4326", crs_transform=[cfg.RES, 0, x, 0, -cfg.RES, y], shape=(n, n), dtype="int16", overwrite=True)
    import numpy as np, rasterio
    with rasterio.open(path) as s: a = s.read()
    for b, (k, sc) in zip(a, cfg.STACK_BANDS.items()):
        v = b[b != cfg.NODATA_I16] / sc
        if v.size: print(f"{k:10s} valid {100*v.size/b.size:5.1f}%  min {v.min():8.2f}  med {np.median(v):8.2f}  max {v.max():8.2f}")
        else: print(f"{k:10s} NET DANNYH")
    os.remove(path)

def vrt():
    ok = sorted(os.path.join(cfg.STACK_DIR, f) for f in os.listdir(cfg.STACK_DIR)
                if f.startswith("stack_") and f.endswith("N.tif"))
    if not ok:
        print("VRT: net ni odnogo tayla v", cfg.STACK_DIR); return
    subprocess.run(["gdalbuildvrt", "-srcnodata", str(cfg.NODATA_I16), "-vrtnodata", str(cfg.NODATA_I16),
                    cfg.STACK_VRT] + ok, check=True)
    print(f"VRT: {len(ok)} taylov -> {cfg.STACK_VRT}")

if __name__ == "__main__":
    ee.Initialize(project=cfg.GEE_PROJECT)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"s2dem": s2dem, "esri": esri, "s1": s1, "prod": prod, "vrt": vrt, "probe": probe}
    if cmd == "probe" and len(sys.argv) >= 4: probe(float(sys.argv[2]), float(sys.argv[3]))
    elif cmd == "stack": stack([float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else None)
    elif cmd == "all":
        for f in [s2dem, esri, s1, prod]: f()
    elif cmd in fns: fns[cmd]()
    else: print("probe | stack [w s e n] | vrt | s2dem | s1 | prod | esri | all")
