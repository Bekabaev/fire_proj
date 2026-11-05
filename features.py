"""Priznaki i vyravnivanie na setku S2_indices"""
import os, sys, subprocess
import numpy as np, rasterio
import config as cfg

S1_VRT   = f"{cfg.DATA_DIR}/S1_VVVH_{cfg.AOI_NAME}.vrt"
S1_ALN   = f"{cfg.DATA_DIR}/S1_VVVH_aligned_{cfg.AOI_NAME}.tif"
PROD_VRT = f"{cfg.DATA_DIR}/productivity_20m_{cfg.AOI_NAME}.vrt"
PROD_ALN = f"{cfg.DATA_DIR}/productivity_aligned_{cfg.AOI_NAME}.tif"
ESRI_RAW = f"{cfg.DATA_DIR}/ESRI_LC_2022_{cfg.AOI_NAME}.tif"
ESRI_ALN = cfg.ESRI_ALIGNED

def _warp(src, dst, resample):
    with rasterio.open(cfg.S2_INDICES) as ref:
        b = ref.bounds
        te = [str(b.left), str(b.bottom), str(b.right), str(b.top)]
        ts = [str(ref.width), str(ref.height)]
    print("Vyravnivayu ->", dst)
    subprocess.run(["gdalwarp", "-t_srs", "EPSG:4326", "-te", *te, "-ts", *ts,
                    "-r", resample, "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES",
                    "-co", "TILED=YES", "-overwrite", src, dst], check=True)

def indices():
    print("Indeksy ->", cfg.S2_INDICES)
    with rasterio.open(cfg.S2_RAW) as src:
        p = src.profile.copy()
        p.update(count=4, dtype="float32", nodata=np.nan, BIGTIFF="YES",
                 compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(cfg.S2_INDICES, "w", **p) as dst:
            dst.descriptions = ("NDVI", "NDMI", "SAVI", "NBR")
            for _, win in src.block_windows():
                b = src.read(window=win).astype("float32")
                b[b == src.nodata] = np.nan
                b = b / 10000.0
                red, nir, swir1, swir2 = b[0], b[1], b[2], b[3]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi = (nir - red) / (nir + red)
                    ndmi = (nir - swir1) / (nir + swir1)
                    savi = 1.5 * (nir - red) / (nir + red + 0.5)
                    nbr = (nir - swir2) / (nir + swir2)
                dst.write(np.stack([ndvi, ndmi, savi, nbr]).astype("float32"), window=win)

def terrain():
    # syroy DEM 30m mog byt udalen - togda berem uzhe vyrovnenny DEM_20m_aligned
    raw = os.path.exists(cfg.DEM_RAW)
    src = cfg.DEM_RAW if raw else cfg.DEM_ALIGNED
    if not raw: print("Net", cfg.DEM_RAW, "-> uklon iz", src)
    with rasterio.open(src, "r+") as dem:             # vmesto gdal_edit.py -a_nodata
        dem.nodata = -32768
    # uklon v metrah: v EPSG:4326 gradus dolgoty na 45° ~79 km, a ne 111, -s 111120 iskazhal uklon
    subprocess.run(["gdalwarp", "-t_srs", cfg.TM_SRS, "-tr", "30", "30", "-r", "bilinear", "-ot", "Float32",
                    "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-co", "TILED=YES",
                    "-overwrite", src, cfg.DEM_TM], check=True)
    subprocess.run(["gdaldem", "slope", cfg.DEM_TM, cfg.SLOPE_RAW, "-compute_edges",
                    "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-co", "TILED=YES"], check=True)
    if raw: _warp(cfg.DEM_RAW, cfg.DEM_ALIGNED, "bilinear")
    _warp(cfg.SLOPE_RAW, cfg.SLOPE_ALIGNED, "bilinear")

def align(target="all"):
    jobs = {"s1": (S1_VRT, S1_ALN, "bilinear"),
            "prod": (PROD_VRT, PROD_ALN, "bilinear"),
            "esri": (ESRI_RAW, ESRI_ALN, "near")}
    todo = jobs if target == "all" else {target: jobs[target]}
    for src, dst, r in todo.values():
        _warp(src, dst, r)

if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "indices"
    if cmd == "indices": indices()
    elif cmd == "terrain": terrain()
    elif cmd == "align": align(a[1] if len(a) > 1 else "all")
    else: print("indices | terrain | align [s1|prod|esri|all]")
