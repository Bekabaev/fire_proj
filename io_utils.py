import numpy as np
import rasterio
from rasterio.windows import Window
import config as cfg


def open_sources():
    if cfg.STACK_MODE:
        return {"stack": rasterio.open(cfg.STACK_VRT)}
    return {
        "idx":   rasterio.open(cfg.S2_INDICES),
        "dem":   rasterio.open(cfg.DEM_ALIGNED),
        "slope": rasterio.open(cfg.SLOPE_ALIGNED),
        "s1":    rasterio.open(f"{cfg.DATA_DIR}/S1_VVVH_aligned_{cfg.AOI_NAME}.tif"),
        "prod":  rasterio.open(f"{cfg.DATA_DIR}/productivity_aligned_{cfg.AOI_NAME}.tif"),
    }


def close_sources(srcs):
    for s in srcs.values():
        s.close()


def grid_shape(srcs):
    s = srcs["stack"] if "stack" in srcs else srcs["idx"]
    return s.height, s.width


def ref_profile(srcs):
    s = srcs["stack"] if "stack" in srcs else srcs["idx"]
    p = s.profile.copy()
    p["driver"] = "GTiff"          # u VRT driver=VRT
    return p


def valid_fraction(srcs, step=64):
    """dolya pikseley s dannymi (po prorezhennomu chteniyu pervogo kanala)"""
    s = srcs["stack"] if "stack" in srcs else srcs["idx"]
    a = s.read(1, out_shape=(max(1, s.height // step), max(1, s.width // step)))
    ok = np.isfinite(a)
    if s.nodata is not None and np.isfinite(s.nodata):
        ok &= a != s.nodata
    return float(ok.mean())


def open_lc(srcs):
    """(fayl, nomer kanala) klassa ESRI na setke priznakov"""
    if "stack" in srcs:
        return rasterio.open(cfg.STACK_VRT), len(cfg.STACK_BANDS)
    return rasterio.open(cfg.ESRI_ALIGNED), 1


def _read_stack(st, win):
    nf = len(cfg.STACK_BANDS) - 1                      # poslednij kanal - lc, ne priznak
    a = st.read(list(range(1, nf + 1)), window=win)
    flat = a.reshape(nf, -1).T
    valid = (flat != cfg.NODATA_I16).all(axis=1)
    scale = np.array([v for k, v in cfg.STACK_BANDS.items() if k != "lc"], dtype="float32")
    return flat.astype("float32") / scale, valid


def iter_windows(H, W, block):
    for row in range(0, H, block):
        for col in range(0, W, block):
            yield Window(col, row, min(block, W - col), min(block, H - row))


def read_window(srcs, win):
    if "stack" in srcs:
        return _read_stack(srcs["stack"], win)
    idx = srcs["idx"].read(window=win).astype("float32")
    dem = srcs["dem"].read(1, window=win).astype("float32")
    slp = srcs["slope"].read(1, window=win).astype("float32")
    s1  = srcs["s1"].read(window=win).astype("float32")
    prd = srcs["prod"].read(window=win).astype("float32")
    stack = np.vstack([idx, dem[None], slp[None], s1, prd])
    flat = stack.reshape(stack.shape[0], -1).T
    valid = ~np.isnan(flat).any(axis=1)
    valid &= ~np.isinf(flat).any(axis=1)
    dn, sn = srcs["dem"].nodata, srcs["slope"].nodata
    if dn is not None:
        valid &= flat[:, 4] != dn
    if sn is not None:
        valid &= flat[:, 5] != sn
    for key, off, nb in [("s1", 6, 2), ("prod", 8, 3)]:
        nd = srcs[key].nodata
        if nd is not None:
            for j in range(off, off + nb):
                valid &= flat[:, j] != nd
    return flat, valid
