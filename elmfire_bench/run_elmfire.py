"""Прогон ELMFIRE на 40 тестовых пожарах -> case_XX/elmfire_toa.tif (время прихода огня, с от t0, сетка кейса 96 x 96, 500 м).

  python elmfire_bench/run_elmfire.py                 все кейсы (уже посчитанные пропускаются, --force - пересчитать)
  python elmfire_bench/run_elmfire.py 4 21            только case_04 и case_21

Нужны: бинарник elmfire_<версия> в PATH (или $ELMFIRE_BIN), утилиты GDAL в PATH, numpy, rasterio.
Входы ELMFIRE собираются по образцу tutorials/02-transient-wind:
  landscape.tif - 8 слоёв Int16: dem, slp, asp, fbfm40, cc, ch, cbh, cbd (крона = 0, степь)
  adj.tif = 1; phi.tif = 1. Начальный фронт - точечные зажигания (t = 0) во всех клетках границы стартовой гари: путь
  через phi <= 0 в ELMFIRE 1.1 помечает клетки сгоревшими, но не ставит вокруг них фронт (огонь не идёт); лимит 300 точек.
  ws/wd/m1/m10/m100/lh/lw.tif - по 73 слоя через 1 ч (0..72 ч), кусочно-постоянно: 0-12 ч из окна 12, 12-24 из 24,
  24-48 из 48, 48-72 из 72 (meta.json -> weather_windows[k].elmfire; ws - миль/ч на 20 фт, wd - откуда дует, °).
Кейс с ok = false в любом окне пропускается (нет погоды) и пишется в cases/elmfire_skipped.txt.
"""
import glob, json, os, shutil, subprocess, sys
import numpy as np, rasterio
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
ELM = os.environ.get("ELMFIRE_BASE_DIR", os.path.expanduser("~/elmfire"))
VER = open(os.path.join(ELM, "VERSION")).read().strip()
BIN = os.environ.get("ELMFIRE_BIN") or shutil.which(f"elmfire_{VER}") or os.path.join(ELM, "build/linux/bin", f"elmfire_{VER}")
TSTOP_H = 72
MAX_IGN = 300                                                    # размер массивов X_IGN/Y_IGN/T_IGN в elmfire_vars.f90
WX = ("ws", "wd", "m1", "m10", "m100", "lh", "lw")
WX_KEY = dict(ws="ws_20ft_mph", wd="wind_from_deg", m1="m1_pct", m10="m10_pct", m100="m100_pct", lh="mlh_pct", lw="mlw_pct")

DATA = """&INPUTS
FUELS_AND_TOPOGRAPHY_DIRECTORY = './inputs'
LANDSCAPE_FILENAME             = 'landscape'
ADJ_FILENAME                   = 'adj'
PHI_FILENAME                   = 'phi'
DT_METEOROLOGY                 = 3600.0
WEATHER_DIRECTORY              = './inputs'
WS_FILENAME                    = 'ws'
WD_FILENAME                    = 'wd'
M1_FILENAME                    = 'm1'
M10_FILENAME                   = 'm10'
M100_FILENAME                  = 'm100'
USE_CONSTANT_LH                = .FALSE.
MLH_FILENAME                   = 'lh'
USE_CONSTANT_LW                = .FALSE.
MLW_FILENAME                   = 'lw'
/

&OUTPUTS
OUTPUTS_DIRECTORY    = './outputs'
DTDUMP               = {tstop}
DUMP_FLIN            = .FALSE.
DUMP_SPREAD_RATE     = .FALSE.
DUMP_TIME_OF_ARRIVAL = .TRUE.
CONVERT_TO_GEOTIFF   = .FALSE.
/

&TIME_CONTROL
SIMULATION_DT    = 1.0
TARGET_CFL       = 0.2
SIMULATION_TSTOP = {tstop}
/

&MONTE_CARLO
NUM_METEOROLOGY_TIMES = {nmet}
/

&SIMULATOR
NUM_IGNITIONS = {nign}
{ign}
WX_BILINEAR_INTERPOLATION = .FALSE.
WSMFEFF_LOW_MULT = 0.011364
/

&MISCELLANEOUS
PATH_TO_GDAL                   = 'auto'
SCRATCH                        = './scratch'
/
"""


def write(path, arr, prof, dtype):
    arr = np.asarray(arr)
    arr = arr[None] if arr.ndim == 2 else arr
    p = dict(prof, driver="GTiff", count=arr.shape[0], dtype=dtype, nodata=-9999, compress="deflate")
    with rasterio.open(path, "w", **p) as r:
        r.write(arr.astype(dtype))


def weather_bands(meta):
    w = [x["elmfire"] for x in meta["weather_windows"]]
    win = [0 if h < 12 else 1 if h < 24 else 2 if h < 48 else 3 for h in range(TSTOP_H + 1)]
    return {q: np.array([w[k][WX_KEY[q]] for k in win], np.float32) for q in WX}


def run_case(d, force=False):
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    out_tif = os.path.join(d, "elmfire_toa.tif")
    if os.path.exists(out_tif) and not force:
        return "есть"
    if not all(x["ok"] for x in meta["weather_windows"]):
        return "нет погоды"
    run = os.path.join(d, "elmfire_run")
    shutil.rmtree(run, ignore_errors=True)
    for s in ("inputs", "outputs", "scratch"):
        os.makedirs(os.path.join(run, s))
    rd = lambda n: rasterio.open(os.path.join(d, n))
    with rd("fbfm40.tif") as r:
        prof = r.profile; fb = r.read(1).astype(np.int32); fbn = r.nodata
    prof = {k: prof[k] for k in ("width", "height", "crs", "transform")}
    if fbn is not None:
        fb[fb == fbn] = 99
    fb[fb <= 0] = 99
    lay = [fb]
    with rd("elev.tif") as r: dem = r.read(1)
    with rd("slope.tif") as r: slp = r.read(1)
    with rd("aspect.tif") as r: asp = r.read(1)
    with rd("start_burned.tif") as r: st = r.read(1) > 0
    dem, slp, asp = [np.nan_to_num(np.round(a.astype(np.float64)), nan=0.0) for a in (dem, slp, asp)]
    slp, asp = np.clip(slp, 0, 89), np.mod(asp, 360)
    z = np.zeros_like(fb)
    write(os.path.join(run, "inputs/landscape.tif"), np.stack([dem, slp, asp, fb, z, z, z, z]), prof, "int16")
    write(os.path.join(run, "inputs/adj.tif"), np.ones(fb.shape), prof, "float32")
    write(os.path.join(run, "inputs/phi.tif"), np.ones(fb.shape), prof, "float32")
    for q, v in weather_bands(meta).items():
        write(os.path.join(run, f"inputs/{q}.tif"), np.broadcast_to(v[:, None, None], (len(v),) + fb.shape), prof, "float32")
    edge = st & ~binary_erosion(st)                              # внутренние клетки старта на фронт не влияют
    iy, ix = np.nonzero(edge)
    if not 0 < len(ix) <= MAX_IGN:
        return f"ОШИБКА: клеток границы старта {len(ix)} (нужно 1..{MAX_IGN})"
    xs, ys = prof["transform"] * (ix + 0.5, iy + 0.5)
    ign = "\n".join(f"X_IGN({i})={x:.1f}\nY_IGN({i})={y:.1f}\nT_IGN({i})=0.0" for i, (x, y) in enumerate(zip(xs, ys), 1))
    open(os.path.join(run, "inputs/elmfire.data"), "w").write(
        DATA.format(tstop=f"{TSTOP_H * 3600.0:.1f}", nmet=TSTOP_H + 1, nign=len(ix), ign=ign))
    with open(os.path.join(run, "elmfire.log"), "w") as log:
        rc = subprocess.run([BIN, "./inputs/elmfire.data"], cwd=run, stdout=log, stderr=subprocess.STDOUT).returncode
    toa_f = sorted(glob.glob(os.path.join(run, "outputs/time_of_arrival*.bil")) + glob.glob(os.path.join(run, "outputs/time_of_arrival*.tif")))
    if rc != 0 or not toa_f:
        return f"ОШИБКА rc={rc}, см. {run}/elmfire.log"
    with rasterio.open(toa_f[0]) as r:
        toa = r.read(1).astype(np.float32); nod = r.nodata
    if nod is not None:
        toa[toa == nod] = -1
    toa[~np.isfinite(toa)] = -1
    write(out_tif, toa, prof, "float32")
    new = (toa > 0) & ~st
    return f"ok: новая гарь за 72 ч {new.sum() * 0.25:.1f} км² (факт {np.load(os.path.join(d, 'truth.npz'))['Y'][3].sum() * 0.25:.1f})"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    dirs = sorted(glob.glob(os.path.join(HERE, "cases", "case_*")))
    if args:
        dirs = [os.path.join(HERE, "cases", f"case_{int(a):02d}") for a in args]
    skipped = []
    for d in dirs:
        res = run_case(d, force)
        print(os.path.basename(d), res, flush=True)
        if res == "нет погоды":
            skipped.append(os.path.basename(d))
    if not args:
        open(os.path.join(HERE, "cases", "elmfire_skipped.txt"), "w").write("\n".join(skipped) + "\n")
