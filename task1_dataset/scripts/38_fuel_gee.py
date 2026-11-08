"""Шаг 38. Карта топлива FBFM40 (пайплайн README_2.md: acquire._stack + правила label.fbfm_rule) сразу на национальной сетке 500 м.

Отличия от README_2: без k-means - правила label.py применяются попиксельно (кластеры там нужны только для ручной разметки);
признаки считаются в GEE на 100 м (не 20 м: в ~40 раз дешевле по квоте), затем доли каждого кода в ячейке 500 м EPSG:6933.
Пашня ESRI (класс 5) - отдельная доля «crop», а не NB3: в Казахстане много палов по стерне.

Выход: data/ext_kz/fuel_s2.npz - frac [len(CODES), GH, GW] uint8 (% площади ячейки), codes, valid (есть данные).
Тайлы 400 × 400 ячеек в data/ext_kz/fuel_tiles/ (готовые пропускаются - можно перезапускать после обрыва).
Запуск: EE_PROJECT=<id проекта GEE> python -X utf8 38_fuel_gee.py [tiles|merge|all]
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from kz_common import KZ, GH, GW, X0, Y1, CELL, KZ_BOUNDARY
import ee, geemap

# коды выхода: 20 кодов топлива label.py + пашня + негорючее
CODES = [101, 102, 103, 104, 105, 106, 107, 108, 109, 121, 122, 123, 124, 141, 142, 143, 148, 161, 162, 183]
CROP, NB = 250, 99
BANDS = CODES + [CROP, NB]
TILE = int(os.environ.get('TILE', 200)); FEAT_SCALE = 100
GEE_CRS = 'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'   # EPSG:6933 - GEE не знает этот код, передаём WKT
TDIR = os.path.join(KZ, "fuel_tiles"); OUT = os.path.join(KZ, "fuel_s2.npz")

# пороги label.py (fbfm_rule) - перенесены 1:1
FOREST_H, SHRUB_H, GS_H, HUMID_NDMI, BARE_INDVI = 6.0, 3.0, 1.0, 0.05, 0.08
RULE = (f"iv < {BARE_INDVI} ? 99 : "
        f"h >= {FOREST_H} ? (el >= 1400 ? 183 : iv >= 0.45 ? 162 : 161) : "
        f"h >= {SHRUB_H} ? (md > {HUMID_NDMI} ? (iv >= 0.5 ? 148 : 143) : (iv >= 0.25 ? 142 : 141)) : "
        f"h >= {GS_H} ? (md > {HUMID_NDMI} ? (iv >= 0.45 ? 124 : 123) : (iv >= 0.25 ? 122 : 121)) : "
        f"md > {HUMID_NDMI} ? (iv >= 0.5 && el < 800 && md > 0.25 ? 109 : iv >= 0.55 ? 108 : iv >= 0.45 ? 106 : iv >= 0.35 ? 105 : 103) : "
        f"iv >= 0.35 ? 107 : iv >= 0.25 ? 104 : iv >= 0.15 ? 102 : 101")


def fuel_image():
    import acquire, config as cfg
    st = acquire._stack(acquire._kz())
    names = list(cfg.STACK_BANDS)
    ok = st.select("INDVI").neq(cfg.NODATA_I16).And(st.select("lc").neq(cfg.NODATA_I16))
    f = st.toFloat().divide(ee.Image.constant(list(cfg.STACK_BANDS.values())).rename(names)).updateMask(ok)
    f = f.reproject(crs=GEE_CRS, scale=FEAT_SCALE)
    code = f.expression(RULE, {"iv": f.select("INDVI"), "h": f.select("canopy_h"), "md": f.select("NDMI"), "el": f.select("elevation")})
    lc = f.select("lc")
    code = code.where(lc.eq(5), CROP)
    for c in (1, 7, 8, 9):                       # вода, застройка, голый грунт, снег/лёд -> негорючее
        code = code.where(lc.eq(c), NB)
    code = code.updateMask(lc.neq(0).And(lc.neq(10)))     # нет данных / облака
    masks = ee.Image.cat([code.eq(b).rename(f"c{b}") for b in BANDS] + [code.mask().rename("valid")]).unmask(0)
    masks = masks.reproject(crs=GEE_CRS, scale=FEAT_SCALE)
    return masks.reduceResolution(ee.Reducer.mean(), maxPixels=64).multiply(100).round().toUint8()


def tiles():
    try:
        import geopandas as gpd
        from shapely.geometry import box
        kz = gpd.read_file(KZ_BOUNDARY).to_crs("EPSG:6933").union_all()
    except Exception as ex:
        print("граница не прочитана, качаю все тайлы:", ex); kz = None
    out = []
    for r0 in range(0, GH, TILE):
        for c0 in range(0, GW, TILE):
            h, w = min(TILE, GH - r0), min(TILE, GW - c0)
            x0, y1 = X0 + c0 * CELL, Y1 - r0 * CELL
            if kz is not None and not kz.intersects(box(x0, y1 - h * CELL, x0 + w * CELL, y1)):
                continue
            out.append((r0, c0, h, w, x0, y1))
    return out


def pixels(img, h, w, x0, y1):
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": w, "height": h}, "crsWkt": GEE_CRS,
                 "affineTransform": {"scaleX": CELL, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -CELL, "translateY": y1}}})
    return np.stack([arr[n] for n in arr.dtype.names]).astype(np.uint8)


def split4(img, h, w, x0, y1, depth=0):
    """Тайл по частям (рекурсивно, если и часть не влезает в память GEE)."""
    out = np.zeros((len(BANDS) + 1, h, w), np.uint8); h2, w2 = (h + 1) // 2, (w + 1) // 2
    for r, c, hh, ww in ((0, 0, h2, w2), (0, w2, h2, w - w2), (h2, 0, h - h2, w2), (h2, w2, h - h2, w - w2)):
        if hh == 0 or ww == 0:
            continue
        try:
            out[:, r:r + hh, c:c + ww] = pixels(img, hh, ww, x0 + c * CELL, y1 - r * CELL)
        except Exception as ex:
            if "memory limit" in str(ex).lower() and depth < 2:
                out[:, r:r + hh, c:c + ww] = split4(img, hh, ww, x0 + c * CELL, y1 - r * CELL, depth + 1)
            else:
                raise
    return out


def download():
    ee.Initialize(project=os.environ["EE_PROJECT"])
    os.makedirs(TDIR, exist_ok=True)
    img = fuel_image(); tl = tiles(); t0 = time.time()
    print(f"тайлов: {len(tl)}", flush=True)
    for k, (r0, c0, h, w, x0, y1) in enumerate(tl, 1):
        p = os.path.join(TDIR, f"fuel_r{r0:04d}_c{c0:04d}.npy")
        sh, ns = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))   # параллельные процессы: SHARD=i/N
        if os.path.exists(p) or k % ns != sh:
            continue
        wait = 60
        for a in range(5):
            try:
                a3 = pixels(img, h, w, x0, y1)
                np.save(p + ".part.npy", a3); os.replace(p + ".part.npy", p); break
            except Exception as ex:
                msg = str(ex)
                if "Restricted Mode" in msg or ("quota" in msg.lower() and "exceeded" in msg.lower()):
                    print("STOP: квота GEE:", msg[:300], flush=True); return
                if "memory limit" in msg.lower():          # запрос слишком тяжёлый - тайл делится на 4 части
                    print(f"   память -> делю тайл r{r0} c{c0} на 4", flush=True); a3 = split4(img, h, w, x0, y1)
                    np.save(p + ".part.npy", a3); os.replace(p + ".part.npy", p); break
                print(f"   ошибка ({a + 1}/5): {msg[:200]} -> пауза {wait} с", flush=True)
                time.sleep(wait); wait *= 2
        print(f"[{k}/{len(tl)}] r{r0} c{c0} {(time.time() - t0) / 60:.1f} мин", flush=True)


def merge():
    fr = np.zeros((len(BANDS) + 1, GH, GW), np.uint8); n = 0
    for f in sorted(os.listdir(TDIR)):
        if not f.endswith(".npy") or ".part" in f:
            continue
        r0, c0 = int(f[6:10]), int(f[12:16])
        a = np.load(os.path.join(TDIR, f)); fr[:, r0:r0 + a.shape[1], c0:c0 + a.shape[2]] = a; n += 1
    np.savez_compressed(OUT, frac=fr[:-1], codes=np.array(BANDS), valid=fr[-1])
    v = fr[-1] > 50
    print(f"тайлов {n}; ячеек с данными {v.sum()}; средние доли по ним:")
    for b, x in zip(BANDS, fr[:-1]):
        print(f"  {b}: {x[v].mean():.1f} %")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("tiles", "all"): download()
    if cmd in ("merge", "all"): merge()
