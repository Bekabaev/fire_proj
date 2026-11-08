"""Проверка факта пожара 7.09.2026 (49,11° с. ш., 74,87° в. д.) по снимкам Sentinel-2: dNBR = NBR до - NBR после.
До: медиана 20.08-07.09.2026, после: 10.09-25.09.2026 (маска облаков SCL). Сетка вывода 0,0004° (~30 × 45 м), EPSG:4326.
  EE_PROJECT=mindful-hull-445505-f7 ../.venv/Scripts/python.exe -X utf8 presentation/tools/check_fire_0907_s2.py   (из project_till_nov)
Выход: presentation/fireskz_0907/s2_dnbr.npz (dnbr, rgb_post, rgb_pre, extent), s2_check.json.
"""
import os, io, json
import numpy as np, ee

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIRE = os.environ.get("FIRE", "0907")
D = os.path.join(ROOT, "presentation", f"fireskz_{FIRE}")
ee.Initialize(project=os.environ.get("EE_PROJECT", "mindful-hull-445505-f7"))
LAT0, LON0 = {"0907": (49.12, 74.785), "0906": (50.23, 64.38), "0919": (47.085, 79.225)}[FIRE]
W, S, E, N = LON0 - 0.235, LAT0 - 0.14, LON0 + 0.235, LAT0 + 0.14
RES = 0.0004
aoi = ee.Geometry.Rectangle([W, S, E, N])


def coll(a, b):
    def mask(im):
        scl = im.select("SCL"); ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return im.updateMask(ok).divide(10000)
    c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(a, b)
    return c, c.map(mask).median()


cpre, pre = coll(os.environ.get("PRE_A", "2026-08-20"), os.environ.get("PRE_B", "2026-09-08"))
cpost, post = coll(os.environ.get("POST_A", "2026-09-10"), os.environ.get("POST_B", "2026-09-26"))
info = dict(n_pre=cpre.size().getInfo(), n_post=cpost.size().getInfo(),
            dates_pre=sorted(set(cpre.aggregate_array("system:index").getInfo()))[:20],
            dates_post=sorted(set(cpost.aggregate_array("system:index").getInfo()))[:20])
print(info, flush=True)
nbr = lambda im: im.normalizedDifference(["B8", "B12"])
img = ee.Image.cat([nbr(pre).subtract(nbr(post)).rename("dnbr"), post.select(["B4", "B3", "B2"]), pre.select(["B4", "B3", "B2"])]).toFloat()
nx, ny = int(round((E - W) / RES)), int(round((N - S) / RES))
raw = ee.data.computePixels({"expression": img, "fileFormat": "NPY",
                             "grid": {"dimensions": {"width": nx, "height": ny},
                                      "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": W, "shearY": 0, "scaleY": -RES, "translateY": N},
                                      "crsCode": "EPSG:4326"}})
a = np.load(io.BytesIO(raw))
dnbr = a["dnbr"].astype(np.float32)
rgb_post = np.stack([a["B4"], a["B3"], a["B2"]], -1).astype(np.float32)
rgb_pre = np.stack([a["B4_1"], a["B3_1"], a["B2_1"]], -1).astype(np.float32)
np.savez_compressed(os.path.join(D, os.environ.get("OUT", "s2_dnbr") + ".npz"), dnbr=dnbr, rgb_post=rgb_post, rgb_pre=rgb_pre, extent=np.array([W, S, E, N, RES]))
info.update(shape=list(dnbr.shape), dnbr_p99=float(np.nanpercentile(dnbr, 99)), burned_km2_gt010=float((dnbr > 0.10).sum()) * (RES * 111.32) ** 2 * np.cos(np.radians(49.1)),
            burned_km2_gt020=float((dnbr > 0.20).sum()) * (RES * 111.32) ** 2 * np.cos(np.radians(49.1)))
json.dump(info, open(os.path.join(D, "s2_check.json"), "w"), indent=1)
print(info)
