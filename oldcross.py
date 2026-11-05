import subprocess
import numpy as np, rasterio
import config as cfg

OLD  = "/data/arnai_fuelmapping/fuel_map/fuelmap_kz_fbfm40Reclass.tif"
CLU  = cfg.CLUSTERS.format(k=cfg.K_CLUSTERS)
OLDG = cfg.DATA_DIR + "/fbfm40_old_ongrid.tif"

with rasterio.open(CLU) as ref:
    b = ref.bounds
    te = [str(b.left), str(b.bottom), str(b.right), str(b.top)]
    ts = [str(ref.width), str(ref.height)]
subprocess.run(["gdalwarp", "-t_srs", "EPSG:4326", "-te", *te, "-ts", *ts, "-r", "near",
                "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-overwrite", OLD, OLDG], check=True)

with rasterio.open(CLU) as c, rasterio.open(OLDG) as o:
    cc = c.read(1, out_shape=(c.height // 10, c.width // 10)).ravel()
    oo = o.read(1, out_shape=(o.height // 10, o.width // 10)).ravel()

for k in range(cfg.K_CLUSTERS):
    m = cc == k
    tot = int(m.sum())
    if tot == 0:
        continue
    vals, cnt = np.unique(oo[m], return_counts=True)
    top = sorted(zip(vals, cnt), key=lambda x: -x[1])[:4]
    s = ",  ".join(f"{int(v)}: {100*c/tot:.0f}%" for v, c in top)
    print(f"klaster {k:2d} (n={tot:>6d}):  {s}")
