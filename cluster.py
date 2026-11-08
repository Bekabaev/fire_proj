"""Klasterizaciya: vyborka -> PCA/KMeans  """
import sys
import numpy as np, joblib, rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import config as cfg
from io_utils import open_sources, close_sources, grid_shape, iter_windows, read_window, ref_profile, open_lc, valid_fraction

NONFUEL_CODE = cfg.ESRI_NONFUEL
EXCLUDE_ESRI = list(NONFUEL_CODE) + cfg.ESRI_NODATA

def sample():
    srcs = open_sources(); m, lb = open_lc(srcs)
    H, W = grid_shape(srcs)
    assert m.shape == (H, W), "maska ne na toj setke!"
    p = cfg.SAMPLE_TARGET / (H * W)
    if cfg.STACK_MODE:   # bol'she poloviny ramki KZ - za granicey, inache vyborka vyshla by vdvoe men'she
        frac = valid_fraction(srcs); p /= max(frac, 1e-3)
        print(f"dolya dannyh v ramke: {100*frac:.1f}% -> p = {p:.5f}")
    n_win = len(range(0, H, cfg.BLOCK)) * len(range(0, W, cfg.BLOCK))
    rng = np.random.default_rng(cfg.RANDOM_SEED); samples = []; done = 0
    for win in iter_windows(H, W, cfg.BLOCK):
        flat, valid = read_window(srcs, win)
        mask = m.read(lb, window=win).reshape(-1)
        valid = valid & ~np.isin(mask, EXCLUDE_ESRI)
        fl = flat[valid]
        if len(fl):
            take = rng.random(len(fl)) < p
            if take.any(): samples.append(fl[take])
        done += 1
        if done % 100 == 0: print(f"okon: {done}/{n_win}")
    X = np.concatenate(samples, axis=0)
    close_sources(srcs); m.close()
    np.save(cfg.X_SAMPLE, X)
    print("Sohraneno:", cfg.X_SAMPLE, "| forma:", X.shape, "| NaN:", int(np.isnan(X).sum()))

def train():
    X = np.load(cfg.X_SAMPLE)
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=cfg.PCA_VAR).fit(scaler.transform(X))
    Xp = pca.transform(scaler.transform(X))
    print("Komponent:", Xp.shape[1], "| dispersiya:", pca.explained_variance_ratio_.round(3))
    km = KMeans(n_clusters=cfg.K_CLUSTERS, random_state=cfg.RANDOM_SEED, n_init=10).fit(Xp)
    joblib.dump(scaler, cfg.SCALER); joblib.dump(pca, cfg.PCA_MODEL); joblib.dump(km, cfg.KMEANS)
    l, c = np.unique(km.labels_, return_counts=True)
    for a, b in zip(l, c): print(f"klaster {a}: {b} ({100*b/len(km.labels_):.1f}%)")
    print("Modeli sohraneny")

def rollout():
    scaler = joblib.load(cfg.SCALER); pca = joblib.load(cfg.PCA_MODEL); km = joblib.load(cfg.KMEANS)
    srcs = open_sources(); m, lb = open_lc(srcs)
    H, W = grid_shape(srcs)
    prof = ref_profile(srcs)
    prof.update(count=1, dtype="uint8", nodata=255, compress="zstd",
                BIGTIFF="YES", tiled=True, blockxsize=256, blockysize=256)
    out = cfg.CLUSTERS.format(k=cfg.K_CLUSTERS)
    n_win = len(range(0, H, cfg.BLOCK)) * len(range(0, W, cfg.BLOCK)); done = 0
    with rasterio.open(out, "w", **prof) as dst:
        for win in iter_windows(H, W, cfg.BLOCK):
            done += 1
            if done % 200 == 0: print(f"okon: {done}/{n_win}", flush=True)
            flat, valid = read_window(srcs, win)
            esri = m.read(lb, window=win).reshape(-1)
            o = np.full(flat.shape[0], 255, dtype="uint8")
            fuel = valid & ~np.isin(esri, EXCLUDE_ESRI)
            if fuel.any():
                o[fuel] = km.predict(pca.transform(scaler.transform(flat[fuel]))).astype("uint8")
            for ev, nb in NONFUEL_CODE.items(): o[esri == ev] = nb
            dst.write(o.reshape(win.height, win.width), 1, window=win)
    close_sources(srcs); m.close()
    print("Gotovo:", out)

def fingerprint():
    scaler = joblib.load(cfg.SCALER); pca = joblib.load(cfg.PCA_MODEL); km = joblib.load(cfg.KMEANS)
    cen = scaler.inverse_transform(pca.inverse_transform(km.cluster_centers_))
    _, c = np.unique(km.labels_, return_counts=True); tot = c.sum()
    print("klstr  %plosch | " + " ".join(f"{n:>9s}" for n in cfg.FEATURE_NAMES))
    for k in range(cen.shape[0]):
        print(f"{k:5d} {100*c[k]/tot:6.1f} | " + " ".join(f"{v:9.2f}" for v in cen[k]))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"sample": sample, "train": train, "rollout": rollout, "fingerprint": fingerprint}
    if cmd == "all":
        for f in [sample, train, rollout, fingerprint]: f()
    elif cmd in fns: fns[cmd]()
    else: print("sample | train | rollout | fingerprint | all")
