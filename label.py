"""Klastery -> FBFM40.  python label.py [rules]
STACK_MODE: metki stavyatsya pravilami po centram klasterov (vysota pologa + produktivnost' + vlazhnost'),
inache - ruchnaya tablica dlya almaty_jetisu."""
import sys
import numpy as np, joblib, rasterio
import config as cfg

# perenos so staroy razmetki (sen. 2026) po blizhayshim centram klasterov, posle peresheta uklona
CLUSTER_TO_FBFM = {
    10: 101, 5: 102, 0: 103, 3: 104, 17: 105, 12: 106, 7: 107,
    14: 108, 8: 109,
    15: 121, 1: 122, 4: 123, 9: 124,
    13: 161, 16: 161, 6: 162,
    2: 99, 11: 99,
}

# porogi - chernovik, otkalibrovan po klasteram almaty_jetisu; proveryat' po etalonnym tochkam
FOREST_H, SHRUB_H, GS_H = 6.0, 3.0, 1.0   # vysota pologa ETH, m
HUMID_NDMI = 0.05                          # vyshe - vlazhnyy tip (GR3/5/6/8, GS3/4, SH3/8)
BARE_INDVI = 0.08                          # nizhe - prakticheski bez rastitel'nosti

def fbfm_rule(f):
    """f: dict priznakov centra klastera v ishodnyh edinicah -> (kod FBFM40, poyasnenie)"""
    h, iv, md, el = f["canopy_h"], f["INDVI"], f["NDMI"], f["elevation"]
    humid = md > HUMID_NDMI
    if iv < BARE_INDVI:
        return 99, "NB9 golyy grunt/skaly"
    if h >= FOREST_H:
        if el >= 1400: return 183, "TL3 hvoynaya podstilka (gornye el'niki)"
        return (162, "TU2 les s podleskom") if iv >= 0.45 else (161, "TU1 redkiy les s travoy")
    if h >= SHRUB_H:
        if humid: return (148, "SH8 vlazhnyy kustarnik, vysokiy zapas") if iv >= 0.5 else (143, "SH3 vlazhnyy kustarnik")
        return (142, "SH2 suhoy kustarnik") if iv >= 0.25 else (141, "SH1 suhoy kustarnik, malyy zapas (saksaul)")
    if h >= GS_H:
        if humid: return (124, "GS4 vlazhn. trava-kustarnik, vys. zapas") if iv >= 0.45 else (123, "GS3 vlazhn. trava-kustarnik")
        return (122, "GS2 suhaya trava-kustarnik") if iv >= 0.25 else (121, "GS1 suhaya trava-kustarnik (polyn')")
    if humid:
        if iv >= 0.5 and el < 800 and md > 0.25: return 109, "GR9 trostniki/poymy"
        if iv >= 0.55: return 108, "GR8 vysokotrav'e"
        if iv >= 0.45: return 106, "GR6 vlazhnye luga"
        if iv >= 0.35: return 105, "GR5 vlazhnye luga, nizk."
        return 103, "GR3 vlazhnaya trava, malyy zapas"
    if iv >= 0.35: return 107, "GR7 suhaya trava, vys. zapas"
    if iv >= 0.25: return 104, "GR4 step'"
    if iv >= 0.15: return 102, "GR2 suhaya step'"
    return 101, "GR1 polupustynya, redkaya trava"

def rules():
    scaler = joblib.load(cfg.SCALER); pca = joblib.load(cfg.PCA_MODEL); km = joblib.load(cfg.KMEANS)
    cen = scaler.inverse_transform(pca.inverse_transform(km.cluster_centers_))
    _, cnt = np.unique(km.labels_, return_counts=True); tot = cnt.sum()
    table = {}
    print("klstr %plosch  INDVI  NDMI  elev  h_pol  -> FBFM")
    for k, row in enumerate(cen):
        f = dict(zip(cfg.FEATURE_NAMES, row))
        code, why = fbfm_rule(f)
        table[k] = code
        print(f"{k:5d} {100*cnt[k]/tot:6.1f} {f['INDVI']:6.2f} {f['NDMI']:5.2f} {f['elevation']:5.0f} {f['canopy_h']:5.1f}  -> {code} {why}")
    return table

if __name__ == "__main__":
    table = rules() if cfg.STACK_MODE else CLUSTER_TO_FBFM
    if len(sys.argv) > 1 and sys.argv[1] == "rules": sys.exit()
    src_path = cfg.CLUSTERS.format(k=cfg.K_CLUSTERS)
    out_path = cfg.OUTPUT_DIR + f"/fueltype_labeled_{cfg.AOI_NAME}_k{cfg.K_CLUSTERS}.tif"
    lut = np.arange(256, dtype="uint8")
    for c, code in table.items():
        lut[c] = code
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        with rasterio.open(out_path, "w", **prof) as dst:
            for _, win in src.block_windows():
                dst.write(lut[src.read(1, window=win)], 1, window=win)
    print("Gotovo:", out_path)
