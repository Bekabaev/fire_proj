import os

STACK_MODE = True   # True -> ves Kazahstan iz steka taylov (acquire.py stack), False -> starye fayly almaty_jetisu

AOI_BBOX = [73.2, 42.5, 81.5, 47.0]     # запад, юг, восток, север
AOI_NAME = "kz" if STACK_MODE else "almaty_jetisu"

BASE = os.path.dirname(os.path.abspath(__file__))   # папка, где лежит config.py
DATA_DIR   = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output")

def _d(n): return os.path.join(DATA_DIR, f"{n}_{AOI_NAME}.tif")
def _o(n): return os.path.join(OUTPUT_DIR, n)

S2_RAW        = _d("S2_20m_2022")
S2_INDICES    = _d("S2_indices_2022")
DEM_RAW       = _d("DEM_30m")
DEM_TM        = _d("DEM_30m_tm")
SLOPE_RAW     = _d("slope_30m")
ASPECT_RAW    = _d("aspect_30m")
DEM_ALIGNED   = _d("DEM_20m_aligned")
SLOPE_ALIGNED = _d("slope_20m_aligned")
ESRI_ALIGNED  = _d("ESRI_LC_2022_aligned")

# метрическая проекция для уклона: TM с центром в AOI (область лежит в двух зонах UTM, 43 и 44)
TM_SRS = f"+proj=tmerc +lat_0=0 +lon_0={(AOI_BBOX[0] + AOI_BBOX[2]) / 2} +k=1 +x_0=500000 +y_0=0 +datum=WGS84 +units=m"

# ESRI LULC -> код FBFM40 NB (негорючее); эти пиксели не кластеризуются
ESRI_NONFUEL = {1: 98,   # вода -> NB8
                5: 93,   # пашня -> NB3
                7: 91,   # застройка -> NB1
                8: 99,   # голый грунт -> NB9
                9: 92}   # снег/лёд -> NB2
ESRI_NODATA = [0, 10]    # нет данных, облака -> 255

X_SAMPLE  = _o(f"X_sample_{AOI_NAME}.npy")
SCALER    = _o(f"scaler_{AOI_NAME}.joblib")
PCA_MODEL = _o(f"pca_{AOI_NAME}.joblib")
KMEANS    = _o(f"kmeans_{AOI_NAME}.joblib")
CLUSTERS  = _o(f"clusters_{AOI_NAME}_k{{k}}.tif")

GEE_PROJECT = "fuel-map-507306"
DATE_START, DATE_END, CLOUD_MAX = "2022-06-01", "2022-08-31", 40

SCALE, BLOCK = 20, 2048
SAMPLE_TARGET, PCA_VAR, K_CLUSTERS, RANDOM_SEED = 9_000_000, 0.95, 18, 42
FEATURE_NAMES = ["NDVI", "NDMI", "SAVI", "NBR", "elevation", "slope", "VV", "VH", "INDVI", "NIRv", "NDVI_amp"]

# --- ves Kazahstan: odin stek priznakov na tayl 2°x2°, int16 + ZSTD ---
TILE_DEG  = 2.0
TILE_PX   = 11132                       # 2° / 11132 = 0.00017966° ~ 20 m; odna setka dlya vseh taylov, styki bez sdviga
RES       = TILE_DEG / TILE_PX
KZ_GRID   = [46.0, 40.0, 88.0, 56.0]    # запад, юг, восток, север (кратно TILE_DEG)
STACK_DIR = os.path.join(DATA_DIR, "stack_kz")
STACK_VRT = os.path.join(DATA_DIR, "stack_kz.vrt")
NODATA_I16 = -32768
# kanal: masshtab, v fayle round(znachenie * masshtab); poslednim - klass ESRI (maska negoryuchego)
STACK_BANDS = {"NDVI": 1000, "NDMI": 1000, "SAVI": 1000, "NBR": 1000, "elevation": 1, "slope": 100,
               "VV": 100, "VH": 100, "INDVI": 1000, "NIRv": 1000, "NDVI_amp": 1000, "canopy_h": 1, "lc": 1}
if STACK_MODE:
    FEATURE_NAMES = [b for b in STACK_BANDS if b != "lc"]
    K_CLUSTERS = 40                     # vsya strana raznoobraznee odnoy oblasti; metki stavyatsya pravilami (label.py)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STACK_DIR, exist_ok=True)
