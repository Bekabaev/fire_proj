"""Пути и константы прод-движка. Сетка - как у датасета v3 (task1, slices_v3_channels.json)."""
import os, json

ROOT = os.environ.get("FIRE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KZ = os.environ.get("FIRE_KZ", os.path.join(ROOT, "task1_dataset", "data", "ext_kz"))      # национальные слои
MODELS = os.environ.get("FIRE_MODELS", os.path.join(ROOT, "task2_model", "out_v3"))       # wx_best.pt, weather_stats.json, prod_hybrid.json
SCRIPTS_T2 = os.path.join(ROOT, "task2_model", "scripts")
SCRIPTS_T1 = os.path.join(ROOT, "task1_dataset", "scripts")

_meta = json.load(open(os.path.join(KZ, "slices_v3_channels.json"), encoding="utf8"))
CH = _meta["channels"]
CELL = _meta["cell_m"]; N = _meta["size"]; CRS = _meta["crs"]
X0, Y1, GW, GH = (_meta["grid"][k] for k in ("X0", "Y1", "GW", "GH"))
HZ = (12, 24, 48, 72)
