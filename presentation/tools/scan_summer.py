"""Поиск летних пожаров для сравнения с fires.kz: крупные пожары на дату и их рост за 72 ч по FIRMS.

  ../.venv/Scripts/python.exe -X utf8 presentation/tools/scan_summer.py 2026-08-10 2026-08-21     (из project_till_nov)
"""
import sys
import numpy as np, pandas as pd
from scipy import ndimage
sys.path.insert(0, ".")
from prod.live import Live

L = Live(None)
rows = []
for date in pd.date_range(sys.argv[1], sys.argv[2]):
    date = str(date.date())
    res, past, after = L.clusters(date)
    t = pd.Timestamp(res["t_utc"])
    for f in res["fires"]:
        if f["flare"] or not f["in_kz"] or f["cells"] < 6:
            continue
        g = past[(abs(past.lat - f["lat"]) < 0.3) & (abs(past.lon - f["lon"]) < 0.45)]
        cur = set(zip(g.r, g.c))
        a = after[(after.t <= t + pd.Timedelta(hours=72)) & (abs(after.lat - f["lat"]) < 0.3) & (abs(after.lon - f["lon"]) < 0.45)]
        new = {x for x in zip(a.r, a.c) if x not in cur}
        rows.append(dict(date=date, lat=f["lat"], lon=f["lon"], cells=f["cells"], new72=len(new), last=f["last_utc"]))
    print(date, len(res["fires"]), flush=True)
df = pd.DataFrame(rows).sort_values("new72", ascending=False)
df.to_csv("presentation/fireskz/summer_scan.csv", index=False)
print(df.head(25).to_string())
