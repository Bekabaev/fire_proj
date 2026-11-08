"""HTTP API прод-движка для сайта (FastAPI).

Запуск (из project_till_nov):  ../.venv/Scripts/python.exe -X utf8 -m uvicorn prod.server:app --host 0.0.0.0 --port 8000
  GET  /health
  POST /predict   {"lon":..,"lat":..,"radius_km":1, "t_utc":"2026-07-20T06:00", "hotspots":[[lon,lat,часов_до_t,frp],..], "weather":"gfs"}
                  или {"burned":[[lon,lat],..], ...} - прогноз от текущего состояния пожара на 12/24/48/72 ч
                  mode: "expected" (по умолчанию; ожидаемый исход) | "active" (если пожар не погаснет)
  POST /simulate  {"lon":..,"lat":..,"hours":72, "weather":"gfs" | {"winds":[скорость м/с, откуда °], "T":30, "RH":20}}
Ответ: контуры GeoJSON по горизонтам, площади, флаги данных; grids=true - ещё сетки вероятности и времени прихода (u8, 96 × 96)
и углы окна (lon/lat) для наложения растра на карту.
"""
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import HZ
from .engine import FireEngine

app = FastAPI(title="Fire spread KZ", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ENGINE = None


def engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = FireEngine()
    return ENGINE


class PredictReq(BaseModel):
    lon: float | None = None
    lat: float | None = None
    radius_km: float = 1.0
    burned: list[list[float]] | None = None
    hotspots: list[list[float]] | None = None
    t_utc: str | None = None
    weather: str | dict = "gfs"
    age_h: float = 6.0
    mode: str = "expected"
    grids: bool = False


class SimReq(BaseModel):
    lon: float
    lat: float
    hours: int = 72
    radius_km: float = 0.5
    t_utc: str | None = None
    weather: str | dict = "gfs"
    mode: str = "active"
    grids: bool = False


def _pack(r, grids):
    o = {k: r[k] for k in ("t_utc", "window", "flags", "area_km2", "weather")}
    o["contours"] = {h: dict(type="FeatureCollection", features=f) for h, f in r["contours"].items()}
    if grids:
        o["grids"] = dict(prob={f"{h}h": np.round(r["prob"][k] * 255).astype(np.uint8).tolist() for k, h in enumerate(HZ)},
                          arrival_h=np.minimum(r["arrival_h"], 255).astype(np.uint8).tolist())
    return o


def _check(lon, lat):
    if lon is None or lat is None or not (46 <= lon <= 88 and 40 <= lat <= 56):
        raise HTTPException(422, "точка вне Казахстана (46–88° в. д., 40–56° с. ш.)")


@app.get("/")
def index():                                                   # демо-страница: карта, точка возгорания, кадры симуляции
    return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))


DEMO = None


def demo():
    global DEMO
    if DEMO is None:
        from .demo import Demo
        DEMO = Demo(engine())
    return DEMO


@app.get("/demo/cases")
def demo_cases():                                              # известные пожары теста 2023-24 (полные входы, как офлайн)
    d = demo(); return dict(pooled_F1=d.meta["pooled_F1"], n=d.meta["n"],
                            cases=[dict(id=i, title=c["title"], F1=c["F1"], lon=c["lon"], lat=c["lat"]) for i, c in enumerate(d.rows)])


@app.get("/demo/run/{i}")
def demo_run(i: int, model: str = "hybrid"):
    d = demo()
    if not 0 <= i < len(d.rows):
        raise HTTPException(404, "нет такого пожара")
    return d.run(i, model if model in ("phys", "unet") else "hybrid")


LIVE = None


STORE = BOT = MON = None


def live():
    global LIVE, STORE
    if LIVE is None:
        from .live import Live
        from .store import Store
        STORE = STORE or Store()
        LIVE = Live(engine(), STORE)
    return LIVE


@app.on_event("startup")
def _start_monitor():                                          # фоновый опрос FIRMS + Telegram-бот (FIRE_MONITOR=0 - выключить)
    global BOT, MON
    if os.environ.get("FIRE_MONITOR", "1") == "0":
        return
    from .monitor import Monitor
    from .tgbot import Bot
    L = live()
    BOT = Bot(status_fn=lambda: MON.status(), fires_fn=lambda: MON.fires,
              detail_fn=lambda f: MON._message("info", f, dict(first_seen=f["first_utc"], n_cells=f["cells"], alerted_cells=0)))
    MON = Monitor(L, STORE, BOT)
    MON.start(); BOT.start()


@app.get("/live/status")
def live_status():                                             # состояние мониторинга (без секретов)
    if MON is None:
        return dict(monitor=False)
    from .tgbot import site_url
    return dict(MON.status(), monitor=True, bot_username=BOT.username if BOT else None, site=site_url())


def _date(date):
    if not date:
        return None
    try:
        d = pd.Timestamp(date)
    except Exception:
        raise HTTPException(422, "дата: ГГГГ-ММ-ДД")
    now = pd.Timestamp.utcnow().tz_localize(None)
    if not now - pd.Timedelta(days=60) <= d <= now:
        raise HTTPException(422, "дата: за последние 60 дней (оперативные данные FIRMS)")
    return str(d.date())


@app.get("/live/fires")
def live_fires(date: str | None = None):                      # активные пожары по термоточкам FIRMS (сейчас или на дату)
    try:
        return live().clusters(_date(date))[0]
    except RuntimeError as ex:
        raise HTTPException(502, str(ex))


@app.get("/live/forecast/{cid}")
def live_forecast(cid: int, date: str | None = None, mode: str = "expected", lat: float | None = None, lon: float | None = None):
    if mode not in ("expected", "active"):
        raise HTTPException(422, "mode: expected | active")
    try:
        return live().forecast(cid, _date(date), mode, near=(lat, lon) if lat is not None and lon is not None else None)
    except KeyError as ex:
        raise HTTPException(404, str(ex))
    except RuntimeError as ex:
        raise HTTPException(502, str(ex))


@app.get("/health")
def health():
    e = engine()
    return dict(ok=True, variant=e.variant, device=e.dev, horizons_h=list(HZ))


@app.post("/predict")
def predict(q: PredictReq):
    if q.burned:
        _check(*np.mean(q.burned, 0))
    else:
        _check(q.lon, q.lat)
    try:
        r = engine().predict(burned_lonlat=q.burned, lon=q.lon, lat=q.lat, t_utc=q.t_utc, radius_km=q.radius_km,
                             hotspots=q.hotspots, weather=q.weather, age_h=q.age_h, mode=q.mode)
    except Exception as ex:                                    # сеть Open-Meteo и т. п.
        raise HTTPException(502, f"{type(ex).__name__}: {ex}")
    return _pack(r, q.grids)


@app.post("/simulate")
def simulate(q: SimReq):
    _check(q.lon, q.lat)
    if not 1 <= q.hours <= 240:
        raise HTTPException(422, "hours: 1–240")
    try:
        r = engine().simulate(q.lon, q.lat, t_utc=q.t_utc, hours=q.hours, radius_km=q.radius_km, weather=q.weather, mode=q.mode)
    except Exception as ex:
        raise HTTPException(502, f"{type(ex).__name__}: {ex}")
    out = dict(start=r["start"], frames=[dict(hour=f["hour"], area_km2=f["area_km2"], edge_clipped=f.get("edge_clipped", False),
                                              contour=dict(type="FeatureCollection", features=f["geojson"])) for f in r["frames"]])
    out["first_pass"] = _pack(r["first_pass"], q.grids)
    return out
