"""Фоновый мониторинг пожаров: каждые POLL_MIN минут забирает термоточки FIRMS (VIIRS, 3 спутника), складывает в базу,
отслеживает пожары между опросами и рассылает в Telegram новые и быстро растущие пожары с прогнозом модели.

Заводы, факелы и ТЭЦ отсекаются маской постоянных источников (prod/static_mask.py) - в оповещения не попадают.
Первый опрос на пустой базе только запоминает текущие пожары (без рассылки сотни старых).
Лог: prod/monitor_log.txt (без ключа FIRMS и токена бота).
"""
import os, time, threading, logging, html
import numpy as np, pandas as pd

from .config import CELL

POLL_MIN = 20
MIN_NEW_CELLS = 3            # новый пожар - от 3 клеток (0,75 км²): одиночные точки - обычно палы на день
GROW_CELLS = 8               # «быстро растёт»: +8 клеток (2 км²) и хотя бы вдвое с прошлого оповещения
MAX_ALERTS = 8               # оповещений за один опрос; остальные - одной строкой
KZ_TZ = pd.Timedelta(hours=5)

log = logging.getLogger("monitor")
if not log.handlers:
    h = logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_log.txt"), encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s")); log.addHandler(h); log.setLevel(logging.INFO)


def _kz(t):
    return (pd.Timestamp(t) + KZ_TZ).strftime("%d.%m %H:%M")


class Monitor:
    def __init__(self, live, store, bot):
        self.live, self.store, self.bot = live, store, bot
        self.state = dict(last_poll_utc=None, next_poll_utc=None, n_fires=None, n_industrial=None, n_alerts=0, error=None, polls=0)
        self.fires = []

    def status(self):
        return dict(self.state, bot=self.bot.enabled, subscribers=len(self.bot.subs), bot_error=self.bot.err, poll_min=POLL_MIN)

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="monitor").start()

    def _loop(self):
        time.sleep(5)
        while True:
            try:
                self.poll()
            except Exception as ex:
                self.state["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"; log.info("ошибка опроса: %s", self.state["error"])
            self.state["next_poll_utc"] = str((pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(minutes=POLL_MIN)).floor("min"))
            time.sleep(POLL_MIN * 60)

    # ---------- один опрос ----------
    def poll(self):
        L = self.live
        L._cache.clear()                                          # свежие данные FIRMS
        info, past, _ = L.clusters(None)
        now = pd.Timestamp.utcnow().tz_localize(None)
        fires = [f for f in info["fires"] if not f["flare"]]
        self.fires = fires
        self.state.update(last_poll_utc=str(now.floor("s")), n_fires=len(fires), n_industrial=sum(f["flare"] for f in info["fires"]),
                          error=None, polls=self.state["polls"] + 1)
        first = self.store.meta("initialized") is None
        tracked = self.store.fires(since=now - pd.Timedelta(days=3))
        near = {}                                                 # клетка (с допуском 2 клетки) -> отслеживаемый пожар
        for t in tracked:
            for (r, c) in t["cells"]:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        near.setdefault((r + dr, c + dc), t)
        alerts = []
        for f in fires:
            g = past[past.cl == f["id"]]; cells = set(zip(g.r.tolist(), g.c.tolist()))
            t = next((near[x] for x in cells if x in near), None)
            if t is None:
                t = dict(fid=None, first_seen=f["first_utc"], cells=set(), alerted_cells=0, alerted_at=None)
            t["cells"] |= cells; t.update(last_seen=f["last_utc"], lat=f["lat"], lon=f["lon"], n_cells=len(t["cells"]))
            if first:
                t["alerted_cells"] = t["n_cells"]                    # пожары, которые уже горели до запуска мониторинга
            elif t["alerted_cells"] == 0 and t["n_cells"] >= MIN_NEW_CELLS:
                alerts.append(("new", f, t))
            elif t["alerted_cells"] > 0 and t["n_cells"] - t["alerted_cells"] >= max(GROW_CELLS, t["alerted_cells"]):
                alerts.append(("grow", f, t))
            self.store.save_fire(t)
            for x in cells:
                near.setdefault(x, t)
        if first:
            self.store.meta("initialized", now)
        alerts.sort(key=lambda a: -a[2]["n_cells"])
        sent = 0
        for kind, f, t in alerts[:MAX_ALERTS]:
            msg = self._message(kind, f, t)
            if self.bot.enabled:
                self.bot.broadcast(msg)
            t.update(alerted_cells=t["n_cells"], alerted_at=str(now.floor("s"))); self.store.save_fire(t); sent += 1
        if len(alerts) > MAX_ALERTS and self.bot.enabled:
            self.bot.broadcast(f"И ещё {len(alerts) - MAX_ALERTS} новых или растущих пожаров — список: /fires")
            for _, _, t in alerts[MAX_ALERTS:]:
                t.update(alerted_cells=t["n_cells"], alerted_at=str(now.floor("s"))); self.store.save_fire(t)
        self.state["n_alerts"] += sent
        log.info("опрос: пожаров %d, промышленных %d, отслеживается %d, оповещений %d%s", len(fires), self.state["n_industrial"],
                 len(tracked), sent, " (первый опрос - только запомнили)" if first else "")
        return alerts

    def _message(self, kind, f, t):
        km2 = t["n_cells"] * (CELL / 1000) ** 2
        head = "🔥 <b>Новый пожар</b>" if kind == "new" else f"📈 <b>Пожар растёт</b>: было {t['alerted_cells'] * (CELL / 1000) ** 2:.1f} км²"
        lines = [f"{head} · {km2:.1f} км² по термоточкам",
                 f"📍 {f['lat']:.3f}° с. ш., {f['lon']:.3f}° в. д.",
                 f"Первая точка {_kz(t['first_seen'])}, последняя {_kz(f['last_utc'])} (время Астаны)"]
        try:
            e = self.live.forecast(f["id"], None, "expected", near=(f["lat"], f["lon"]))
            a = self.live.forecast(f["id"], None, "active", near=(f["lat"], f["lon"]))
            H, A = e["horizons"], a["horizons"]
            lines.append(f"Прогноз новой гари: +{H['24h']['pred_km2']:.1f} км² за 24 ч, +{H['72h']['pred_km2']:.1f} км² за 72 ч"
                         f" (если не погаснет: +{A['72h']['pred_km2']:.1f} км²)")
        except Exception as ex:
            lines.append(f"Прогноз не посчитан: {html.escape(type(ex).__name__)}")
        from .tgbot import fire_link
        u = fire_link(f["lat"], f["lon"])
        if u:
            lines.append(f'<a href="{u}">Открыть прогноз на карте</a>')
        return "\n".join(lines)
