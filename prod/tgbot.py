"""Telegram-бот оповещений о пожарах (Bot API через requests, без сторонних библиотек).

Токен - файл prod/.tg_token (одна строка от @BotFather; в git не идёт, в логи и ответы не попадает).
Нет файла - бот выключен, мониторинг работает без оповещений.
Подписчики - prod/tg_subs.json (id чатов; в git не идёт). Команды: /start - подписаться, /stop - отписаться,
/fires - крупнейшие активные пожары, /status - состояние мониторинга, /help.
"""
import os, json, re, time, threading, html
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
F_TOKEN, F_SUBS = os.path.join(HERE, ".tg_token"), os.path.join(HERE, "tg_subs.json")
F_TUNNEL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "tunnel_log.txt")

HELP = ("Бот прогноза степных пожаров Казахстана.\n"
        "Каждые 20 минут сервер забирает термоточки VIIRS из NASA FIRMS, отбрасывает заводы и факелы "
        "и присылает сюда новые и быстро растущие пожары с прогнозом площади на 24 и 72 ч.\n\n"
        "/fires — крупнейшие активные пожары сейчас\n/status — когда был последний опрос\n/stop — отписаться\n/start — подписаться снова")


def site_url():
    """Адрес сайта: FIRE_SITE_URL или текущий адрес туннеля Cloudflare (меняется при перезапуске туннеля)."""
    u = os.environ.get("FIRE_SITE_URL")
    if u:
        return u.rstrip("/")
    try:
        m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", open(F_TUNNEL, encoding="utf-8", errors="ignore").read())
        return m[-1] if m else None
    except OSError:
        return None


def fire_link(lat, lon):
    u = site_url()
    return f"{u}/#live@{lat:.4f},{lon:.4f}" if u else None


class Bot:
    def __init__(self, status_fn=None, fires_fn=None):
        self.token = open(F_TOKEN).read().strip() if os.path.exists(F_TOKEN) else None
        self.status_fn, self.fires_fn = status_fn, fires_fn
        self._lock = threading.Lock(); self.err = None
        self.subs = set(json.load(open(F_SUBS))) if os.path.exists(F_SUBS) else set()

    @property
    def enabled(self):
        return bool(self.token)

    def _api(self, method, _t=30, **params):
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=params, timeout=_t)
            d = r.json()
        except Exception as ex:                                   # без адреса запроса: в нём токен
            self.err = f"Telegram недоступен: {type(ex).__name__}"; return None
        if not d.get("ok"):
            self.err = f"Telegram: {d.get('description', 'ошибка')}"
            if d.get("error_code") == 403 and "chat_id" in params:   # пользователь заблокировал бота
                self._unsub(params["chat_id"])
            return None
        self.err = None
        return d["result"]

    def _save(self):
        json.dump(sorted(self.subs), open(F_SUBS, "w"))

    def _sub(self, chat):
        with self._lock:
            self.subs.add(chat); self._save()

    def _unsub(self, chat):
        with self._lock:
            self.subs.discard(chat); self._save()

    def send(self, chat, text):
        return self._api("sendMessage", chat_id=chat, text=text, parse_mode="HTML", disable_web_page_preview=True)

    def broadcast(self, text):
        for chat in list(self.subs):
            self.send(chat, text); time.sleep(0.05)

    # ---- приём команд (long polling в отдельном потоке) ----
    def start(self):
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True, name="tgbot").start()

    def _loop(self):
        off = None
        while True:
            res = self._api("getUpdates", _t=70, timeout=50, **({"offset": off} if off else {}))
            if res is None:
                time.sleep(30); continue
            for u in res:
                off = u["update_id"] + 1
                m = u.get("message") or {}
                txt = (m.get("text") or "").strip().split("@")[0].lower(); chat = (m.get("chat") or {}).get("id")
                if chat is None:
                    continue
                try:
                    self._command(chat, txt)
                except Exception as ex:
                    self.send(chat, f"Ошибка: {html.escape(type(ex).__name__)}")

    def _command(self, chat, txt):
        if txt.startswith("/start"):
            self._sub(chat); self.send(chat, "Подписка включена.\n\n" + HELP)
        elif txt.startswith("/stop"):
            self._unsub(chat); self.send(chat, "Подписка выключена. /start — включить снова.")
        elif txt.startswith("/status") and self.status_fn:
            s = self.status_fn()
            self.send(chat, f"Последний опрос FIRMS: {s.get('last_poll_utc') or '—'} UTC\nАктивных пожаров: {s.get('n_fires', '—')} "
                            f"(ещё {s.get('n_industrial', '—')} источников — заводы и факелы, скрыты)\n"
                            f"Сайт: {site_url() or 'туннель не запущен'}" + (f"\nОшибка: {s['error']}" if s.get("error") else ""))
        elif txt.startswith("/fires") and self.fires_fn:
            fs = self.fires_fn()[:10]
            if not fs:
                self.send(chat, "Активных пожаров за сутки нет."); return
            lines = [f"{i + 1}. {f['area_km2']:.1f} км², {f['n_points']} точек · {f['lat']:.2f}°, {f['lon']:.2f}° · последняя {f['last_utc'][5:16]} UTC"
                     + (f" · <a href=\"{fire_link(f['lat'], f['lon'])}\">карта</a>" if fire_link(f['lat'], f['lon']) else "") for i, f in enumerate(fs)]
            self.send(chat, "Крупнейшие активные пожары:\n" + "\n".join(lines))
        else:
            self.send(chat, HELP)
