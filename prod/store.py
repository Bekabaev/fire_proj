"""SQLite-хранилище для мониторинга: термоточки FIRMS (история дольше 3 суток API) и отслеживаемые пожары.
Файл prod/live.db (в git не идёт). Переживает перезапуск сервера и отключение питания."""
import os, json, sqlite3, threading
import pandas as pd

F_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.db")


class Store:
    def __init__(self, path=F_DB):
        self.path = path; self._lock = threading.Lock()
        with self._con() as c:
            c.execute("CREATE TABLE IF NOT EXISTS points (lon REAL, lat REAL, t TEXT, frp REAL, r INT, c INT, PRIMARY KEY (lon, lat, t))")
            c.execute("CREATE INDEX IF NOT EXISTS points_t ON points (t)")
            c.execute("CREATE TABLE IF NOT EXISTS fires (fid INTEGER PRIMARY KEY AUTOINCREMENT, first_seen TEXT, last_seen TEXT, "
                      "lat REAL, lon REAL, cells TEXT, n_cells INT, alerted_cells INT, alerted_at TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")

    def _con(self):
        c = sqlite3.connect(self.path, timeout=30); c.execute("PRAGMA journal_mode=WAL"); return c

    # ---- точки ----
    def add_points(self, d):
        if not len(d):
            return
        rows = list(zip(d.lon.round(5), d.lat.round(5), d.t.astype(str), d.frp.astype(float), d.r.astype(int), d.c.astype(int)))
        with self._lock, self._con() as c:
            c.executemany("INSERT OR IGNORE INTO points VALUES (?,?,?,?,?,?)", rows)

    def points_since(self, t0):
        with self._lock, self._con() as c:
            d = pd.read_sql_query("SELECT * FROM points WHERE t >= ?", c, params=(str(t0),))
        d["t"] = pd.to_datetime(d.t)
        return d

    # ---- пожары ----
    def fires(self, since):
        with self._lock, self._con() as c:
            rows = c.execute("SELECT fid, first_seen, last_seen, lat, lon, cells, n_cells, alerted_cells, alerted_at FROM fires WHERE last_seen >= ?",
                             (str(since),)).fetchall()
        return [dict(fid=r[0], first_seen=r[1], last_seen=r[2], lat=r[3], lon=r[4], cells={tuple(x) for x in json.loads(r[5])},
                     n_cells=r[6], alerted_cells=r[7], alerted_at=r[8]) for r in rows]

    def save_fire(self, f):
        cells = json.dumps(sorted([list(x) for x in f["cells"]]))
        with self._lock, self._con() as c:
            if f.get("fid"):
                c.execute("UPDATE fires SET last_seen=?, lat=?, lon=?, cells=?, n_cells=?, alerted_cells=?, alerted_at=? WHERE fid=?",
                          (f["last_seen"], f["lat"], f["lon"], cells, f["n_cells"], f["alerted_cells"], f["alerted_at"], f["fid"]))
            else:
                cur = c.execute("INSERT INTO fires (first_seen, last_seen, lat, lon, cells, n_cells, alerted_cells, alerted_at) VALUES (?,?,?,?,?,?,?,?)",
                                (f["first_seen"], f["last_seen"], f["lat"], f["lon"], cells, f["n_cells"], f["alerted_cells"], f["alerted_at"]))
                f["fid"] = cur.lastrowid
        return f

    def meta(self, k, v=None):
        with self._lock, self._con() as c:
            if v is None:
                r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone(); return r[0] if r else None
            c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))
