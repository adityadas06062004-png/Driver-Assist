"""SQLite storage for sessions, events and a once-a-second risk timeline.

Uses only the standard library, so the core package stays free of UI and
heavy dependencies. A new database file is created on first use.
"""

from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

DEFAULT_DB = Path("~/.driver_assist/driver_assist.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    camera INTEGER,
    ear_baseline REAL,
    ear_threshold REAL,
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    level INTEGER,
    risk REAL,
    detail TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    risk REAL,
    level INTEGER,
    ear REAL,
    mar REAL,
    perclos REAL,
    blink_rate REAL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, t);
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id, t);
"""

LEVEL_NAMES = {0: "OK", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}

EXPORT_COLUMNS = {
    "events": ("session_id", "t", "ts", "kind", "level", "risk", "detail"),
    "samples": ("session_id", "t", "risk", "level", "ear", "mar", "perclos", "blink_rate"),
    "sessions": ("id", "started_at", "ended_at", "camera", "ear_baseline", "ear_threshold", "note"),
}


def _num(x, digits: int = 3) -> Optional[float]:
    """NaN is not a useful stored number, so keep it as NULL. Round to keep files small."""
    if x is None:
        return None
    x = float(x)
    return None if x != x else round(x, digits)


class Storage:
    def __init__(self, path: Union[str, Path] = DEFAULT_DB,
                 clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        if str(path) != ":memory:":
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path = p
        self.path = path
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    # writing -----------------------------------------------------------
    def start_session(self, camera: int = 0, note: str = "") -> int:
        cur = self._db.execute(
            "INSERT INTO sessions (started_at, camera, note) VALUES (?, ?, ?)",
            (self._clock(), camera, note))
        self._db.commit()
        return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        self._db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?",
                         (self._clock(), session_id))
        self._db.commit()

    def set_calibration(self, session_id: int, ear_baseline: float, ear_threshold: float) -> None:
        self._db.execute("UPDATE sessions SET ear_baseline = ?, ear_threshold = ? WHERE id = ?",
                         (ear_baseline, ear_threshold, session_id))
        self._db.commit()

    def log_event(self, session_id: int, t: float, kind: str, level: Optional[int] = None,
                  risk: Optional[float] = None, detail: str = "") -> None:
        self._db.execute(
            "INSERT INTO events (session_id, t, ts, kind, level, risk, detail) VALUES (?,?,?,?,?,?,?)",
            (session_id, round(t, 2), round(self._clock(), 3), kind, level, _num(risk, 1), detail))
        self._db.commit()

    def log_sample(self, session_id: int, t: float, risk: float, level: int, ear: float = None,
                   mar: float = None, perclos: float = None, blink_rate: float = None) -> None:
        self._db.execute(
            "INSERT INTO samples (session_id, t, risk, level, ear, mar, perclos, blink_rate) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (session_id, round(t, 2), _num(risk, 1), level, _num(ear, 3), _num(mar, 3),
             _num(perclos, 3), _num(blink_rate, 1)))
        self._db.commit()

    def delete_session(self, session_id: int) -> None:
        self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._db.commit()

    # reading -----------------------------------------------------------
    def sessions(self) -> List[dict]:
        rows = self._db.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def events(self, session_id: Optional[int] = None) -> List[dict]:
        return self._rows("events", session_id)

    def samples(self, session_id: Optional[int] = None) -> List[dict]:
        return self._rows("samples", session_id)

    def _rows(self, table: str, session_id: Optional[int]) -> List[dict]:
        if session_id is None:
            rows = self._db.execute(f"SELECT * FROM {table} ORDER BY session_id, t, id").fetchall()
        else:
            rows = self._db.execute(f"SELECT * FROM {table} WHERE session_id = ? ORDER BY t, id",
                                    (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self, session_id: int) -> Dict[str, object]:
        """Headline numbers for one session. Time per level counts one sample as one second."""
        s = self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if s is None:
            raise KeyError(f"no session {session_id}")
        agg = self._db.execute(
            "SELECT COUNT(*) AS n, AVG(risk) AS avg_risk, MAX(risk) AS max_risk, MAX(t) AS last_t "
            "FROM samples WHERE session_id = ?", (session_id,)).fetchone()
        per_level = {name: 0 for name in LEVEL_NAMES.values()}
        for r in self._db.execute(
                "SELECT level, COUNT(*) AS n FROM samples WHERE session_id = ? GROUP BY level",
                (session_id,)):
            per_level[LEVEL_NAMES.get(r["level"], str(r["level"]))] = r["n"]
        counts = {r["kind"]: r["n"] for r in self._db.execute(
            "SELECT kind, COUNT(*) AS n FROM events WHERE session_id = ? GROUP BY kind", (session_id,))}
        alerts = {name: 0 for name in ("CAUTION", "WARNING", "CRITICAL")}
        for r in self._db.execute(
                "SELECT level, COUNT(*) AS n FROM events WHERE session_id = ? AND kind = 'alert' "
                "GROUP BY level", (session_id,)):
            name = LEVEL_NAMES.get(r["level"])
            if name in alerts:
                alerts[name] = r["n"]
        end = s["ended_at"] if s["ended_at"] is not None else self._clock()
        return {
            "id": session_id,
            "duration_s": max(end - s["started_at"], 0.0),
            "samples": agg["n"],
            "avg_risk": agg["avg_risk"] or 0.0,
            "max_risk": agg["max_risk"] or 0.0,
            "seconds_at_level": per_level,
            "alerts": alerts,
            "yawns": counts.get("yawn", 0),
            "nods": counts.get("nod", 0),
            "issues": counts.get("issue", 0),
            "phone_calls": counts.get("phone_start", 0),
        }

    # exporting ---------------------------------------------------------
    def export_csv(self, path: Union[str, Path], what: str = "events",
                   session_id: Optional[int] = None) -> int:
        """Write a table to CSV. Returns the number of data rows written."""
        if what not in EXPORT_COLUMNS:
            raise ValueError(f"what must be one of {sorted(EXPORT_COLUMNS)}")
        cols = EXPORT_COLUMNS[what]
        if what == "sessions":
            rows = list(reversed(self.sessions()))
        elif what == "events":
            rows = self.events(session_id)
        else:
            rows = self.samples(session_id)
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow(["" if r[c] is None else r[c] for c in cols])
        return len(rows)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
