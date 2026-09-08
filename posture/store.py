"""Durable stats in local SQLite. Derived numbers only, never an image.

What is stored
--------------
Three tables, all of them numbers and short strings:

* ``samples``  -- one row every couple of seconds: when, the posture score, and
  which state the detector was in.
* ``episodes`` -- the stretches spent in bad posture, with which metrics were
  at fault and how far past baseline they went.
* ``alerts``   -- when the escalation moved, and to which step.

There are no image columns and there never will be; ``tests/test_invariants.py``
checks the schema for any, so the "frames never reach disk" promise survives the
arrival of a database.

Time attribution
----------------
Daily totals come from walking consecutive samples and attributing the interval
between them to the earlier one's state. Gaps longer than ``max_gap`` are
attributed to nothing at all: the app was closed, or asleep, and counting that
as either good or bad posture would be inventing time you did not spend at the
desk. That is the difference between "you sat well for six hours" and "the app
was shut for six hours".

Threading
---------
One connection, one lock. Writes are a row every two seconds, so contention is
not a concern and a connection pool would be ceremony. WAL keeps the reader
(the panel) from blocking the writer (the capture thread).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    t     REAL PRIMARY KEY,
    score INTEGER,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started     REAL NOT NULL,
    ended       REAL,
    offenders   TEXT NOT NULL DEFAULT '',
    worst_score INTEGER,
    peak        TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    t         REAL NOT NULL,
    level     INTEGER NOT NULL,
    kind      TEXT NOT NULL,
    offenders TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS samples_t ON samples(t);
CREATE INDEX IF NOT EXISTS episodes_started ON episodes(started);
CREATE INDEX IF NOT EXISTS alerts_t ON alerts(t);
"""

# A gap between samples longer than this is unrecorded time, not posture. Six
# seconds is three missed samples at the default rate -- long enough not to
# trip on a slow frame, short enough that a closed laptop is never counted.
DEFAULT_MAX_GAP = 6.0

DEFAULT_PATH = Path.home() / ".posture" / "posture.db"


@dataclass
class DayStats:
    """One day's totals, in the terms the brief asks for."""

    day: str
    good_seconds: float = 0.0
    bad_seconds: float = 0.0
    unknown_seconds: float = 0.0
    away_seconds: float = 0.0
    longest_good_streak: float = 0.0
    alerts: int = 0
    episodes: int = 0
    samples: int = 0
    average_score: int | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    timeline: list[dict] = field(default_factory=list)

    @property
    def measured_seconds(self) -> float:
        """Time actually spent at the desk being measured."""
        return self.good_seconds + self.bad_seconds

    @property
    def good_fraction(self) -> float | None:
        total = self.measured_seconds
        return None if total <= 0 else self.good_seconds / total

    def to_dict(self) -> dict[str, Any]:
        fraction = self.good_fraction
        return {
            "day": self.day,
            "good_seconds": round(self.good_seconds, 1),
            "bad_seconds": round(self.bad_seconds, 1),
            "unknown_seconds": round(self.unknown_seconds, 1),
            "away_seconds": round(self.away_seconds, 1),
            "measured_seconds": round(self.measured_seconds, 1),
            "good_percent": None if fraction is None else round(fraction * 100, 1),
            "longest_good_streak": round(self.longest_good_streak, 1),
            "alerts": self.alerts,
            "episodes": self.episodes,
            "samples": self.samples,
            "average_score": self.average_score,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "timeline": self.timeline,
        }


def day_bounds(day: str) -> tuple[float, float]:
    """Local-midnight epoch bounds for a YYYY-MM-DD string.

    Local rather than UTC because "how did I sit today" is a question about the
    user's day, not about a timezone-neutral one.
    """
    parsed = date.fromisoformat(day)
    start = datetime(parsed.year, parsed.month, parsed.day).timestamp()
    end = (datetime(parsed.year, parsed.month, parsed.day) + timedelta(days=1)).timestamp()
    return start, end


def today() -> str:
    return date.fromtimestamp(time.time()).isoformat()


class Store:
    """SQLite-backed history. Safe to call from any thread."""

    def __init__(self, path: str | Path | None = None, *,
                 max_gap: float = DEFAULT_MAX_GAP) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.max_gap = max_gap
        self._lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            # WAL so the panel reading stats never blocks the capture thread
            # writing them. NORMAL trades a fsync per commit for durability
            # only at the level of "an OS crash may lose the last few seconds",
            # which for posture samples is an easy trade.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema', ?)",
                (str(SCHEMA_VERSION),))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._db.commit()
                self._db.close()
            except Exception:  # pragma: no cover - teardown best effort
                log.debug("closing the stats database failed", exc_info=True)

    # -- writing -----------------------------------------------------------

    def add_sample(self, t: float, score: int | None, state: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO samples(t, score, state) VALUES (?, ?, ?)",
                (float(t), None if score is None else int(score), str(state)))
            self._db.commit()

    def add_alert(self, t: float, level: int, kind: str,
                  offenders: tuple[str, ...] = ()) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO alerts(t, level, kind, offenders) VALUES (?, ?, ?, ?)",
                (float(t), int(level), str(kind), ",".join(offenders)))
            self._db.commit()

    def upsert_episode(self, started: float, ended: float | None,
                       offenders: tuple[str, ...], worst_score: int,
                       peak: dict[str, float]) -> None:
        """Record or update one bad-posture stretch, keyed on its start.

        Keyed on ``started`` rather than an id the caller has to track, because
        an episode is written repeatedly while it is still open and the caller
        (an in-memory ring) has no durable identity to hand back.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM episodes WHERE started = ?", (float(started),)).fetchone()
            payload = (float(started), None if ended is None else float(ended),
                       ",".join(offenders), int(worst_score), json.dumps(peak))
            if row is None:
                self._db.execute(
                    "INSERT INTO episodes(started, ended, offenders, worst_score, peak)"
                    " VALUES (?, ?, ?, ?, ?)", payload)
            else:
                self._db.execute(
                    "UPDATE episodes SET ended = ?, offenders = ?, worst_score = ?,"
                    " peak = ? WHERE id = ?",
                    (payload[1], payload[2], payload[3], payload[4], row["id"]))
            self._db.commit()

    def prune(self, before: float) -> int:
        """Drop everything older than a cutoff. Returns rows removed."""
        with self._lock:
            removed = 0
            for table, column in (("samples", "t"), ("alerts", "t"),
                                  ("episodes", "started")):
                cur = self._db.execute(
                    f"DELETE FROM {table} WHERE {column} < ?", (float(before),))
                removed += cur.rowcount or 0
            self._db.commit()
        return removed

    # -- reading -----------------------------------------------------------

    def _rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def samples(self, since: float, until: float) -> list[tuple[float, int | None, str]]:
        rows = self._rows(
            "SELECT t, score, state FROM samples WHERE t >= ? AND t < ? ORDER BY t",
            (float(since), float(until)))
        return [(r["t"], r["score"], r["state"]) for r in rows]

    def episodes(self, since: float, until: float, limit: int = 200) -> list[dict]:
        rows = self._rows(
            "SELECT * FROM episodes WHERE started >= ? AND started < ?"
            " ORDER BY started DESC LIMIT ?", (float(since), float(until), int(limit)))
        out = []
        for r in rows:
            peak = {}
            try:
                peak = json.loads(r["peak"] or "{}")
            except json.JSONDecodeError:
                pass
            out.append({
                "started": r["started"], "ended": r["ended"],
                "ongoing": r["ended"] is None,
                "duration": round((r["ended"] or time.time()) - r["started"], 1),
                "offenders": [o for o in (r["offenders"] or "").split(",") if o],
                "worst_score": r["worst_score"], "peak": peak,
            })
        return out

    def days(self, limit: int = 30) -> list[str]:
        """Days that have any samples, newest first."""
        rows = self._rows(
            "SELECT DISTINCT date(t, 'unixepoch', 'localtime') AS d"
            " FROM samples ORDER BY d DESC LIMIT ?", (int(limit),))
        return [r["d"] for r in rows if r["d"]]

    def day_stats(self, day: str | None = None, buckets: int = 96) -> DayStats:
        """Totals and a simple timeline for one local day."""
        day = day or today()
        start, end = day_bounds(day)
        stats = DayStats(day=day)
        # Counted before the early return below: alerts and episodes are their
        # own records and a day can hold them with no surviving samples (after
        # a prune, say). Returning zeros for them would misreport the day.
        stats.alerts = len(self._rows(
            "SELECT 1 FROM alerts WHERE t >= ? AND t < ? AND kind = 'escalate'"
            " AND level >= 2", (start, end)))
        stats.episodes = len(self._rows(
            "SELECT 1 FROM episodes WHERE started >= ? AND started < ?", (start, end)))

        rows = self.samples(start, end)
        stats.samples = len(rows)
        if not rows:
            return stats

        stats.first_seen = rows[0][0]
        stats.last_seen = rows[-1][0]

        scored = [s for _t, s, _st in rows if s is not None]
        if scored:
            stats.average_score = int(round(sum(scored) / len(scored)))

        buckets_acc: list[dict] = [
            {"good": 0.0, "bad": 0.0, "other": 0.0, "score_sum": 0, "score_n": 0}
            for _ in range(buckets)
        ]
        span = max(1e-6, end - start)
        streak = 0.0

        for (t1, _s1, state1), (t2, _s2, _st2) in zip(rows, rows[1:]):
            dt = t2 - t1
            if dt <= 0 or dt > self.max_gap:
                # Unrecorded time: the app was closed or the machine asleep.
                # Attributing it to any state would invent time at the desk.
                streak = 0.0
                continue
            if state1 == "good":
                stats.good_seconds += dt
                streak += dt
                stats.longest_good_streak = max(stats.longest_good_streak, streak)
            else:
                streak = 0.0
                if state1 == "bad":
                    stats.bad_seconds += dt
                elif state1 == "away":
                    stats.away_seconds += dt
                else:
                    stats.unknown_seconds += dt

            index = min(buckets - 1, max(0, int((t1 - start) / span * buckets)))
            slot = buckets_acc[index]
            if state1 == "good":
                slot["good"] += dt
            elif state1 == "bad":
                slot["bad"] += dt
            else:
                slot["other"] += dt

        for t, score, _state in rows:
            if score is None:
                continue
            index = min(buckets - 1, max(0, int((t - start) / span * buckets)))
            buckets_acc[index]["score_sum"] += score
            buckets_acc[index]["score_n"] += 1

        bucket_span = span / buckets
        # Every bucket is emitted, including empty ones. Dropping them collapsed
        # a day with one busy hour into a single full-width bar, which lost the
        # thing a day timeline is for: *when*. An empty bucket is a real fact
        # about the day -- you were not at the desk -- and the chart should show
        # the gap rather than close it up.
        for i, slot in enumerate(buckets_acc):
            stats.timeline.append({
                "t": round(start + i * bucket_span, 1),
                "good": round(slot["good"], 1),
                "bad": round(slot["bad"], 1),
                "other": round(slot["other"], 1),
                "score": (None if not slot["score_n"]
                          else int(round(slot["score_sum"] / slot["score_n"]))),
            })

        return stats
