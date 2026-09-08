"""Recent posture history: a score track plus the episodes worth marking.

The in-memory ring is what the live timeline reads: the panel polls it every
few seconds and a query per poll would be waste. Durability is a write-through
to :mod:`posture.store`, which is where the daily stats come from. The panel's
live view did not change when the database arrived, which is the point of
keeping the ring.

Nothing here is invented. Samples are the scores the app actually computed, and
an episode is a real stretch the detector spent in a bad state. When there is
no data yet the timeline is empty rather than populated with plausible-looking
noise.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# One sample every two seconds keeps an hour of track in 1800 points, which is
# more than a timeline a few hundred pixels wide can show and small enough to
# send whole.
SAMPLE_INTERVAL_S = 2.0
DEFAULT_WINDOW_S = 3600.0

# Episodes shorter than this are movement, not posture. Recording them would
# bury the real ones.
MIN_EPISODE_S = 5.0


@dataclass
class Episode:
    """A continuous stretch the detector considered bad posture."""

    started: float
    ended: float | None = None
    offenders: tuple[str, ...] = ()
    worst_score: int = 100
    peak: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return (self.ended if self.ended is not None else time.time()) - self.started

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "ended": self.ended,
            "ongoing": self.ended is None,
            "duration": round(self.duration, 1),
            "offenders": list(self.offenders),
            "worst_score": self.worst_score,
            "peak": {k: round(v, 2) for k, v in self.peak.items()},
        }


class History:
    """Bounded score track and episode log."""

    def __init__(self, window_s: float = DEFAULT_WINDOW_S,
                 interval_s: float = SAMPLE_INTERVAL_S, store=None) -> None:
        self.window_s = window_s
        self.interval_s = interval_s
        # Optional: without one, history is exactly what it was before -- a
        # bounded ring that forgets on restart.
        self.store = store
        self._samples: deque[tuple[float, int | None, str]] = deque(
            maxlen=int(window_s / interval_s) + 2)
        self._episodes: deque[Episode] = deque(maxlen=200)
        self._last_sample = 0.0
        self._score_total = 0
        self._score_count = 0

    # -- recording ---------------------------------------------------------

    def record(self, score: int | None, state: str, verdict: Any,
               now: float | None = None) -> None:
        """Fold one reading in. Cheap enough to call on every sample."""
        now = time.time() if now is None else now
        if now - self._last_sample >= self.interval_s:
            self._last_sample = now
            self._samples.append((now, score, state))
            self._trim(now)
            if score is not None:
                self._score_total += score
                self._score_count += 1
            self._persist(lambda: self.store.add_sample(now, score, state))
        before = len(self._episodes)
        current = self._episodes[-1] if self._episodes else None
        open_before = current.ended is None if current else False
        self._track_episode(score, state, verdict, now)
        self._persist_episode(before, open_before)

    def _persist(self, action) -> None:
        """Write through to the store, never letting it break monitoring.

        A failing database should cost you your stats, not your posture
        detection, so every write is best-effort and logged once at debug.
        """
        if self.store is None:
            return
        try:
            action()
        except Exception:
            log.debug("stats write failed", exc_info=True)

    def _persist_episode(self, before: int, open_before: bool) -> None:
        """Persist whichever episode the last update touched.

        Episodes are written repeatedly while open (so an ongoing one survives
        a crash) and once more when they close. A short episode that gets
        dropped as movement is deleted again rather than left behind.
        """
        if self.store is None:
            return
        current = self._episodes[-1] if self._episodes else None
        if len(self._episodes) < before:
            # The last one was discarded for being too short to be posture.
            return
        if current is None:
            return
        self._persist(lambda: self.store.upsert_episode(
            current.started, current.ended, current.offenders,
            current.worst_score, current.peak))

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _track_episode(self, score: int | None, state: str, verdict: Any,
                       now: float) -> None:
        current = self._episodes[-1] if self._episodes else None
        open_episode = current if current is not None and current.ended is None else None

        if state != "bad":
            if open_episode is not None:
                open_episode.ended = now
                # Too short to be posture rather than movement: drop it, so the
                # timeline shows episodes worth reading.
                if open_episode.duration < MIN_EPISODE_S:
                    self._episodes.pop()
            return

        offenders = tuple(getattr(verdict, "offenders", ()) or ())
        if open_episode is None:
            open_episode = Episode(started=now, offenders=offenders,
                                   worst_score=score if score is not None else 100)
            self._episodes.append(open_episode)
        else:
            open_episode.offenders = tuple(sorted(set(open_episode.offenders) | set(offenders)))
            if score is not None:
                open_episode.worst_score = min(open_episode.worst_score, score)
        for m in getattr(verdict, "metrics", ()):
            if m.flagged and m.excess is not None:
                open_episode.peak[m.key] = max(open_episode.peak.get(m.key, 0.0),
                                               float(m.excess))

    # -- reading -----------------------------------------------------------

    @property
    def session_average(self) -> int | None:
        if not self._score_count:
            return None
        return int(round(self._score_total / self._score_count))

    def samples(self, since: float | None = None) -> list[tuple[float, int | None, str]]:
        if since is None:
            return list(self._samples)
        return [s for s in self._samples if s[0] >= since]

    def episodes(self, limit: int = 40) -> list[Episode]:
        return list(self._episodes)[-limit:]

    def to_dict(self, window_s: float | None = None) -> dict[str, Any]:
        now = time.time()
        since = None if window_s is None else now - window_s
        rows = self.samples(since)
        return {
            "now": now,
            "window": window_s or self.window_s,
            "interval": self.interval_s,
            "samples": [{"t": round(t, 1), "score": s, "state": st} for t, s, st in rows],
            "episodes": [e.to_dict() for e in self.episodes()],
            "session_average": self.session_average,
            "count": len(rows),
        }
