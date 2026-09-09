"""Per-metric tracks, sustained-deviation accounting, and how much you move.

Three questions the score cannot answer, all of which need the session rather
than the moment:

*How has this one metric behaved?* The score is the worst axis at an instant.
It cannot show that forward head has been creeping up all morning while
everything else held, which is the shape of the problem people actually have.
So each metric keeps its own track.

*For how long, and in what stretches?* Twenty minutes past tolerance in one
sitting is a different thing from the same twenty minutes in forty scattered
half-minutes, and a total alone cannot separate them. Both are recorded, and
runs shorter than :data:`MIN_RUN_S` are excluded from the sustained figures --
reaching for a mug is movement, not posture, and counting it would bury the
stretches that matter.

*How much do you move at all?* Sitting rigidly still in a good position is not
the goal and is arguably worse than drifting between several reasonable ones.
Nothing here rewards stillness: a settled position that changes a few times an
hour is reported as such, without a verdict attached, because what the right
number is depends on the person and the app does not know it.

Everything is derived from readings the app already took. Nothing is modelled,
smoothed a second time, or extrapolated across a gap where nobody was measured.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# How coarse the chart tracks are. A chart a few hundred pixels wide cannot
# show more, and an hour at this interval is 720 points per metric -- small
# enough to send whole on the slow poll rather than paginate.
TRACK_INTERVAL_S = 5.0
TRACK_WINDOW_S = 3600.0

# A run past tolerance shorter than this is movement, not posture. Same
# reasoning and same number as the episode log uses.
MIN_RUN_S = 5.0

# A settled position has to move by this much of its own tolerance, and hold
# there this long, before it counts as having changed. Both are needed: the
# distance alone would count jitter, and the dwell alone would count a slow
# drift back to where it started.
MOVE_RATIO = 1.0
MOVE_DWELL_S = 3.0

# A gap longer than this means nobody was measured. Time inside it is credited
# to nothing: closing the open run rather than letting it swallow the gap is
# the difference between "you slouched for twenty minutes" and "you left".
GAP_S = 5.0

# Variability bands, in tolerances of typical deviation. Descriptive only.
VARIABILITY_BANDS = ((0.35, "low"), (0.9, "moderate"), (float("inf"), "high"))


@dataclass
class MetricTrack:
    """One metric's recent values and its time past tolerance."""

    key: str
    label: str
    unit: str
    baseline: float | None = None
    tolerance: float = 0.0
    points: deque = field(default_factory=lambda: deque(
        maxlen=int(TRACK_WINDOW_S / TRACK_INTERVAL_S) + 2))
    total_over_s: float = 0.0
    longest_over_s: float = 0.0
    runs: int = 0
    measured_s: float = 0.0
    changes: int = 0
    _last_t: float | None = None
    _run_start: float | None = None
    _last_point_t: float = 0.0
    # Deviation in tolerances, kept for the variability summary.
    _ratios: deque = field(default_factory=lambda: deque(maxlen=720))
    # Where this metric was last settled, and since when it has been elsewhere.
    _settled: float | None = None
    _away_since: float | None = None

    def update(self, value: float | None, ratio: float | None, over: bool,
               now: float) -> None:
        elapsed = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        if elapsed > GAP_S:
            self._close_run(now - elapsed)
            elapsed = 0.0
        self._last_t = now
        if value is None:
            self._close_run(now)
            return

        self.measured_s += elapsed
        if ratio is not None:
            self._ratios.append(abs(ratio))
        if over:
            self.total_over_s += elapsed
            if self._run_start is None:
                self._run_start = now
            self.longest_over_s = max(self.longest_over_s, now - self._run_start)
        else:
            self._close_run(now)

        self._track_movement(value, now)
        if now - self._last_point_t >= TRACK_INTERVAL_S:
            self._last_point_t = now
            self.points.append((round(now, 1), round(value, 2), bool(over)))

    def _close_run(self, now: float) -> None:
        if self._run_start is None:
            return
        if now - self._run_start >= MIN_RUN_S:
            self.runs += 1
        self._run_start = None

    def _track_movement(self, value: float, now: float) -> None:
        """Count settled positions, not wobble. See MOVE_RATIO / MOVE_DWELL_S."""
        if self.tolerance <= 0:
            return
        if self._settled is None:
            self._settled = value
            return
        if abs(value - self._settled) < MOVE_RATIO * self.tolerance:
            self._away_since = None
            return
        if self._away_since is None:
            self._away_since = now
        elif now - self._away_since >= MOVE_DWELL_S:
            self.changes += 1
            self._settled = value
            self._away_since = None

    @property
    def typical_ratio(self) -> float | None:
        """Median deviation in tolerances. Median, so one lunge does not
        become the description of a whole morning."""
        if not self._ratios:
            return None
        return sorted(self._ratios)[len(self._ratios) // 2]

    def to_dict(self, session_s: float) -> dict[str, Any]:
        share = (self.total_over_s / session_s) if session_s > 0 else 0.0
        return {
            "key": self.key,
            "label": self.label,
            "unit": self.unit,
            "baseline": None if self.baseline is None else round(self.baseline, 2),
            "tolerance": round(self.tolerance, 3),
            "points": [{"t": t, "v": v, "over": o} for t, v, o in self.points],
            "total_over": round(self.total_over_s, 1),
            "longest_over": round(self.longest_over_s, 1),
            "share_over": round(min(1.0, share), 4),
            "runs": self.runs,
            "changes": self.changes,
            "measured": round(self.measured_s, 1),
        }


class SessionAnalysis:
    """Every metric's track for one run of the app."""

    def __init__(self, started: float | None = None) -> None:
        self.started = time.time() if started is None else started
        self.tracks: dict[str, MetricTrack] = {}
        self._measured_s = 0.0
        self._last_t: float | None = None

    def record(self, verdict: Any, now: float | None = None) -> None:
        """Fold one verdict in. Called on every sample, so it stays cheap."""
        now = time.time() if now is None else now
        state = getattr(verdict, "state", "unknown")
        elapsed = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if state in ("good", "bad") and elapsed <= GAP_S:
            self._measured_s += elapsed

        for m in getattr(verdict, "metrics", ()):
            key = getattr(m, "key", None)
            if key is None:
                continue
            track = self.tracks.get(key)
            if track is None:
                # Duck-typed like the score, so this module stays a leaf and
                # can be exercised against plain stand-ins. A row that carries
                # only a key still gets a track; it just labels itself.
                track = self.tracks[key] = MetricTrack(
                    key=key, label=getattr(m, "label", None) or key,
                    unit=getattr(m, "unit", "") or "")
            track.baseline = getattr(m, "baseline", None)
            track.tolerance = getattr(m, "tolerance", 0.0) or 0.0
            # A reading nobody trusts is not evidence of a posture. It stops
            # this metric's clock rather than counting against you, the same
            # way being out of frame does.
            trusted = not getattr(m, "low_confidence", False)
            ratio = getattr(m, "ratio", None)
            value = getattr(m, "value", None)
            over = bool(trusted and ratio is not None and ratio > 1.0)
            track.update(value if trusted else None, ratio, over, now)

    @property
    def elapsed(self) -> float:
        return max(0.0, (self._last_t or self.started) - self.started)

    def variability(self) -> dict[str, Any]:
        """How much the posture moved, described rather than judged."""
        ratios = [t.typical_ratio for t in self.tracks.values()
                  if t.typical_ratio is not None]
        changes = sum(t.changes for t in self.tracks.values())
        hours = self._measured_s / 3600.0
        typical = max(ratios) if ratios else None
        band = None
        if typical is not None:
            band = next(name for limit, name in VARIABILITY_BANDS if typical < limit)
        return {
            "band": band,
            "typical_ratio": None if typical is None else round(typical, 2),
            "changes": changes,
            # Below a few minutes an hourly rate is an extrapolation from
            # almost nothing, so it is withheld rather than invented.
            "changes_per_hour": (round(changes / hours, 1) if hours >= 180.0 / 3600.0
                                 else None),
            "measured": round(self._measured_s, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "elapsed": round(self.elapsed, 1),
            "measured": round(self._measured_s, 1),
            "interval": TRACK_INTERVAL_S,
            "min_run": MIN_RUN_S,
            "tracks": [t.to_dict(self._measured_s) for t in self.tracks.values()],
            "variability": self.variability(),
        }
