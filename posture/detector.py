"""Deciding whether posture is actually bad, from a stream of noisy samples.

Three ideas do the work here.

**A rolling window, not an instant reading.** Pose landmarks jitter, and people
move. A metric being out of tolerance right now means nothing; being out for
most of the last minute means something. So each metric keeps a time window of
verdicts and posture counts as bad only once the out-of-tolerance fraction
passes a threshold.

**Hysteresis on the tolerance, not just the fraction.** Sitting exactly at the
boundary would otherwise flip the state every few samples. Once a metric is
flagged, it is judged against a *tighter* tolerance until it clears, so
recovering takes a real change in posture rather than a lucky sample.

**Presence, and the difference between kinds of silence.** An empty chair is
not bad posture, and neither is a frame where the hip landmark dropped below
the visibility threshold. Both suspend judgement, but they are different states
and the UI has to be able to say which one it is. Samples that arrive while
suspended are not counted as good either -- they simply do not enter the
window, so the window stays a record of what was actually observed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from . import metrics as met
from .calibration import Baseline

# The four things the rest of the app can be told about your posture.
GOOD = "good"
BAD = "bad"
UNKNOWN = "unknown"  # someone is there, but the landmarks we need are not
AWAY = "away"        # nobody there; judgement suspended

STATE_LABELS = {
    GOOD: "Posture OK",
    BAD: "Posture needs fixing",
    UNKNOWN: "Cannot see you properly",
    AWAY: "Nobody at the desk",
}


@dataclass
class DetectionSettings:
    """Everything about how strict the detector is."""

    window_seconds: float = 60.0
    # Fraction of the window a metric must be out of tolerance before posture
    # counts as bad.
    bad_fraction: float = 0.70
    # Once flagged, the metric is judged against tolerance * this until it
    # clears. Below 1.0 means "tighter to exit than to enter", which is the
    # hysteresis that stops boundary flapping.
    exit_ratio: float = 0.70
    # How long without a person before judgement is suspended.
    absence_seconds: float = 15.0
    # Multiplier on your calibrated spread when deriving a tolerance.
    tolerance_multiplier: float = 3.0
    # Per-metric hard overrides, by metric key.
    overrides: dict[str, float] = field(default_factory=dict)
    # A window with less than this fraction of its nominal sample count is too
    # sparse to judge; prevents a verdict from three samples after a restart.
    min_window_fill: float = 0.30


@dataclass(frozen=True)
class MetricVerdict:
    """One metric's standing over the window."""

    key: str
    label: str
    unit: str
    value: float | None
    baseline: float | None
    tolerance: float
    excess: float | None       # how far past baseline, in the bad direction
    ratio: float | None        # excess / tolerance; 1.0 is exactly at the limit
    out_fraction: float        # share of the window spent out of tolerance
    flagged: bool
    samples: int
    # Where the tolerance came from, so the panel can explain itself rather
    # than showing a bare number nobody can account for.
    floor: float = 0.0
    ceiling: float = 0.0
    derived: float | None = None   # multiplier * spread, before clamping
    source: str = "learned"        # learned | floored | capped | manual

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "unit": self.unit,
            "value": None if self.value is None else round(self.value, 2),
            "baseline": None if self.baseline is None else round(self.baseline, 2),
            "tolerance": round(self.tolerance, 3),
            "excess": None if self.excess is None else round(self.excess, 3),
            "ratio": None if self.ratio is None else round(self.ratio, 3),
            "out_fraction": round(self.out_fraction, 3),
            "flagged": self.flagged,
            "samples": self.samples,
            "floor": round(self.floor, 3),
            "ceiling": round(self.ceiling, 3),
            "derived": None if self.derived is None else round(self.derived, 3),
            "source": self.source,
        }


@dataclass(frozen=True)
class PostureVerdict:
    """The whole picture at one instant."""

    state: str
    since: float
    metrics: tuple[MetricVerdict, ...] = ()
    offenders: tuple[str, ...] = ()
    worst_ratio: float = 0.0
    calibrated: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        return STATE_LABELS.get(self.state, self.state)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "label": self.label,
            "reason": self.reason,
            "offenders": list(self.offenders),
            "worst_ratio": round(self.worst_ratio, 3),
            "calibrated": self.calibrated,
            "metrics": [m.to_dict() for m in self.metrics],
        }


def choose_rate(state: str, worst_ratio: float, *, peak_hz: float,
                idle_hz: float, away_hz: float, active_ratio: float = 0.5) -> float:
    """Pick a sampling rate from the current verdict.

    Running pose inference five times a second all day costs more CPU than this
    app is allowed. It only needs to be that responsive when a decision is
    close, so the rate follows the state:

    * an empty chair barely needs watching, just often enough to notice you
      coming back;
    * comfortably good posture needs enough samples to fill the window, not to
      resolve the instant a metric moves;
    * anything near a tolerance, or already bad, runs at full rate -- including
      while bad, because the hold-to-clear mechanic in phase 3 has to see you
      recover promptly.

    The threshold is a fraction of tolerance rather than the tolerance itself,
    so the rate rises *before* the verdict changes and the window is already
    dense by the time it matters.
    """
    if state == AWAY:
        return away_hz
    if state in (BAD, UNKNOWN):
        return peak_hz
    return peak_hz if worst_ratio >= active_ratio else idle_hz


class MetricWindow:
    """Rolling record of whether one metric was out of tolerance.

    Stores ``(timestamp, excess)`` rather than a bare boolean so the
    out-of-tolerance fraction can be recomputed against the tighter exit
    tolerance without keeping a second window.
    """

    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = window_seconds
        self._points: deque[tuple[float, float]] = deque()

    def add(self, t: float, excess: float) -> None:
        self._points.append((t, excess))
        self.trim(t)

    def trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def __len__(self) -> int:
        return len(self._points)

    @property
    def span(self) -> float:
        """Seconds between the oldest and newest retained sample."""
        if len(self._points) < 2:
            return 0.0
        return self._points[-1][0] - self._points[0][0]

    @property
    def latest(self) -> float | None:
        """Most recent excess, or None if the window is empty."""
        return self._points[-1][1] if self._points else None

    @property
    def last_t(self) -> float | None:
        """Timestamp of the most recent sample, or None if empty."""
        return self._points[-1][0] if self._points else None

    def out_fraction(self, tolerance: float) -> float:
        """Share of retained samples whose excess exceeded ``tolerance``."""
        if not self._points:
            return 0.0
        over = sum(1 for _, excess in self._points if excess > tolerance)
        return over / len(self._points)

    def clear(self) -> None:
        self._points.clear()


class PostureDetector:
    """Turns per-camera metric samples into one posture verdict.

    Fusion happens here and only at the metric level, as specified: each camera
    is compared against its own baseline, and metrics that several cameras can
    see are combined by taking the worst reading. Nothing tries to reconcile
    the cameras geometrically, and a single camera works unchanged -- it just
    contributes fewer metrics.
    """

    def __init__(self, settings: DetectionSettings | None = None,
                 baselines: dict[int, Baseline] | None = None) -> None:
        self.settings = settings or DetectionSettings()
        self.baselines: dict[int, Baseline] = dict(baselines or {})
        self._windows: dict[tuple[int, str], MetricWindow] = {}
        # Last reading per (camera, metric), kept so the UI can show a value
        # for a camera that did not happen to report in the newest call.
        self._latest: dict[tuple[int, str], tuple[float, float, float]] = {}
        self._flagged: set[tuple[int, str]] = set()
        self._state = AWAY
        self._state_since = 0.0
        self._last_person_at: float | None = None
        self._last_usable_at: float | None = None

    # -- configuration -----------------------------------------------------

    def set_baselines(self, baselines: dict[int, Baseline]) -> None:
        """Swap baselines in. Recalibrating discards the window: verdicts built
        against the old baseline say nothing about the new one."""
        self.baselines = dict(baselines)
        self.reset_windows()

    def set_settings(self, settings: DetectionSettings) -> None:
        self.settings = settings
        for window in self._windows.values():
            window.window_seconds = settings.window_seconds
        self.reset_windows()

    def reset_windows(self) -> None:
        """Drop the accumulated evidence and the verdict built from it.

        The state has to go too. While a window refills the detector holds its
        previous verdict rather than flapping, which is right for a normal gap
        but wrong here: recalibrating means "this is my good posture now", and
        carrying a stale BAD across it would alert you seconds after you
        finished sitting up straight for the camera.
        """
        for window in self._windows.values():
            window.clear()
        self._latest.clear()
        self._flagged.clear()
        self._state = GOOD

    @property
    def calibrated(self) -> bool:
        return any(b.metrics for b in self.baselines.values())

    def tolerance(self, spec: met.MetricSpec, camera: int) -> float | None:
        """Tolerance for one metric on one camera, or None if uncalibrated."""
        baseline = self.baselines.get(camera)
        if baseline is None or spec.key not in baseline.metrics:
            return None
        return baseline.metrics[spec.key].tolerance(
            spec, self.settings.tolerance_multiplier,
            self.settings.overrides.get(spec.key),
        )

    # -- the main loop -----------------------------------------------------

    def update(self, samples: Iterable[met.MetricSample],
               now: float | None = None) -> PostureVerdict:
        """Fold new per-camera samples in and return the current verdict.

        Callers may pass one camera's sample or several. Cameras run at their
        own rates and are not synchronised, so each ``(camera, metric)`` pair
        keeps its own window and fusion happens at verdict time instead: a
        metric is flagged if *any* camera's window says so. That is metric-level
        fusion in the sense the brief asks for, and it means an extra camera
        can only add evidence, never dilute another camera's.
        """
        now = time.monotonic() if now is None else now
        samples = list(samples)
        settings = self.settings

        if any(s.person for s in samples):
            self._last_person_at = now
        if any(s.usable for s in samples):
            self._last_usable_at = now

        fresh: dict[tuple[int, str], tuple[float, float, float, float, float]] = {}
        for sample in samples:
            baseline = self.baselines.get(sample.camera_index)
            if baseline is None:
                continue
            for spec in met.SPECS_BY_ROLE.get(sample.role, ()):
                value = sample.values.get(spec.key)
                if value is None or spec.key not in baseline.metrics:
                    continue
                tol = self.tolerance(spec, sample.camera_index)
                if tol is None:
                    continue
                centre = baseline.metrics[spec.key].centre
                excess = met.signed_excess(spec, value, centre)
                ratio = excess / tol if tol > 0 else 0.0
                ident = (sample.camera_index, spec.key)
                fresh[ident] = (value, centre, tol, ratio, excess)
                self._latest[ident] = (value, centre, tol)

        for ident, (_v, _c, _t, _r, excess) in fresh.items():
            window = self._windows.get(ident)
            if window is None:
                window = self._windows[ident] = MetricWindow(settings.window_seconds)
            window.add(now, excess)
        for ident, window in self._windows.items():
            if ident not in fresh:
                window.trim(now)

        state, reason = self._decide(now, samples)
        if state != self._state:
            self._state = state
            self._state_since = now

        verdicts = self._verdicts(now)
        offenders = tuple(v.key for v in verdicts if v.flagged)
        worst = max((v.ratio or 0.0 for v in verdicts), default=0.0)
        return PostureVerdict(
            state=self._state, since=self._state_since, metrics=verdicts,
            offenders=offenders, worst_ratio=worst, calibrated=self.calibrated,
            reason=reason,
        )

    def _live_windows(self, now: float) -> dict[tuple[int, str], MetricWindow]:
        """Windows still holding an observation recent enough to reason about.

        Keyed off recency rather than off whichever camera happened to report
        in this call, so a camera sampling at 2 Hz does not look unmeasurable
        every time a faster one reports first.
        """
        cutoff = now - self.settings.absence_seconds
        return {k: w for k, w in self._windows.items()
                if w.last_t is not None and w.last_t >= cutoff}

    def _decide(self, now: float, samples: list[met.MetricSample]) -> tuple[str, str]:
        settings = self.settings

        away_for = None if self._last_person_at is None else now - self._last_person_at
        if away_for is None or away_for > settings.absence_seconds:
            # An empty chair suspends judgement. The windows are deliberately
            # left alone: coming back to your desk should not wipe the evidence
            # of how you were sitting a minute ago.
            self._flagged.clear()
            return AWAY, "no person detected"

        if not self.calibrated:
            return UNKNOWN, "not calibrated yet"

        live = self._live_windows(now)
        if not live:
            if any(s.person and not s.usable for s in samples):
                missing = sorted({m for s in samples for m in s.missing})
                detail = ", ".join(missing[:4]) if missing else "required landmarks"
                return UNKNOWN, f"cannot see {detail}"
            return UNKNOWN, "no calibrated metric is currently measurable"

        flagged_now: set[tuple[int, str]] = set()
        sparse = True
        for ident, window in live.items():
            camera, key = ident
            spec = met.SPEC_BY_KEY.get(key)
            if spec is None:
                continue
            tol = self.tolerance(spec, camera)
            if tol is None:
                continue
            # "Full enough" is measured in elapsed time, not sample count, so
            # it stays correct while the sample rate varies at runtime.
            if window.span >= settings.min_window_fill * settings.window_seconds:
                sparse = False
            # Hysteresis: a metric already flagged is judged against a tighter
            # tolerance, so clearing it takes a genuine change rather than the
            # same posture and a kinder sample.
            effective = tol * settings.exit_ratio if ident in self._flagged else tol
            if window.out_fraction(effective) > settings.bad_fraction:
                flagged_now.add(ident)

        if sparse:
            return (self._state if self._state in (GOOD, BAD) else GOOD,
                    "still filling the window")

        self._flagged = flagged_now
        if flagged_now:
            names = sorted({met.SPEC_BY_KEY[k].label for _cam, k in flagged_now})
            return BAD, f"out of tolerance: {', '.join(names)}"
        return GOOD, ""

    def _provenance(self, spec: met.MetricSpec,
                    camera: int) -> tuple[float | None, str]:
        """What produced this metric's tolerance: your data, a clamp, or you."""
        if self.settings.overrides.get(spec.key) is not None:
            return None, "manual"
        baseline = self.baselines.get(camera)
        if baseline is None or spec.key not in baseline.metrics:
            return None, "learned"
        derived = self.settings.tolerance_multiplier * baseline.metrics[spec.key].spread
        if derived > spec.max_tolerance:
            return derived, "capped"
        if derived < spec.min_tolerance:
            return derived, "floored"
        return derived, "learned"

    def _verdicts(self, now: float) -> tuple[MetricVerdict, ...]:
        """One row per metric, reporting whichever camera reads it worst."""
        live = self._live_windows(now)
        out: list[MetricVerdict] = []
        for spec in met.SPECS:
            candidates = [(cam, w) for (cam, key), w in live.items() if key == spec.key]
            if not candidates:
                continue
            best_row: MetricVerdict | None = None
            for camera, window in candidates:
                value, centre, tol = self._latest.get(
                    (camera, spec.key), (None, None, self.tolerance(spec, camera) or 0.0))
                excess = window.latest
                ratio = (excess / tol) if (excess is not None and tol > 0) else None
                flagged = (camera, spec.key) in self._flagged
                effective = tol * self.settings.exit_ratio if flagged else tol
                derived, source = self._provenance(spec, camera)
                row = MetricVerdict(
                    key=spec.key, label=spec.label, unit=spec.unit, value=value,
                    baseline=centre, tolerance=tol, excess=excess, ratio=ratio,
                    out_fraction=window.out_fraction(effective),
                    flagged=flagged, samples=len(window),
                    floor=spec.min_tolerance, ceiling=spec.max_tolerance,
                    derived=derived, source=source,
                )
                if best_row is None or (row.ratio or 0.0) > (best_row.ratio or 0.0):
                    best_row = row
            if best_row is not None:
                out.append(best_row)
        return tuple(out)
