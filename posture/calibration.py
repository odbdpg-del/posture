"""Baselines: what *your* good posture measures, and how much you normally move.

Bad posture is defined as deviation from your own baseline, never from a
textbook ideal. A camera two degrees off level, a chair at a different height,
or simply your proportions would all make an absolute threshold wrong, and
wrong in a way that is invisible until it nags you all day.

Robust statistics, not mean and standard deviation
--------------------------------------------------
Ten seconds of calibration is roughly fifty samples, and some of them will be
rubbish: you settling at the start, a glance at the door, a landmark flickering
below the visibility threshold. A mean is dragged by exactly those samples and
a standard deviation is dragged much harder.

So the centre is the median and the spread is the median absolute deviation,
scaled by 1.4826 so that for normally-distributed data it estimates the same
quantity as the standard deviation. Both need half the samples to be bad before
they move much, which is the property we want from a ten-second window that
nobody supervises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import metrics as met

# Converts a median absolute deviation into a standard-deviation-equivalent for
# normally distributed data, so the tolerance multiplier below means roughly
# what a reader expects it to mean.
MAD_TO_SIGMA = 1.4826

# A metric needs at least this many good samples before its baseline is worth
# keeping. Below it the median is not meaningfully more robust than any single
# reading.
MIN_SAMPLES = 15

# How far past the ceiling a derived tolerance has to land before the capture
# is called noisy. Hitting the ceiling is normal and fine -- the clamp exists to
# be used. Landing at several times it means the ten seconds caught you moving
# around and there was nothing to learn.
NOISE_FACTOR = 2.0


@dataclass(frozen=True)
class MetricBaseline:
    """Where one metric sits, and how much it wanders, when you sit well."""

    key: str
    centre: float
    spread: float
    samples: int

    def tolerance(self, spec: met.MetricSpec, multiplier: float,
                  override: float | None = None) -> float:
        """How far this metric may stray from centre before it counts as bad.

        Derived from your own measured wobble, floored by the spec so that a
        very still calibration cannot produce a hair-trigger, and overridable
        outright from the config for anyone who wants to set a number by hand.
        """
        if override is not None:
            return max(0.0, float(override))
        derived = multiplier * self.spread
        return min(spec.max_tolerance, max(spec.min_tolerance, derived))

    def noisy(self, spec: met.MetricSpec, multiplier: float) -> bool:
        """True when the measured spread was wide enough to hit the ceiling.

        Merely reaching the ceiling is not noise: the clamp is there to be
        used, and with a sensibly tight ceiling most calibrations will reach it.
        This fires only when the measured spread is so wide that the capture
        taught us nothing, where the fix is a calmer ten seconds rather than a
        setting.
        """
        return multiplier * self.spread > NOISE_FACTOR * spec.max_tolerance

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "centre": self.centre, "spread": self.spread,
                "samples": self.samples}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricBaseline":
        return cls(key=str(data["key"]), centre=float(data["centre"]),
                   spread=float(data.get("spread", 0.0)),
                   samples=int(data.get("samples", 0)))


@dataclass(frozen=True)
class Baseline:
    """One camera's calibration: a baseline per metric it could measure."""

    camera: int
    role: str
    metrics: dict[str, MetricBaseline] = field(default_factory=dict)
    captured_at: float = 0.0
    duration: float = 0.0

    @property
    def complete(self) -> bool:
        """True when every metric for this role got a usable baseline."""
        expected = {s.key for s in met.SPECS_BY_ROLE.get(self.role, ())}
        return bool(expected) and expected <= set(self.metrics)

    @property
    def missing(self) -> tuple[str, ...]:
        expected = {s.key for s in met.SPECS_BY_ROLE.get(self.role, ())}
        return tuple(sorted(expected - set(self.metrics)))

    def quality_notes(self, multiplier: float) -> list[str]:
        """Human-readable complaints about how this baseline was captured."""
        notes: list[str] = []
        for key, summary in self.metrics.items():
            spec = met.SPEC_BY_KEY.get(key)
            if spec is not None and summary.noisy(spec, multiplier):
                notes.append(
                    f"{spec.label} varied by +/-{summary.spread:.1f} {spec.unit} "
                    "during calibration, so its tolerance is capped rather than "
                    "learned. Recalibrate while sitting still for a tighter fit."
                )
        return notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "role": self.role,
            "captured_at": self.captured_at,
            "duration": self.duration,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Baseline":
        return cls(
            camera=int(data.get("camera", 0)),
            role=str(data.get("role", "side")),
            captured_at=float(data.get("captured_at", 0.0)),
            duration=float(data.get("duration", 0.0)),
            metrics={k: MetricBaseline.from_dict(v)
                     for k, v in (data.get("metrics") or {}).items()},
        )


def median(values: list[float]) -> float:
    """Median of a non-empty list."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def mad_spread(values: list[float], centre: float) -> float:
    """Median absolute deviation, scaled to compare with a standard deviation."""
    if len(values) < 2:
        return 0.0
    return MAD_TO_SIGMA * median([abs(v - centre) for v in values])


def summarise(key: str, values: list[float],
              min_samples: int = MIN_SAMPLES) -> MetricBaseline | None:
    """Reduce one metric's calibration samples to a baseline, or None."""
    if len(values) < min_samples:
        return None
    centre = median(values)
    return MetricBaseline(key=key, centre=centre,
                          spread=mad_spread(values, centre), samples=len(values))


class CalibrationSession:
    """Collects samples from one camera and turns them into a baseline.

    Only complete-enough samples count: a frame where the metric could not be
    computed contributes nothing rather than contributing a guess. The session
    reports its own progress so the UI can show a countdown that reflects
    usable samples rather than elapsed time -- if you are out of frame for
    three of the ten seconds, calibration should notice.
    """

    def __init__(self, camera: int, role: str, *, duration: float = 10.0,
                 min_samples: int = MIN_SAMPLES) -> None:
        self.camera = camera
        self.role = role
        self.duration = duration
        self.min_samples = min_samples
        self.started_at: float | None = None
        self.finished_at: float | None = None
        # Last clock value seen by add(), so progress() reports against the
        # same time source that is driving the session rather than reading the
        # wall clock behind its back.
        self._now: float | None = None
        self._values: dict[str, list[float]] = {}
        self._frames = 0
        self._usable = 0

    def start(self, now: float | None = None) -> None:
        self.started_at = time.monotonic() if now is None else now
        self.finished_at = None
        self._now = self.started_at
        self._values.clear()
        self._frames = self._usable = 0

    @property
    def running(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        if self.finished_at is not None:
            end = self.finished_at
        elif now is not None:
            end = now
        elif self._now is not None:
            end = self._now
        else:
            end = time.monotonic()
        return max(0.0, end - self.started_at)

    def add(self, sample: met.MetricSample, now: float | None = None) -> None:
        """Feed one frame's metrics in."""
        if not self.running:
            return
        # Advance our clock on every sample, whether the caller supplied a time
        # or not. Falling back to the stored value when none is given would
        # freeze elapsed() at zero and the session would never finish.
        self._now = time.monotonic() if now is None else now
        self._frames += 1
        if sample.usable:
            self._usable += 1
        for key, value in sample.values.items():
            self._values.setdefault(key, []).append(value)
        if self.elapsed(now) >= self.duration:
            self.finished_at = self.started_at + self.duration

    @property
    def progress(self) -> float:
        """0..1 through the capture window.

        A finished session reports exactly 1.0 rather than a computed ratio:
        ``elapsed`` is ``(started + duration) - started``, which floating point
        renders as 0.9999999999854481, and "finished" is a fact we already know
        rather than something to re-derive.
        """
        if self.started_at is None or self.duration <= 0:
            return 0.0
        if self.finished_at is not None:
            return 1.0
        return min(1.0, self.elapsed() / self.duration)

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._values.items()}

    def result(self) -> tuple[Baseline, list[str]]:
        """Build the baseline, plus a list of human-readable problems.

        Returns whatever it managed rather than raising, because a partial
        baseline is still useful: a side camera that could not see your ear can
        still calibrate torso lean, and the UI should say so.
        """
        problems: list[str] = []
        metrics: dict[str, MetricBaseline] = {}
        for spec in met.SPECS_BY_ROLE.get(self.role, ()):
            values = self._values.get(spec.key, [])
            summary = summarise(spec.key, values, self.min_samples)
            if summary is None:
                problems.append(
                    f"{spec.label}: only {len(values)} usable sample(s), "
                    f"need {self.min_samples}"
                )
                continue
            metrics[spec.key] = summary

            # A one-sided metric measures you against your own baseline, so a
            # baseline captured somewhere a neutral posture cannot reach makes
            # the metric permanently angry: sitting normally reads as a
            # deviation, and no amount of sitting up ever clears it.
            #
            # Seen in the wild. A torso lean baseline of -15.9 degrees -- taken
            # while reclining -- put an upright torso 15.9 past baseline
            # against a 7 degree tolerance, so the posture score sat near zero
            # all day and blamed torso lean while the person sat perfectly
            # straight. Nothing else in the app can notice this: a reclining
            # torso is a physically plausible reading, so no guard rejects it,
            # and the spread was tight, so the baseline looked high quality.
            drift = met.signed_excess(spec, spec.neutral, summary.centre)
            if drift > spec.min_tolerance:
                problems.append(
                    f"{spec.label}: baseline of {summary.centre:.1f} is far enough "
                    f"from neutral ({spec.neutral:.0f} {spec.unit}) that sitting "
                    f"neutrally reads {drift:.1f} past it, beyond the "
                    f"{spec.min_tolerance:.0f} tolerance -- so normal posture will "
                    "always look wrong. Recalibrate sitting the way you want to sit."
                )

        if self._frames and self._usable / self._frames < 0.5:
            problems.append(
                f"only {self._usable} of {self._frames} frames were usable -- "
                "check the camera can see you throughout"
            )
        baseline = Baseline(
            camera=self.camera, role=self.role, metrics=metrics,
            captured_at=time.time(), duration=self.elapsed(),
        )
        return baseline, problems


def build_baseline(camera: int, role: str, samples: Iterable[met.MetricSample],
                   *, min_samples: int = MIN_SAMPLES) -> tuple[Baseline, list[str]]:
    """Build a baseline from an existing run of samples.

    The path the tests use, and the reason the statistics can be checked
    without a camera in the room.
    """
    session = CalibrationSession(camera, role, duration=float("inf"),
                                 min_samples=min_samples)
    session.start(now=0.0)
    for sample in samples:
        session.add(sample, now=0.0)
    session.finished_at = session.started_at
    return session.result()
