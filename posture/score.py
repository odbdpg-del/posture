"""A single posture score, derived from measurements the app already makes.

Deliberately isolated in its own module with no dependencies beyond the
detector's verdict, because the formula below is a judgement call and will want
revising once there is real usage behind it. Nothing else in the app should
grow its own opinion about what "92" means.

What it is built from
---------------------
Two numbers per metric, both already computed by the detector:

* ``ratio`` -- how far past your calibrated tolerance the metric is *right now*,
  where 1.0 sits exactly on the limit.
* ``out_fraction`` -- how much of the rolling window it has spent out of
  tolerance, which is what the app already uses to decide "bad".

Instantaneous deviation alone would make the score twitch every time you
reached for a mug; sustained fraction alone would take a minute to react to a
genuine slump. Blending them gives a number that moves when you move but only
falls a long way when a deviation persists.

The overall score takes the *worst* metric rather than an average, matching how
the detector already works: one axis being wrong is enough. Averaging would let
three good metrics hide a badly forward head.

There is no score when there is nobody there, when the landmarks cannot be
read, or before calibration. Those are absences of information, and reporting
100 for an empty chair would be a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How the instantaneous and sustained parts are mixed. Weighted toward "right
# now" so the number is responsive, with enough of the window's memory that a
# brief stretch does not tank it.
NOW_WEIGHT = 0.6
SUSTAINED_WEIGHT = 0.4

# Deviation at which a metric is considered as bad as it is going to get, in
# units of its own tolerance. Past this the score has already bottomed out for
# that metric, so there is nothing to gain from scaling further.
RATIO_FLOOR = 1.5

# Score bands. Ordered worst-first so the first match wins on a simple scan.
BANDS: tuple[tuple[int, str, str], ...] = (
    (0, "needs correction", "bad"),
    (50, "fair", "warn"),
    (70, "good", "ok"),
    (85, "excellent", "good"),
)


@dataclass(frozen=True)
class MetricScore:
    """One metric's contribution, kept so the UI can explain the number."""

    key: str
    label: str
    severity: float  # 0 = at baseline, 1 = as bad as this metric gets
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label,
                "severity": round(self.severity, 4), "score": self.score}


@dataclass(frozen=True)
class PostureScore:
    """The headline number, plus enough detail to justify it."""

    value: int | None                 # None when there is nothing to score
    band: str = "unknown"             # human label, e.g. "excellent"
    tone: str = "idle"                # good | ok | warn | bad | idle
    reason: str = ""                  # why there is no score, when there isn't
    worst: str | None = None          # label of the metric dragging it down
    metrics: tuple[MetricScore, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "band": self.band,
            "tone": self.tone,
            "reason": self.reason,
            "worst": self.worst,
            "metrics": [m.to_dict() for m in self.metrics],
        }


def band_for(value: int) -> tuple[str, str]:
    """Human label and tone for a score."""
    label, tone = BANDS[0][1], BANDS[0][2]
    for threshold, name, colour in BANDS:
        if value >= threshold:
            label, tone = name, colour
    return label, tone


def metric_severity(ratio: float | None, out_fraction: float) -> float:
    """Blend "how far out now" with "how long it has been out", on 0..1."""
    now = 0.0 if ratio is None else min(max(ratio, 0.0), RATIO_FLOOR) / RATIO_FLOOR
    sustained = min(max(out_fraction, 0.0), 1.0)
    return NOW_WEIGHT * now + SUSTAINED_WEIGHT * sustained


def score_verdict(verdict: Any) -> PostureScore:
    """Turn a :class:`detector.PostureVerdict` into a score.

    Takes the verdict duck-typed rather than imported, so this module stays a
    leaf and can be unit-tested against plain stand-ins.
    """
    state = getattr(verdict, "state", "unknown")
    if not getattr(verdict, "calibrated", False):
        return PostureScore(None, "not calibrated", "idle",
                            reason="calibrate to start scoring")
    if state == "away":
        return PostureScore(None, "away", "idle", reason="nobody at the desk")
    if state == "unknown":
        return PostureScore(None, "no reading", "idle",
                            reason=getattr(verdict, "reason", "") or
                            "cannot read the landmarks needed")

    rows: list[MetricScore] = []
    for metric in getattr(verdict, "metrics", ()):
        severity = metric_severity(metric.ratio, metric.out_fraction)
        rows.append(MetricScore(
            key=metric.key, label=metric.label, severity=severity,
            score=int(round(100 * (1.0 - severity))),
        ))
    if not rows:
        return PostureScore(None, "no reading", "idle",
                            reason="no metric is currently measurable")

    # Worst axis wins: the detector already treats one bad metric as bad
    # posture, and an average would let good axes mask a badly forward head.
    worst = max(rows, key=lambda r: r.severity)
    value = int(round(100 * (1.0 - worst.severity)))
    label, tone = band_for(value)
    return PostureScore(value=value, band=label, tone=tone,
                        worst=worst.label if worst.severity > 0.05 else None,
                        metrics=tuple(rows))
