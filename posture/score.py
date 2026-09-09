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

The blend alone did not deliver that second half. Any ratio at or past
``RATIO_FLOOR`` maxes the instantaneous term, so a single wild reading cost 60
points on its own, whatever the window said. Measured live: a metric out of
tolerance for half its window scored 19 out of 100 -- "needs correction" -- while
the detector's own state was "good" and no alert was firing, because the
detector will not call a metric bad until it has been out for ``bad_fraction``
of the window. The score has no business contradicting the verdict it is
derived from.

So the blend is capped by how much of the window actually supports it, and the
cap is anchored to the bands, because the band is what the panel shows:

* nothing in the window supports the deviation -- it may dent the score, but
  not push it out of "good";
* the window is as full as the detector needs to call it bad -- the score may
  reach the bottom of "fair", but not cross into "needs correction" while the
  app's own verdict is that your posture is fine;
* past that the detector agrees, the cap lifts, and the score is free to
  bottom out.

The cap makes the score plateau rather than keep falling once a deviation
outruns its evidence. That is the point: how far out you are right now is worth
something, but not more than the window will vouch for.

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

# The detector will not call a metric bad until it has been out of tolerance
# for this much of its rolling window. Mirrored rather than imported, so this
# module stays a leaf testable against plain stand-ins; the real value is
# passed in by the monitor, and a test pins this default to the detector's.
DEFAULT_BAD_FRACTION = 0.70

# The two anchors of the cap, in severity. Severity 0.30 is a score of 70, the
# floor of "good"; 0.50 is a score of 50, the floor of "fair". Both are read
# off BANDS below rather than chosen independently -- a cap that did not land
# on a band boundary would show up as a number that stops just short of a
# label for no visible reason.
UNSUPPORTED_CEILING = 0.30
THRESHOLD_CEILING = 0.50

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


def severity_ceiling(out_fraction: float,
                     bad_fraction: float = DEFAULT_BAD_FRACTION,
                     flagged: bool = False) -> float:
    """How far down the score may go on this much sustained evidence.

    Piecewise linear through the two band anchors, and continuous at both
    joints, so the number never jumps as the window fills.

    ``flagged`` lifts the cap outright, because it is the detector's own answer
    and the cap exists only to avoid contradicting it. It is not the same
    question as ``out_fraction > bad_fraction``: hysteresis keeps a metric
    flagged while its out-fraction falls back under the threshold, and a
    reassuring score beside an alert that is still on screen would be the same
    contradiction in the other direction.
    """
    if flagged:
        return 1.0
    out_fraction = min(max(out_fraction, 0.0), 1.0)
    if bad_fraction <= 0.0:          # every deviation counts as confirmed
        return 1.0
    if out_fraction <= bad_fraction:
        share = out_fraction / bad_fraction
        return UNSUPPORTED_CEILING + (THRESHOLD_CEILING - UNSUPPORTED_CEILING) * share
    if bad_fraction >= 1.0:          # nothing short of the whole window counts
        return THRESHOLD_CEILING
    share = (out_fraction - bad_fraction) / (1.0 - bad_fraction)
    return THRESHOLD_CEILING + (1.0 - THRESHOLD_CEILING) * share


def metric_severity(ratio: float | None, out_fraction: float,
                    bad_fraction: float = DEFAULT_BAD_FRACTION,
                    flagged: bool = False) -> float:
    """Blend "how far out now" with "how long it has been out", on 0..1.

    Capped by :func:`severity_ceiling`, so the blend can never claim more than
    the window supports.
    """
    now = 0.0 if ratio is None else min(max(ratio, 0.0), RATIO_FLOOR) / RATIO_FLOOR
    sustained = min(max(out_fraction, 0.0), 1.0)
    blended = NOW_WEIGHT * now + SUSTAINED_WEIGHT * sustained
    return min(blended, severity_ceiling(out_fraction, bad_fraction, flagged))


def score_verdict(verdict: Any,
                  bad_fraction: float = DEFAULT_BAD_FRACTION) -> PostureScore:
    """Turn a :class:`detector.PostureVerdict` into a score.

    Takes the verdict duck-typed rather than imported, so this module stays a
    leaf and can be unit-tested against plain stand-ins. ``bad_fraction`` is
    the detector's own threshold, passed in for the same reason -- the caller
    has the settings that produced this verdict, and a copy here could drift
    from them silently.
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
        severity = metric_severity(metric.ratio, metric.out_fraction, bad_fraction,
                                   getattr(metric, "flagged", False))
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
