"""Per-frame posture metrics, computed independently per camera role.

Each camera is processed on its own and produces a :class:`MetricSample`.
Nothing here knows about other cameras: fusion happens later, at the metric
level, exactly as specified. That keeps a single camera fully functional.

Scale normalization
-------------------
Metrics that are lengths get divided by a body-derived reference so that
sitting closer to the camera does not change them.

The reference is chosen per role, and this is a deliberate deviation from the
brief's "normalize by shoulder width":

* front role -> shoulder width. Correct and stable; the shoulder line is fully
  in view and roughly fronto-parallel.
* side role -> torso length (shoulder-to-hip distance) when the hip is visible.
  From a side view the two shoulder landmarks project almost on top of each
  other, so the measured shoulder width collapses toward zero and its relative
  noise explodes. Dividing by it would make the side metrics wildly unstable at
  exactly the angle where the side camera is most useful. Torso length is the
  longest body segment reliably visible from the side.
* side role without a hip -> ear-to-shoulder distance, which only scales the
  neck-tilt readout since that metric is an angle and needs no normalizer.

Whichever is in use is reported on the sample as ``scale``/``scale_kind``.

Hips are optional
-----------------
This app is for people sitting at a desk, where the hips are usually under it,
below the frame, or behind an armrest. So the side role has two tiers: ear plus
shoulder gives neck tilt, and a visible hip adds neck flexion, forward head and
torso lean on top. Only the shoulder is ever mandatory. Requiring hips meant a
side camera reported nothing for most of the day, which is worse than reporting
less.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from . import geometry as geo
from . import landmarks as lmk

Role = Literal["side", "front"]

# A length reference this small means the landmarks have collapsed together and
# any ratio built on them is noise. Units are frame heights.
MIN_SCALE = 0.02

# The ears must be at least this fraction of shoulder width apart before head
# roll is meaningful. Turn your head far enough toward the camera axis and the
# ear-to-ear line foreshortens to nothing; the angle then swings wildly on
# sub-pixel noise, so we refuse to report it rather than report garbage.
MIN_EAR_SEPARATION_RATIO = 0.15

# How face-on you have to be before the front metrics mean anything, as the
# fraction of the shoulder span that is actually horizontal in the image --
# cos(turn) for a level pair of shoulders.
#
# The front metrics all assume you are facing the camera. Turn away and the
# shoulder pair foreshortens: the horizontal separation collapses toward zero
# while the vertical offset between the two shoulders does not, so
# ``arctan2(dy, |dx|)`` swings toward +/-90 on a body that has not moved.
# MIN_SCALE does not catch this. It guards the straight-line distance between
# the shoulders, and that distance stays healthy precisely because of the
# vertical offset causing the trouble -- at 85 degrees of turn it still reads
# 0.038 against a limit of 0.02.
#
# Not hypothetical. Measured on a real camera: shoulder tilt -52 degrees and
# head roll -59 against a 4-degree tolerance, from someone who had turned to
# talk to somebody. The detector was right not to alert -- the deviation was
# not sustained -- but the posture score reads the instantaneous ratio, so the
# headline number dropped to 19 out of 100 and stayed there.
#
# Since |shoulder_tilt| is exactly arccos(directness), this doubles as the
# plausibility cap the side metrics already have -- 0.866 bounds the reported
# tilt at 30 degrees. That is 2.5x the largest tolerance the metric can be
# given (max_tolerance is 12) so it cannot suppress a tilt worth flagging,
# while sitting far below the readings a turn produces.
#
# It bounds the damage rather than detecting the turn: this is a projection, so
# "turned 80 degrees" and "shoulders genuinely tilted" are the same picture,
# and nothing in a single shoulder pair can separate them. A moderate turn
# still inflates the reading somewhat -- it just can no longer reach the
# absurd values that were reaching the score.
MIN_FACING_DIRECTNESS = 0.866

# How far the torso may tilt from vertical before we stop believing the
# landmarks. A person leaning hard over a desk reaches maybe 45 degrees; past
# 60 the shoulder is barely above the hip at all.
#
# This is not a hypothetical guard. When the hips sit near or below the bottom
# of the frame the pose model extrapolates them rather than admitting it cannot
# see them, and it reports the guess with high visibility -- so the usual
# threshold does not catch it. Measured on a real camera, that produced torso
# lean values of -90 and -178 degrees, meaning "shoulders below hips", and a
# ten-second calibration inherited a spread of 170 degrees. A tolerance derived
# from that spread could never flag anything, so the metric would have been
# silently dead rather than visibly broken.
MAX_PLAUSIBLE_LEAN_DEG = 60.0

# The same guard for the head. Neck tilt is the angle of the ear-over-shoulder
# segment from vertical, so past this the ear is level with or below the
# shoulder, which is not a seated head -- it is the pose model fitting a person
# to a chair, a coat, or a badly cropped frame.
#
# Without it the failure is silent and flattering rather than obvious: a reading
# of -149 degrees is enormously *below* baseline, neck tilt only counts upward
# deviation as bad, so the excess is zero and the posture score reports a
# confident 100. Observed exactly that on an empty chair.
MAX_PLAUSIBLE_NECK_TILT_DEG = 60.0


# Which way a metric has to move from your baseline before it is worth
# complaining about. Getting this wrong is worse than a bad threshold: an
# "either" metric that should be one-sided nags you for improving your posture.
LOW_IS_BAD = "low"      # dropping below baseline is the problem
HIGH_IS_BAD = "high"    # rising above baseline is the problem
EITHER_IS_BAD = "both"  # asymmetry in either direction is the problem


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one metric."""

    key: str
    role: Role
    label: str
    short: str
    unit: str
    direction: str
    # Smallest deviation from baseline that counts as bad, in the metric's own
    # units. Calibration widens this to match how much you actually move, but
    # never narrows it below this floor: sit unnaturally still for the ten
    # seconds of calibration and the measured spread approaches zero, which
    # without a floor would make every subsequent breath a posture violation.
    min_tolerance: float
    # Largest deviation that can still count as normal. The mirror of the
    # floor, and just as necessary: calibrate while fidgeting and the measured
    # spread balloons, so a tolerance derived from it would be so wide the
    # metric could never flag anything. A floored-only tolerance fails loudly
    # (nagging); a ceilinged-only one fails silently (dead metric), which is
    # worse. These are the deviations past which posture is bad for anyone,
    # whatever their baseline.
    #
    # Raising a ceiling makes a metric *less* sensitive, not more: it lets a
    # wider learned tolerance through, so you have to deviate further before
    # anything is said. Lower it to make a metric complain sooner.
    max_tolerance: float
    description: str
    # The anatomically neutral value: ear over shoulder, torso vertical,
    # shoulders level. Not a target -- the app judges you against your own
    # calibrated baseline, not against an ideal -- but calibration needs it to
    # notice that a baseline has been captured somewhere a neutral posture
    # cannot reach. See ``CalibrationSession.result``.
    neutral: float = 0.0


SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        # The only side metric that needs no hip, and the reason the side role
        # still works at a desk. A camera beside a desk very often cannot see
        # your hips at all -- they are under the desk, or below the frame -- so
        # anything that required them would spend most of the day reporting
        # nothing. Ear-over-shoulder against vertical is measurable from head
        # and shoulders alone, and forward head is the desk failure mode
        # anyway.
        "neck_tilt", "side", "Neck tilt", "tilt", "deg", HIGH_IS_BAD, 6.0, 15.0,
        "Ear-over-shoulder line versus vertical. Positive means your head is "
        "in front of your shoulders. Works without hips in frame.",
    ),
    MetricSpec(
        "neck_flexion", "side", "Neck flexion", "neck", "deg", LOW_IS_BAD, 8.0, 15.0,
        "Ear-shoulder-hip angle. 180 is a perfectly stacked head; smaller is more slouched.",
        neutral=180.0,
    ),
    MetricSpec(
        "forward_head", "side", "Forward head", "fwdhd", "x torso", HIGH_IS_BAD, 0.08, 0.25,
        "Horizontal ear-ahead-of-shoulder offset over torso length. Positive is head forward.",
    ),
    MetricSpec(
        # One-sided on purpose. Leaning further forward than your baseline is
        # the desk-work failure mode; sitting back into the chair is not
        # something to interrupt someone about.
        "torso_lean", "side", "Torso lean", "lean", "deg", HIGH_IS_BAD, 7.0, 20.0,
        "Shoulder-over-hip line versus vertical. Positive is leaning forward.",
    ),
    MetricSpec(
        "shoulder_tilt", "front", "Shoulder tilt", "shtilt", "deg", EITHER_IS_BAD, 4.0, 12.0,
        "Shoulder line versus horizontal. Positive means your right shoulder is higher.",
    ),
    MetricSpec(
        "head_roll", "front", "Head roll", "roll", "deg", EITHER_IS_BAD, 6.0, 15.0,
        "Ear line versus horizontal. Positive means your head is tipped toward your left.",
    ),
    MetricSpec(
        "lateral_offset", "front", "Lateral offset", "latof", "x shoulders",
        EITHER_IS_BAD, 0.10, 0.25,
        "Ear midpoint versus shoulder midpoint, sideways. Positive is toward your right.",
    ),
)


SPECS_BY_ROLE: dict[str, tuple[MetricSpec, ...]] = {
    role: tuple(s for s in SPECS if s.role == role) for role in ("side", "front")
}
SPEC_BY_KEY: dict[str, MetricSpec] = {s.key: s for s in SPECS}


def signed_excess(spec: MetricSpec, value: float, baseline: float) -> float:
    """How far past the baseline this value sits, in the bad direction only.

    Zero means "at or better than baseline". Movement in the good direction is
    never reported as excess, which is what keeps a one-sided metric one-sided.
    """
    delta = value - baseline
    if spec.direction == HIGH_IS_BAD:
        return max(0.0, delta)
    if spec.direction == LOW_IS_BAD:
        return max(0.0, -delta)
    return abs(delta)


@dataclass(frozen=True)
class MetricSample:
    """One camera's reading for one frame.

    ``person`` false means the model found nobody. ``person`` true with empty
    ``values`` means we can see someone but not the parts we need. These are
    different states and the UI is required to tell them apart.
    """

    role: Role
    t: float
    person: bool
    values: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    scale: float | None = None
    scale_kind: str | None = None
    facing: int | None = None
    near_side: str | None = None
    notes: tuple[str, ...] = ()
    # Which camera produced this. The detector compares each camera against its
    # own baseline, so a sample has to carry its origin with it.
    camera_index: int = 0

    @property
    def usable(self) -> bool:
        """True when at least one metric for this role came out."""
        return bool(self.values)

    @property
    def complete(self) -> bool:
        """True when every metric for this role came out."""
        return len(self.values) == len(SPECS_BY_ROLE[self.role])


def _visible(arr: np.ndarray, idx: int, thresh: float) -> bool:
    return bool(arr[idx, 3] >= thresh)


def _pick_near_side(arr: np.ndarray, thresh: float) -> tuple[str | None, tuple[str, ...]]:
    """Choose which body side faces a side-mounted camera.

    Ear and shoulder are weighted fully and the hip only half, because the hip
    decides how *much* we can measure while ear and shoulder decide whether we
    can measure anything at all. Scoring them equally would let a confidently
    detected hip drag the choice onto the side whose head we cannot see.

    Returns the side name, or None with the faint landmark names when even the
    minimum is not visible, so the UI can say what is missing rather than "no".
    """
    best_side: str | None = None
    best_score = -1.0
    best_missing: tuple[str, ...] = ()
    for side, chain in lmk.SIDE_CHAIN.items():
        ear, shoulder, hip = chain
        score = float(arr[ear, 3] + arr[shoulder, 3] + 0.5 * arr[hip, 3])
        missing = tuple(lmk.NAMES[i] for i in chain if not _visible(arr, i, thresh))
        if score > best_score:
            best_side, best_score, best_missing = side, score, missing
    if best_side is None:
        return None, ()
    # The shoulder is the only hard requirement: it is the vertex every side
    # metric measures from. The ear unlocks neck tilt, the hip unlocks the
    # torso metrics, and either on its own is still worth reporting. Requiring
    # both would silence a desk camera, which usually has one or the other.
    _ear, shoulder, _hip = lmk.SIDE_CHAIN[best_side]
    if not _visible(arr, shoulder, thresh):
        return None, best_missing
    return best_side, best_missing


def _facing(arr: np.ndarray, pts: np.ndarray, ear_idx: int, thresh: float) -> int | None:
    """Which way the subject faces in the frame: +1 toward +x, -1 toward -x.

    Uses nose-versus-ear because both are head landmarks that appear and
    disappear together, and the nose is unambiguously in front of the ear.
    Knowing this lets forward-head and torso-lean carry a consistent sign
    whether the camera sits on your left or your right, so moving it to the
    other side of the desk does not invert your calibration.
    """
    if not (_visible(arr, lmk.NOSE, thresh) and _visible(arr, ear_idx, thresh)):
        return None
    dx = pts[lmk.NOSE, 0] - pts[ear_idx, 0]
    if abs(dx) < 1e-6:
        return None
    return 1 if dx > 0 else -1


def compute_side(arr: np.ndarray, aspect: float, vis_thresh: float, t: float) -> MetricSample:
    """Side-camera metrics, degrading gracefully when the hips are not in view.

    Two tiers. Ear and shoulder alone give neck tilt, which is the forward-head
    signal and the thing desk posture is usually about. Add a visible hip and
    the torso becomes measurable too: neck flexion, forward head as a fraction
    of torso length, and torso lean.

    Hips are deliberately not required. Sitting at a desk, they are frequently
    under it, below the frame, or occluded by an armrest, and a side camera that
    refused to report anything without them would be silent most of the day.
    """
    pts = geo.to_metric_frame(arr[:, :2], aspect)
    near_side, missing = _pick_near_side(arr, vis_thresh)
    if near_side is None:
        return MetricSample("side", t, True, missing=missing or ("shoulder",))

    ear_i, sh_i, hip_i = lmk.SIDE_CHAIN[near_side]
    ear, shoulder = pts[ear_i], pts[sh_i]
    ear_ok = _visible(arr, ear_i, vis_thresh)
    hip_ok = _visible(arr, hip_i, vis_thresh)
    facing = _facing(arr, pts, ear_i, vis_thresh) if ear_ok else None

    values: dict[str, float] = {}
    notes: list[str] = []
    scale: float | None = None
    scale_kind: str | None = None

    # -- tier one: head and shoulder --------------------------------------
    if ear_ok:
        neck_vec = ear - shoulder
        neck_len = geo.norm(neck_vec)
        if neck_len < MIN_SCALE:
            notes.append("ear and shoulder too close together to read")
        else:
            scale, scale_kind = neck_len, "neck"
            tilt = geo.signed_angle_from_vertical(neck_vec)
            if tilt is None:
                pass
            elif abs(tilt) > MAX_PLAUSIBLE_NECK_TILT_DEG:
                notes.append("head landmarks implausible (ear is not above the "
                             "shoulder)")
            elif facing is not None:
                values["neck_tilt"] = tilt * facing
            else:
                notes.append("neck_tilt needs the nose to fix its sign")

    # -- tier two: everything that needs a hip ----------------------------
    if not hip_ok:
        notes.append("hips not in frame; torso metrics unavailable")
        return MetricSample(
            "side", t, True, values=values, missing=missing, scale=scale,
            scale_kind=scale_kind, facing=facing, near_side=near_side,
            notes=tuple(notes),
        )

    hip = pts[hip_i]
    torso = geo.norm(shoulder - hip)
    raw_lean = geo.signed_angle_from_vertical(shoulder - hip)
    # Reject anatomically impossible torsos before anything is derived from
    # them, but only discard the torso metrics -- neck tilt was measured from
    # landmarks the bad hip never touched, so it survives.
    if torso < MIN_SCALE:
        notes.append("torso reference too short to normalize")
    elif raw_lean is None or abs(raw_lean) > MAX_PLAUSIBLE_LEAN_DEG:
        notes.append("torso landmarks implausible (hips may be out of frame)")
    else:
        scale, scale_kind = torso, "torso"
        # Sign so that positive always means leaning toward what you face.
        values["torso_lean"] = raw_lean * facing if facing is not None else raw_lean
        if facing is None:
            notes.append("torso_lean sign unresolved (nose not visible)")
        if ear_ok:
            neck = geo.angle_at(ear, shoulder, hip)
            if neck is not None:
                values["neck_flexion"] = neck
            if facing is not None:
                values["forward_head"] = float((ear[0] - shoulder[0]) * facing / torso)
            else:
                notes.append("forward_head needs the nose to fix its sign")

    return MetricSample(
        "side", t, True, values=values, missing=missing, scale=scale,
        scale_kind=scale_kind, facing=facing, near_side=near_side,
        notes=tuple(notes),
    )


def compute_front(arr: np.ndarray, aspect: float, vis_thresh: float, t: float) -> MetricSample:
    """Front-camera metrics: shoulder tilt, head roll, lateral head offset."""
    pts = geo.to_metric_frame(arr[:, :2], aspect)
    shoulders = (lmk.LEFT_SHOULDER, lmk.RIGHT_SHOULDER)
    missing = tuple(lmk.NAMES[i] for i in shoulders if not _visible(arr, i, vis_thresh))
    if missing:
        return MetricSample("front", t, True, missing=missing)

    ls, rs = pts[lmk.LEFT_SHOULDER], pts[lmk.RIGHT_SHOULDER]
    scale = geo.norm(rs - ls)
    notes: list[str] = []
    if scale < MIN_SCALE:
        notes.append("shoulders too close together to normalize (are you side-on?)")
        return MetricSample("front", t, True, scale=scale, scale_kind="shoulders",
                            notes=tuple(notes))

    # Every front metric divides by a horizontal span that a turn collapses, so
    # the gate is on all three rather than on the angle that shows it worst.
    # Reporting a lateral offset normalized by a foreshortened shoulder span
    # would be the same error one step further on.
    spread = abs(rs[0] - ls[0])
    if spread < MIN_FACING_DIRECTNESS * scale:
        notes.append("not facing the camera squarely enough to read "
                     "the front metrics (are you turned?)")
        return MetricSample("front", t, True, scale=scale, scale_kind="shoulders",
                            notes=tuple(notes))

    values: dict[str, float] = {}
    # Measured as "how much higher is the right end than the left" against the
    # horizontal separation, rather than as a directed line angle. Written this
    # way the sign means the same thing whether or not the feed is mirrored,
    # and it never wraps near +/-180.
    values["shoulder_tilt"] = float(np.degrees(np.arctan2(rs[1] - ls[1], spread)))

    ears = (lmk.LEFT_EAR, lmk.RIGHT_EAR)
    ear_missing = tuple(lmk.NAMES[i] for i in ears if not _visible(arr, i, vis_thresh))
    if ear_missing:
        missing = missing + ear_missing
    else:
        le, re = pts[lmk.LEFT_EAR], pts[lmk.RIGHT_EAR]
        ear_sep = abs(re[0] - le[0])
        if ear_sep < MIN_EAR_SEPARATION_RATIO * scale:
            notes.append("head turned too far to read roll reliably")
        else:
            values["head_roll"] = float(np.degrees(np.arctan2(re[1] - le[1], ear_sep)))

        ear_mid = geo.midpoint(le, re)
        sh_mid = geo.midpoint(ls, rs)
        right_dir = 1.0 if rs[0] > ls[0] else -1.0
        values["lateral_offset"] = float((ear_mid[0] - sh_mid[0]) * right_dir / scale)

    return MetricSample(
        "front", t, True, values=values, missing=missing, scale=scale,
        scale_kind="shoulders", notes=tuple(notes),
    )


def compute(role: Role, arr: np.ndarray | None, aspect: float, vis_thresh: float,
            t: float, camera_index: int = 0) -> MetricSample:
    """Dispatch to the right role. ``arr`` of None means no person detected."""
    if role not in ("side", "front"):
        raise ValueError(f"unknown camera role: {role!r}")
    if arr is None:
        sample = MetricSample(role, t, person=False)
    elif role == "side":
        sample = compute_side(arr, aspect, vis_thresh, t)
    else:
        sample = compute_front(arr, aspect, vis_thresh, t)
    return replace(sample, camera_index=camera_index)
