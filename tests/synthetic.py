"""Builders for synthetic landmark frames.

The whole point of these tests is that changing a threshold or a sign should
not require sitting in front of a camera and slouching. So poses are built in
the same y-up, aspect-corrected metric frame the geometry code works in, then
converted back into MediaPipe's normalized top-down coordinates. Anything the
metric code gets wrong on the way back out shows up as a failing assertion.
"""

from __future__ import annotations

import math

import numpy as np

from posture import landmarks as lmk


def to_normalized(point: tuple[float, float], aspect: float) -> tuple[float, float]:
    """Invert :func:`posture.geometry.to_metric_frame` for one point."""
    x, y = point
    return (x / aspect, -y)


def build(points: dict[int, tuple[float, float]], aspect: float,
          visibility: float = 1.0,
          faint: tuple[int, ...] = ()) -> np.ndarray:
    """Build a landmark array from metric-frame points.

    ``faint`` names landmarks that should be present but below any sane
    visibility threshold, which is how we simulate an occluded joint.
    """
    arr = np.zeros((lmk.NUM_LANDMARKS, 4), dtype=np.float64)
    for idx, pt in points.items():
        nx, ny = to_normalized(pt, aspect)
        arr[idx] = (nx, ny, 0.0, 0.05 if idx in faint else visibility)
    return arr


def side_pose(*, aspect: float = 4 / 3, torso: float = 0.40, lean_deg: float = 0.0,
              neck_len: float = 0.16, head_forward: float = 0.0,
              nose_ahead: float = 0.06, side: str = "left",
              origin: tuple[float, float] = (0.55, -0.75),
              faint: tuple[int, ...] = (), mirrored: bool = False) -> np.ndarray:
    """A profile view built from anatomy-shaped parameters.

    Args:
        torso: hip-to-shoulder distance, in frame heights.
        lean_deg: torso tilt toward the facing direction. Positive is forward.
        neck_len: vertical shoulder-to-ear distance.
        head_forward: how far the ear sits ahead of the shoulder, in the same
            units as ``torso``. A ratio of ``head_forward / torso`` is what
            ``forward_head`` should report.
        nose_ahead: how far the nose sits ahead of the ear, which is what fixes
            the facing direction.
        mirrored: flip the whole pose about x, i.e. move the camera to the
            other side of the desk. Every reported metric must be unchanged.
    """
    facing = -1.0 if mirrored else 1.0
    hx, hy = origin
    theta = math.radians(lean_deg)
    # Leaning forward tips the shoulder toward the facing direction.
    shoulder = (hx + facing * torso * math.sin(theta), hy + torso * math.cos(theta))
    ear = (shoulder[0] + facing * head_forward, shoulder[1] + neck_len)
    nose = (ear[0] + facing * nose_ahead, ear[1])

    ear_i, sh_i, hip_i = lmk.SIDE_CHAIN[side]
    return build({
        lmk.NOSE: nose,
        ear_i: ear,
        sh_i: shoulder,
        hip_i: (hx, hy),
    }, aspect, faint=faint)


def front_pose(*, aspect: float = 4 / 3, shoulder_width: float = 0.30,
               shoulder_tilt_deg: float = 0.0, ear_width: float = 0.13,
               head_roll_deg: float = 0.0, lateral: float = 0.0,
               neck_len: float = 0.18,
               origin: tuple[float, float] = (0.70, -0.62),
               faint: tuple[int, ...] = (), mirrored: bool = False) -> np.ndarray:
    """A frontal view built from anatomy-shaped parameters.

    ``mirrored`` swaps image left and right, as a mirrored webcam feed does.
    Every reported metric must be unchanged, because they are all defined
    relative to the subject's own body rather than to image sides.
    """
    flip = -1.0 if mirrored else 1.0
    cx, cy = origin
    # Non-mirrored: the subject's right side appears on the image left.
    right_dir = -1.0 * flip

    # Tilt by rotating the shoulder line about its centre, not by raising one
    # end. Raising one end would also lengthen the line, which would change the
    # scale reference and quietly break the ratio metrics that divide by it --
    # a real shoulder does not get further from the other one when it droops.
    half = shoulder_width / 2.0
    tilt = math.radians(shoulder_tilt_deg)
    rs = (cx + right_dir * half * math.cos(tilt), cy + half * math.sin(tilt))
    ls = (cx - right_dir * half * math.cos(tilt), cy - half * math.sin(tilt))

    ear_cx = cx + right_dir * lateral
    ear_cy = cy + neck_len
    ear_half = ear_width / 2.0
    roll = math.radians(head_roll_deg)
    re = (ear_cx + right_dir * ear_half * math.cos(roll), ear_cy + ear_half * math.sin(roll))
    le = (ear_cx - right_dir * ear_half * math.cos(roll), ear_cy - ear_half * math.sin(roll))

    return build({
        lmk.NOSE: (ear_cx, ear_cy - 0.02),
        lmk.LEFT_EAR: le,
        lmk.RIGHT_EAR: re,
        lmk.LEFT_SHOULDER: ls,
        lmk.RIGHT_SHOULDER: rs,
    }, aspect, faint=faint)
