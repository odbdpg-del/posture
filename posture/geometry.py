"""Pure 2D geometry helpers.

Everything here operates on plain numpy arrays and has no dependency on
MediaPipe, OpenCV or any camera. That is deliberate: the geometry is the part
most likely to be wrong, and it is the part that must be testable against
synthetic landmarks without a webcam.

Coordinate convention
---------------------
MediaPipe hands back normalized coordinates with ``x`` in [0, 1] measured
left-to-right and ``y`` in [0, 1] measured TOP-DOWN. Two things follow:

1. ``y`` grows downward, so naive angle math comes out mirrored.
2. ``x`` and ``y`` are normalized by different quantities (frame width and
   frame height). On a 16:9 frame a physically 45-degree line does not have
   slope 1 in normalized space. Any angle computed straight off normalized
   coordinates is wrong by the aspect ratio.

So all metric code first calls :func:`to_metric_frame`, which undoes both
problems: it scales ``x`` by the aspect ratio and flips ``y`` so it points up.
The result is in units of "frame heights" -- an arbitrary but *isotropic*
scale, which is all angles need. Distances are made scale-free separately by
dividing by a body-derived reference length.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_metric_frame",
    "norm",
    "unit",
    "angle_between",
    "angle_at",
    "signed_angle_from_vertical",
    "signed_angle_from_horizontal",
    "midpoint",
]

# Below this length a vector's direction is numerically meaningless.
EPS = 1e-9


def to_metric_frame(xy: np.ndarray, aspect: float) -> np.ndarray:
    """Convert normalized MediaPipe coordinates into an isotropic, y-up frame.

    Args:
        xy: array of shape (..., 2) of normalized coordinates.
        aspect: frame width divided by frame height.

    Returns:
        Array of the same shape, in units of frame heights, with +y pointing up.
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] * aspect
    out[..., 1] = -xy[..., 1]
    return out


def norm(v: np.ndarray) -> float:
    """Euclidean length of a vector."""
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray | None:
    """Unit vector, or None if ``v`` is too short to have a direction."""
    n = norm(v)
    if n < EPS:
        return None
    return np.asarray(v, dtype=np.float64) / n


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float | None:
    """Unsigned angle between two vectors, in degrees on [0, 180]."""
    u1, u2 = unit(v1), unit(v2)
    if u1 is None or u2 is None:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))))


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Interior angle at vertex ``b`` in the path a-b-c, in degrees.

    180 degrees means a, b, c are collinear with b in the middle.
    """
    return angle_between(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))


def signed_angle_from_vertical(v: np.ndarray) -> float | None:
    """Angle of ``v`` away from straight up, in degrees on (-180, 180].

    Positive means leaning toward +x, negative toward -x. A vector pointing
    straight up returns 0.
    """
    u = unit(v)
    if u is None:
        return None
    # atan2(x, y) measures from the +y axis (up) toward +x.
    return float(np.degrees(np.arctan2(u[0], u[1])))


def signed_angle_from_horizontal(v: np.ndarray) -> float | None:
    """Angle of ``v`` away from the +x axis, in degrees on (-180, 180].

    Positive means the vector tilts upward.
    """
    u = unit(v)
    if u is None:
        return None
    return float(np.degrees(np.arctan2(u[1], u[0])))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two points."""
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0
