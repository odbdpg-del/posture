"""Landmark naming, indexing and extraction.

Isolates every assumption about MediaPipe's landmark layout in one place so the
metric code can talk in body-part names instead of magic integers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NUM_LANDMARKS = 33

# The subset we actually care about for posture. The pose model returns all 33,
# but naming only these keeps the intent obvious and the tests readable.
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

NAMES: dict[int, str] = {
    NOSE: "nose",
    LEFT_EYE: "left_eye",
    RIGHT_EYE: "right_eye",
    LEFT_EAR: "left_ear",
    RIGHT_EAR: "right_ear",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
}

# Per-side landmark triples used by the side-view metrics.
SIDE_CHAIN = {
    "left": (LEFT_EAR, LEFT_SHOULDER, LEFT_HIP),
    "right": (RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP),
}


def result_to_array(result: Any) -> np.ndarray | None:
    """Flatten a PoseLandmarkerResult into an (N, 4) array of x, y, z, visibility.

    Returns None when the model found no person at all. A returned array may
    still contain landmarks with low visibility; filtering that is the caller's
    job, because "I can see you but not your hip" and "nobody is here" are
    different states that the UI has to distinguish.
    """
    poses = getattr(result, "pose_landmarks", None)
    if not poses:
        return None
    lms = poses[0]
    if not lms:
        return None
    out = np.zeros((len(lms), 4), dtype=np.float64)
    for i, lm in enumerate(lms):
        out[i, 0] = lm.x or 0.0
        out[i, 1] = lm.y or 0.0
        out[i, 2] = lm.z or 0.0
        # The model always sets visibility; presence is a useful secondary
        # gate, so take the weaker of the two when both are available.
        vis = 1.0 if lm.visibility is None else float(lm.visibility)
        pres = 1.0 if lm.presence is None else float(lm.presence)
        out[i, 3] = min(vis, pres)
    return out


def make_synthetic(points: dict[int, tuple[float, float]], visibility: float = 1.0) -> np.ndarray:
    """Build a landmark array from a sparse dict of normalized coordinates.

    Landmarks not named in ``points`` get visibility 0, so tests can construct
    exactly the body parts a metric needs and nothing else.
    """
    arr = np.zeros((NUM_LANDMARKS, 4), dtype=np.float64)
    for idx, (x, y) in points.items():
        arr[idx] = (x, y, 0.0, visibility)
    return arr
