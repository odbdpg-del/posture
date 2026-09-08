"""Thin wrapper around MediaPipe's PoseLandmarker.

Stack note: MediaPipe 1.0 removed the old ``mp.solutions.pose`` API, so this
uses Tasks Vision, which needs an external ``.task`` model bundle. That bundle
is fetched once by ``scripts/fetch_model.py`` and lives on disk; the running
app never touches the network.

Running mode is VIDEO rather than IMAGE. In VIDEO mode the landmarker reuses
the previous frame's region of interest and skips the (expensive) whole-frame
detector on most frames, which is the single biggest lever we have on the CPU
budget. It costs us a constraint: timestamps must increase strictly and each
landmarker instance must be used from one thread at a time, so every camera
gets its own instance.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from . import landmarks as lmk

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PACKAGE_ROOT / "models" / "pose_landmarker_lite.task"

MODEL_MISSING_HINT = (
    "Pose model not found at {path}.\n"
    "Fetch it once with:  python scripts/fetch_model.py\n"
    "This is the only network access the project ever makes; once the file is "
    "on disk the app runs fully offline."
)

# Skeleton edges for the debug overlay, taken from the model's own topology so
# it cannot drift out of sync with the landmark indices.
CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
)


@dataclass(frozen=True)
class PoseResult:
    """Landmarks for one frame.

    ``array`` is None when no person was found at all, which is a different
    condition from a person whose landmarks are too faint to use.
    """

    array: np.ndarray | None
    infer_ms: float

    @property
    def person(self) -> bool:
        return self.array is not None


class PoseEstimator:
    """One MediaPipe landmarker, bound to one camera stream."""

    def __init__(self, model_path: str | os.PathLike[str] | None = None, *,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(MODEL_MISSING_HINT.format(path=path))
        self.model_path = path
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> PoseResult:
        """Run the landmarker on one BGR frame.

        ``timestamp_ms`` must increase between calls; we nudge it forward if a
        caller hands us a duplicate rather than letting MediaPipe raise.
        """
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return PoseResult(array=lmk.result_to_array(result), infer_ms=infer_ms)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - shutdown best effort
            log.debug("landmarker close failed", exc_info=True)

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
