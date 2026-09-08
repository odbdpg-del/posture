"""Debug-window drawing: skeleton overlay and a live metric readout.

Screen only. Nothing here encodes or saves an image.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lmk
from . import metrics as met
from .pose import CONNECTIONS

# BGR, because OpenCV.
_BONE = (110, 110, 110)
_JOINT = (180, 180, 180)
_KEY_JOINT = (0, 200, 255)
_FAINT = (60, 60, 200)
_TEXT = (240, 240, 240)
_DIM = (150, 150, 150)
_GOOD = (120, 220, 120)
_WARN = (60, 200, 255)
_BAD = (80, 80, 240)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Landmarks each role actually reads, drawn larger so it is obvious at a glance
# whether the ones that matter are being tracked.
_KEY_POINTS: dict[str, tuple[int, ...]] = {
    "side": (lmk.NOSE, lmk.LEFT_EAR, lmk.RIGHT_EAR, lmk.LEFT_SHOULDER,
             lmk.RIGHT_SHOULDER, lmk.LEFT_HIP, lmk.RIGHT_HIP),
    "front": (lmk.LEFT_EAR, lmk.RIGHT_EAR, lmk.LEFT_SHOULDER, lmk.RIGHT_SHOULDER),
}


def draw_skeleton(image: np.ndarray, arr: np.ndarray | None, role: str,
                  vis_thresh: float) -> None:
    """Draw the pose in place. Faint landmarks are drawn in red so you can see
    the difference between "not tracked" and "tracked but not trusted"."""
    if arr is None:
        return
    h, w = image.shape[:2]
    pix = np.column_stack((arr[:, 0] * w, arr[:, 1] * h)).astype(np.int32)
    vis = arr[:, 3]

    for a, b in CONNECTIONS:
        if a >= len(pix) or b >= len(pix):
            continue
        if vis[a] < vis_thresh or vis[b] < vis_thresh:
            continue
        cv2.line(image, tuple(pix[a]), tuple(pix[b]), _BONE, 1, cv2.LINE_AA)

    key = _KEY_POINTS.get(role, ())
    for i in range(len(pix)):
        if vis[i] < 0.1:
            continue
        faint = vis[i] < vis_thresh
        if i in key:
            cv2.circle(image, tuple(pix[i]), 5, _FAINT if faint else _KEY_JOINT, -1,
                       cv2.LINE_AA)
        elif not faint:
            cv2.circle(image, tuple(pix[i]), 2, _JOINT, -1, cv2.LINE_AA)


def draw_panel(image: np.ndarray, sample: met.MetricSample, *, status: str,
               status_colour: tuple[int, int, int], fps: float, infer_ms: float,
               cpu_pct: float) -> None:
    """Draw the metric readout box in place."""
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"{sample.role.upper()}  {status}", status_colour))

    for spec in met.SPECS_BY_ROLE[sample.role]:
        value = sample.values.get(spec.key)
        if value is None:
            lines.append((f"  {spec.label:<15} --", _DIM))
        else:
            lines.append((f"  {spec.label:<15} {value:+8.2f} {spec.unit}", _TEXT))

    if sample.near_side:
        facing = {1: "->", -1: "<-", None: "?"}[sample.facing]
        lines.append((f"  near side {sample.near_side}, facing {facing}", _DIM))
    if sample.scale is not None:
        lines.append((f"  scale ({sample.scale_kind}) {sample.scale:.3f}", _DIM))
    if sample.missing:
        lines.append((f"  faint: {', '.join(sample.missing)}", _WARN))
    for note in sample.notes:
        lines.append((f"  ! {note}", _WARN))
    lines.append((f"  {fps:4.1f} fps   {infer_ms:5.1f} ms/frame   {cpu_pct:4.1f}% cpu", _DIM))
    # Spelled out on the window itself. Closing it should be obvious without
    # remembering which terminal launched it -- and if this window is in the
    # way of something, hunting for that terminal is the last thing you want.
    lines.append(("  press q or Esc to quit", _DIM))

    pad = 8
    line_h = 18
    box_h = min(image.shape[0], pad * 2 + line_h * len(lines))
    box_w = min(image.shape[1], 340)
    # Darken the region in place. Blending against a black image of the same
    # size, rather than filling first, keeps the video faintly visible behind
    # the text instead of blacking it out.
    region = image[0:box_h, 0:box_w]
    cv2.addWeighted(np.zeros_like(region), 0.55, region, 0.45, 0, region)

    y = pad + 12
    for text, colour in lines:
        cv2.putText(image, text, (pad, y), _FONT, 0.42, colour, 1, cv2.LINE_AA)
        y += line_h


def status_for(sample: met.MetricSample) -> tuple[str, tuple[int, int, int]]:
    """Map a sample onto the three states the UI must distinguish."""
    if not sample.person:
        return "NO PERSON", _DIM
    if not sample.usable:
        return "CANNOT SEE YOU PROPERLY", _BAD
    if not sample.complete:
        return "PARTIAL", _WARN
    return "TRACKING", _GOOD
