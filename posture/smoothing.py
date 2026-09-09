"""A short rolling median over each metric, applied before anything judges it.

Why this exists
---------------
The landmarks jitter. Measured on a real side camera, neck tilt -- the angle of
the ear-over-shoulder segment from vertical -- moved like this at two samples a
second::

    +13.4  -0.1  +0.6  +19.4  +4.8  +17.8  +6.3  +28.1  +16.4  +11.3

A neck does not do that. Averaged over 75 seconds the raw signal moved 3.9
degrees between consecutive samples, all day, on a person sitting still enough
that the detector called their posture good the whole time. It is the ear
landmark moving, not the ear.

The angle amplifies it. Ear-to-shoulder is the shortest segment the app
measures, so a couple of pixels of wobble at either end is worth several
degrees -- and on a desk camera it is often the *only* side metric available,
because the hips are under the desk. So the noisiest measurement in the app
carries the side camera on its own.

Nothing downstream was built to absorb that. The detector's window is robust to
it by design, but the ratio, the posture score and the number on the panel all
read the newest sample: over one 75-second stretch of posture the detector
called good throughout, the score wandered between 57 and 86.

What it does
------------
A rolling median, not a mean: jitter arrives as spikes, and a mean drags toward
them while a median steps over them. The same 75 seconds through a 1.5-second
median moves 0.9 degrees per sample instead of 3.9, and what is left reads like
a person -- sitting at 5, leaning to 17, easing back to 11 -- rather than hash.

The window is in seconds rather than samples because adaptive sampling changes
the rate underneath this, and a fixed sample count would mean a window that
silently stretched to twelve seconds when the rate dropped to idle.

It sits in the worker, ahead of both the detector and calibration, so a
baseline is measured from the same signal it will later be judged against. A
tolerance derived from the spread of a raw stream and then applied to a smooth
one would be too loose, and the reverse too tight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from statistics import median

from . import metrics as met

# Long enough to span several samples at any rate the app actually uses, short
# enough that the lag it adds -- about half the window -- is invisible against
# alert thresholds measured in tens of seconds.
WINDOW_SECONDS = 1.5


class SampleSmoother:
    """Per-metric rolling median for one camera's stream of samples."""

    def __init__(self, seconds: float = WINDOW_SECONDS) -> None:
        self.seconds = seconds
        self._history: dict[str, deque[tuple[float, float]]] = {}

    def add(self, sample: met.MetricSample) -> met.MetricSample:
        """Return ``sample`` with its values replaced by their rolling medians.

        Only the values change. Which metrics are present, which landmarks were
        missing and every note the sample carries are passed through untouched,
        so "I can see you but not your hips" still means exactly that -- a
        metric absent from this frame is never filled in from history, because
        a remembered value is not a measurement.
        """
        if not sample.values:
            return sample
        smoothed: dict[str, float] = {}
        for key, value in sample.values.items():
            history = self._history.get(key)
            if history is None:
                history = self._history[key] = deque()
            history.append((sample.t, value))
            # Pruning against this sample's own clock is what lets a metric
            # vanish for a minute and come back clean: everything older than
            # the window goes, however long the gap was.
            cutoff = sample.t - self.seconds
            while history and history[0][0] < cutoff:
                history.popleft()
            smoothed[key] = median(v for _t, v in history)
        return replace(sample, values=smoothed)

    def reset(self) -> None:
        """Forget everything. For when the stream is no longer continuous."""
        self._history.clear()
