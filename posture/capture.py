"""Webcam capture, rate-limited to the sampling rate we actually want.

Privacy invariant
-----------------
Frames live in memory, get turned into numbers, and are dropped. Nothing in
this module (or anywhere else in the package) writes, encodes or transmits an
image. ``tests/test_no_frame_persistence.py`` enforces that by scanning the
source, so the rule survives future edits rather than depending on discipline.

Why a thread, and how staleness is handled
------------------------------------------
A UVC webcam free-runs at 30 fps whether we want the frames or not, and the
driver buffers what we do not read. The usual fix is a thread that calls
``grab()`` continuously to keep that buffer empty. Measured here, that costs
5.5% of a core on its own, and a variant that blocks until it can prove it
holds a brand-new frame costs 8.3% -- most of a 10%-of-one-core budget, spent
entirely on frames we throw away.

What the buffer actually does was worth measuring rather than assuming. Reading
one frame every 200 ms and never draining, the backlog on this machine sits at
2-3 frames after 30 seconds and stays there: the driver keeps a small ring and
drops the oldest, so lag is bounded at roughly 70 ms. Against a 5 Hz sample
rate and a 60-second detection window, that is nothing.

So the default is to sleep until a sample is due and take one frame, which is
the cheap path. What would genuinely hurt is a backend that lets the backlog
*grow*, because then we would drift further behind real time all day. That is
what the periodic probe watches for, and draining switches on only if it
happens. The rule keys on growth, not on depth, because depth alone does not
distinguish a harmless ring from a leak.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Tried in order. DSHOW goes first because Media Foundation is pathologically
# slow to open some USB cameras: measured on this machine, the same Logitech
# device takes 1.1 s through DirectShow and 67.5 s through MSMF. Probing
# several indices with MSMF first makes the app look like it has hung. MSMF
# stays as a fallback because it handles some integrated cameras that DSHOW
# refuses, and ANY lets OpenCV decide on platforms where neither applies.
_BACKENDS: tuple[tuple[str, int], ...] = (
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
)

_BACKEND_BY_NAME = {name: value for name, value in _BACKENDS}

# A grab faster than this came from the driver's queue rather than the sensor.
# Comfortably below one frame period at 60 fps (16.7 ms) and well above the
# sub-millisecond cost of dequeuing something already buffered.
FAST_GRAB_MS = 3.0
# Ceiling on drain effort, so a backend that never blocks cannot turn the drain
# into a spin loop.
MAX_DRAIN = 8
# Backlog depth above which lag is worth paying to remove. A typical driver
# ring holds 2-3 frames (~70 ms at 30 fps), which is harmless; anything beyond
# this is either a much deeper buffer or one that is filling up, and both mean
# we are looking at the past.
STALE_FRAMES = 5
# How often to re-measure the backlog. Rare enough that the blocking grab it
# costs is irrelevant, frequent enough to notice a backend that starts leaking.
PROBE_INTERVAL_S = 30.0


@dataclass(frozen=True)
class Frame:
    """One decoded frame plus the wall-clock time it was taken."""

    image: np.ndarray  # BGR, as OpenCV hands it over
    t: float           # time.monotonic() at retrieve
    seq: int

    @property
    def aspect(self) -> float:
        """Width over height, needed to un-distort normalized landmark angles."""
        h, w = self.image.shape[:2]
        return w / h


class CameraOpenError(RuntimeError):
    """Raised when no backend could open the requested camera.

    Carries the per-backend reasons in ``attempts`` so callers can present them
    properly instead of picking the message apart with string surgery.
    """

    def __init__(self, index: int, attempts: dict[str, str]) -> None:
        self.index = index
        self.attempts = attempts
        detail = "; ".join(f"{name}: {why}" for name, why in attempts.items())
        super().__init__(
            f"camera index {index} could not be opened ({detail}). "
            "Check that nothing else is using it, and that camera access is "
            "allowed for desktop apps in Windows privacy settings."
        )

    @property
    def summary(self) -> str:
        """Short reason suited to a table cell rather than a paragraph."""
        if all("could not open" in why for why in self.attempts.values()):
            return "no backend could open it"
        if any("first read failed" in why for why in self.attempts.values()):
            return "opens but returns no frames (in use, or blocked by privacy settings)"
        return "; ".join(f"{n}: {w}" for n, w in self.attempts.items())


def open_camera(index: int, width: int, height: int,
                backend: str | None = None) -> tuple[cv2.VideoCapture, str]:
    """Open a camera, trying backends in order. Returns the cap and backend name."""
    candidates = (
        ((backend, _BACKEND_BY_NAME[backend]),) if backend else _BACKENDS
    )
    attempts: dict[str, str] = {}
    for name, api in candidates:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Ask the driver to keep only the newest frame. Backends vary in
            # whether they honour this, which is exactly what the queue-depth
            # probe measures rather than assumes.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap, name
            attempts[name] = "opened but first read failed"
            cap.release()
        else:
            attempts[name] = "could not open"
            cap.release()
    raise CameraOpenError(index, attempts)


class CameraCapture:
    """Background capture that yields the newest frame at ``sample_hz``."""

    def __init__(self, index: int = 0, *, sample_hz: float = 5.0, width: int = 640,
                 height: int = 480, backend: str | None = None,
                 name: str | None = None, reopen_delay: float = 2.0) -> None:
        self.index = index
        self.name = name or f"cam{index}"
        self._sample_hz = max(0.05, float(sample_hz))
        self.width = width
        self.height = height
        self.backend = backend
        self.reopen_delay = reopen_delay

        self._interval = 1.0 / self._sample_hz
        self._cap: cv2.VideoCapture | None = None
        self._backend_name: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._last_error: str | None = None
        # Re-measured on the capture thread every PROBE_INTERVAL_S, and reset
        # after a reconnect in case the device came back behaving differently.
        self._queue_depth: int | None = None
        self._deep_queue = False
        self._next_probe = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "CameraCapture":
        if self._thread is not None:
            return self
        self._cap, self._backend_name = open_camera(
            self.index, self.width, self.height, self.backend
        )
        log.info("%s: opened via %s", self.name, self._backend_name)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.name}",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._new.set()  # wake anyone blocked in read()

    def __enter__(self) -> "CameraCapture":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- properties --------------------------------------------------------

    @property
    def backend_name(self) -> str | None:
        return self._backend_name

    @property
    def sample_hz(self) -> float:
        return self._sample_hz

    def set_rate(self, hz: float) -> None:
        """Change the sampling rate while running.

        Adaptive sampling moves this several times a minute, so it has to be
        cheap and safe from another thread. Writing a float is atomic under the
        GIL and the capture loop re-reads the interval every pass, so the worst
        case is one sample taken at the previous rate.
        """
        hz = max(0.05, float(hz))
        self._sample_hz = hz
        self._interval = 1.0 / hz

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def queue_depth(self) -> int | None:
        """Backlog at the last probe, or None before the first one."""
        return self._queue_depth

    # -- capture loop ------------------------------------------------------

    def _probe_queue_depth(self, cap: cv2.VideoCapture) -> int:
        """Count how many frames the backend hands over without waiting.

        A grab that returns faster than ``FAST_GRAB_MS`` came out of the
        driver's buffer; the first slow one waited on the sensor, so everything
        before it was backlog. Costs one frame period, which is why it runs on
        an interval rather than per sample.
        """
        depth = 0
        for _ in range(MAX_DRAIN):
            t0 = time.perf_counter()
            if not cap.grab():
                break
            if (time.perf_counter() - t0) * 1000.0 >= FAST_GRAB_MS:
                break
            depth += 1
        return depth

    def _drain_to_current(self, cap: cv2.VideoCapture) -> bool:
        """Discard queued frames so the next retrieve gets a recent one.

        Only used on backends the probe found to be queuing. Stops at the first
        grab that blocks, which is the one that waited on the sensor.
        """
        for _ in range(MAX_DRAIN):
            t0 = time.perf_counter()
            if not cap.grab():
                return False
            if (time.perf_counter() - t0) * 1000.0 >= FAST_GRAB_MS:
                return True
        return True

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                self._reopen()
                continue

            if time.monotonic() >= self._next_probe:
                depth = self._probe_queue_depth(cap)
                was_deep = self._deep_queue
                self._queue_depth = depth
                self._deep_queue = depth >= STALE_FRAMES
                if self._deep_queue != was_deep:
                    log.info("%s: backlog %d frame(s), draining %s", self.name,
                             depth, "on" if self._deep_queue else "off")
                self._next_probe = time.monotonic() + PROBE_INTERVAL_S
                next_sample = time.monotonic()

            # Sleep until the next sample is due rather than burning CPU on
            # frames we would only throw away.
            wait = next_sample - time.monotonic()
            if wait > 0 and self._stop.wait(wait):
                break

            # Advance the schedule before doing any work, so a failing camera
            # cannot turn this loop into a hot retry. Anchoring to the previous
            # deadline rather than to now keeps the rate from drifting downward
            # as capture time accumulates; if we are already past the next
            # deadline (a slow frame, or the machine resumed from sleep) drop
            # the missed slot instead of bursting to catch up.
            now = time.monotonic()
            next_sample += self._interval
            if next_sample < now:
                next_sample = now + self._interval

            if self._deep_queue:
                grabbed = self._drain_to_current(cap)
            else:
                grabbed = cap.grab()
            if not grabbed:
                self._last_error = "grab failed"
                self._reopen()
                continue

            ok, image = cap.retrieve()
            if not ok or image is None:
                self._last_error = "retrieve failed"
                continue

            self._seq += 1
            # Timed after the drain, not at the top of the loop, so the
            # timestamp reflects when the sensor produced this frame rather
            # than when we started waiting for it.
            frame = Frame(image=image, t=time.monotonic(), seq=self._seq)
            with self._lock:
                self._latest = frame
                self._last_error = None
            self._new.set()

    def _reopen(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._queue_depth = None
        self._next_probe = 0.0
        if self._stop.wait(self.reopen_delay):
            return
        try:
            self._cap, self._backend_name = open_camera(
                self.index, self.width, self.height, self.backend
            )
            log.info("%s: reopened via %s", self.name, self._backend_name)
            self._last_error = None
        except CameraOpenError as exc:
            self._last_error = str(exc)

    # -- consumption -------------------------------------------------------

    def read(self, timeout: float = 2.0) -> Frame | None:
        """Block until the next sample is ready. None on timeout or shutdown."""
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        with self._lock:
            return self._latest

    def frames(self, timeout: float = 2.0) -> Iterator[Frame | None]:
        """Yield frames until stopped. Yields None on a timeout so callers can
        notice a dead camera and surface it rather than hanging forever."""
        while not self._stop.is_set():
            yield self.read(timeout)
