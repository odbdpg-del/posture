"""Runs several cameras at once and publishes what each of them sees.

Each camera gets its own thread, its own capture, and its own pose landmarker.
Nothing is shared between them and no attempt is made to line frames up in
time: the streams are processed independently and only their derived metrics
are ever combined. That is what keeps a single camera fully functional, and it
means one camera dying cannot take the others with it.

The supervisor owns the threads; the UI only ever reads :meth:`Monitor.snapshot`,
which returns a plain dictionary built under a lock. No UI code touches a
camera, a frame, or a landmarker.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import alerts as al
from . import detector as det
from . import metrics as met
from . import score as scoring
from .history import History
from .notify import Notifier
from .overlay_window import OverlayView, OverlayWindow
from .store import Store
from .calibration import Baseline, CalibrationSession
from .capture import CameraCapture, CameraOpenError
from .config import CameraConfig, Config, retains_frames
from .devices import describe
from .pose import PoseEstimator
from .smoothing import SampleSmoother

log = logging.getLogger(__name__)

# States a camera can be in, in the order the UI should treat as increasingly
# wrong. "cannot_see" is deliberately distinct from "no_person": one is a
# person we cannot measure, the other is an empty chair, and conflating them is
# how you end up alerting at furniture.
STARTING = "starting"
TRACKING = "tracking"
PARTIAL = "partial"
CANNOT_SEE = "cannot_see"
NO_PERSON = "no_person"
ERROR = "error"
STOPPED = "stopped"

STATE_LABELS = {
    STARTING: "Starting up",
    TRACKING: "Tracking",
    PARTIAL: "Tracking, some metrics unavailable",
    CANNOT_SEE: "Cannot see you properly",
    NO_PERSON: "Nobody in view",
    ERROR: "Camera problem",
    STOPPED: "Off",
}

# Landmarks worth sending to the UI for the placement preview. The full 33
# would work too, but these are the ones the metrics actually read, and a
# smaller payload keeps the poll cheap.
PREVIEW_LANDMARKS = (0, 2, 5, 7, 8, 11, 12, 13, 14, 23, 24, 25, 26)


@dataclass
class CameraState:
    """Everything the UI needs to know about one camera."""

    index: int
    role: str
    name: str = ""
    enabled: bool = True
    state: str = STARTING
    error: str = ""
    backend: str | None = None
    size: tuple[int, int] | None = None
    fps: float = 0.0
    infer_ms: float = 0.0
    sample: met.MetricSample | None = None
    landmarks: list[list[float]] = field(default_factory=list)
    visibility: dict[str, float] = field(default_factory=dict)
    updated: float = 0.0

    def to_dict(self) -> dict:
        sample = self.sample
        return {
            "index": self.index,
            "name": self.name,
            "role": self.role,
            "enabled": self.enabled,
            "state": self.state,
            "state_label": STATE_LABELS.get(self.state, self.state),
            "error": self.error,
            "backend": self.backend,
            "size": list(self.size) if self.size else None,
            "fps": round(self.fps, 2),
            "infer_ms": round(self.infer_ms, 1),
            "age": round(max(0.0, time.monotonic() - self.updated), 1) if self.updated else None,
            "landmarks": self.landmarks,
            "visibility": {k: round(v, 2) for k, v in self.visibility.items()},
            "metrics": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "unit": spec.unit,
                    "description": spec.description,
                    "value": (None if sample is None else
                              (round(sample.values[spec.key], 2)
                               if spec.key in sample.values else None)),
                }
                for spec in met.SPECS_BY_ROLE.get(self.role, ())
            ],
            "confidence": ({k: round(v, 3) for k, v in sample.confidence.items()}
                           if sample else {}),
            "missing": list(sample.missing) if sample else [],
            "notes": list(sample.notes) if sample else [],
            "near_side": sample.near_side if sample else None,
            "facing": sample.facing if sample else None,
            "scale_kind": sample.scale_kind if sample else None,
        }


class CameraWorker:
    """One camera: capture, landmarks, metrics, published state."""

    def __init__(self, cam: CameraConfig, cfg: Config,
                 on_sample=None) -> None:
        self.cam = cam
        self.cfg = cfg
        # Called with every computed sample. The monitor uses it to drive the
        # detector from the worker thread rather than polling for samples,
        # which would either miss some or double-count them.
        self._on_sample = on_sample
        self._calibration: CalibrationSession | None = None
        self._capture: CameraCapture | None = None
        # Newest frame and landmarks, kept only when the video preview is on.
        # With it off nothing is retained past the loop iteration, which is the
        # behaviour the rest of the app assumes.
        self._preview_lock = threading.Lock()
        self._preview: tuple = ()
        # Ahead of both the detector and calibration, so a baseline is measured
        # from the same signal it will later be judged against.
        self._smoother = SampleSmoother()
        self.state = CameraState(
            index=cam.index, role=cam.role, name=cam.name or describe(cam.index),
            enabled=cam.enabled,
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"worker-{self.cam.index}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self.state.state = STOPPED

    def snapshot(self) -> dict:
        with self._lock:
            data = self.state.to_dict()
        session = self._calibration
        if session is not None and session.running:
            data["calibrating"] = {
                "progress": round(session.progress, 3),
                "elapsed": round(session.elapsed(), 1),
                "duration": session.duration,
                "counts": session.counts,
            }
        capture = self._capture
        data["sample_hz"] = round(capture.sample_hz, 2) if capture else None
        return data

    def set_rate(self, hz: float) -> None:
        capture = self._capture
        if capture is not None:
            capture.set_rate(hz)

    def latest_preview(self) -> tuple:
        """Newest (image, landmarks) for the panel, or () when preview is off.

        The image is handed over without copying: the capture loop replaces the
        reference rather than mutating the array, so the reader always sees a
        whole frame.
        """
        with self._preview_lock:
            return self._preview

    def start_calibration(self) -> CalibrationSession:
        """Begin a baseline capture on this camera."""
        session = CalibrationSession(
            self.cam.index, self.cam.role,
            duration=self.cfg.calibration.duration,
            min_samples=self.cfg.calibration.min_samples,
        )
        session.start()
        self._calibration = session
        return session

    def cancel_calibration(self) -> None:
        self._calibration = None

    @property
    def calibration(self) -> CalibrationSession | None:
        return self._calibration

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self.state, key, value)

    def _run(self) -> None:
        from . import landmarks as lmk

        capture = estimator = None
        try:
            estimator = PoseEstimator(
                self.cfg.model_path,
                min_detection_confidence=self.cfg.sampling.min_detection_confidence,
                min_presence_confidence=self.cfg.sampling.min_presence_confidence,
                min_tracking_confidence=self.cfg.sampling.min_tracking_confidence,
            )
            capture = CameraCapture(
                self.cam.index, sample_hz=self.cfg.sampling.fps,
                width=self.cam.width, height=self.cam.height,
                backend=self.cam.backend,
            ).start()
            self._capture = capture
            self._set(backend=capture.backend_name, error="")
        except (CameraOpenError, FileNotFoundError) as exc:
            log.warning("camera %s failed to start: %s", self.cam.index, exc)
            self._set(state=ERROR, error=str(exc))
            if estimator is not None:
                estimator.close()
            return

        thresh = self.cfg.sampling.visibility_threshold
        interval_start = time.monotonic()
        interval_frames = 0
        try:
            while not self._stop.is_set():
                frame = capture.read(timeout=2.0)
                if frame is None:
                    self._set(state=ERROR,
                              error=capture.last_error or "no frames from camera")
                    continue

                result = estimator.detect(frame.image, int(frame.t * 1000))
                sample = self._smoother.add(met.compute(
                    self.cam.role, result.array, frame.aspect, thresh, frame.t,
                    camera_index=self.cam.index))

                if not sample.person:
                    state = NO_PERSON
                elif sample.complete:
                    state = TRACKING
                elif sample.usable:
                    state = PARTIAL
                else:
                    state = CANNOT_SEE

                preview: list[list[float]] = []
                visibility: dict[str, float] = {}
                if result.array is not None:
                    arr = result.array
                    preview = [[round(float(arr[i, 0]), 4), round(float(arr[i, 1]), 4),
                                round(float(arr[i, 3]), 3)] for i in PREVIEW_LANDMARKS]
                    visibility = {name: float(arr[i, 3]) for i, name in lmk.NAMES.items()}

                interval_frames += 1
                now = time.monotonic()
                fps = self.state.fps
                if now - interval_start >= 2.0:
                    fps = interval_frames / (now - interval_start)
                    interval_start, interval_frames = now, 0

                h, w = frame.image.shape[:2]
                self._set(state=state, sample=sample, landmarks=preview,
                          visibility=visibility, fps=fps, infer_ms=result.infer_ms,
                          size=(w, h), updated=now, error="")

                if retains_frames(self.cfg):
                    with self._preview_lock:
                        self._preview = (frame.image, result.array)
                else:
                    with self._preview_lock:
                        self._preview = ()

                session = self._calibration
                if session is not None and session.running:
                    session.add(sample, landmarks=preview)
                if self._on_sample is not None:
                    self._on_sample(sample)
        finally:
            self._capture = None
            if capture is not None:
                capture.stop()
            if estimator is not None:
                estimator.close()
            with self._lock:
                if self.state.state != ERROR:
                    self.state.state = STOPPED


class Monitor:
    """Supervises one worker per enabled camera, and the detector they feed."""

    def __init__(self, cfg: Config, on_config_change=None) -> None:
        self.cfg = cfg
        # Called when the monitor changes config itself, which today means a
        # finished calibration. Keeps this class free of any opinion about
        # where config lives or when it is written.
        self._on_config_change = on_config_change
        self._workers: list[CameraWorker] = []
        self._lock = threading.Lock()
        self._detector = det.PostureDetector(_settings(cfg), _baselines(cfg))
        self._verdict = self._detector.update([], now=time.monotonic())
        self._calibration_result: dict | None = None
        # Set while a baseline capture is running. Adaptive sampling must not
        # apply during calibration: the rate follows the detector, and an
        # uncalibrated detector reports GOOD or AWAY, which would throttle the
        # capture to the idle rate and collect a fraction of the samples the
        # ten seconds are supposed to buy.
        self._calibrating = False
        self.store = _open_store(cfg)
        self.history = History(store=self.store)
        self._alerts = al.AlertEngine(_alert_settings(cfg))
        self._alert_state = self._alerts.state()
        # Effects are created lazily: a Notifier picks a backend and the
        # overlay opens a Tk root, and neither should happen just because
        # something imported Monitor.
        self._notifier: Notifier | None = None
        self._overlay: OverlayWindow | None = None
        self._score = scoring.PostureScore(None, "starting", "idle")
        self.started_at = time.monotonic()

    def set_config_change_hook(self, fn) -> None:
        """Register where to persist config the monitor changes itself.

        Set after construction because the control panel, which owns the config
        path, is built around the monitor rather than before it.
        """
        self._on_config_change = fn

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        self._workers = [CameraWorker(cam, self.cfg, on_sample=self._ingest)
                         for cam in self.cfg.cameras if cam.enabled]
        for worker in self._workers:
            worker.start()
        log.info("monitor started with %d camera(s)", len(self._workers))

    def stop(self) -> None:
        with self._lock:
            workers, self._workers = self._workers, []
        for worker in workers:
            worker.stop()
        if self._overlay is not None:
            self._overlay.stop()
            self._overlay = None
        if self.store is not None:
            self.store.close()
            self.store = None

    def apply_config(self, cfg: Config) -> None:
        """Swap in a new config by restarting the workers.

        Restarting everything is heavier than diffing which cameras changed,
        but config edits are a human-scale event and a camera takes a second to
        reopen. The simpler code is worth more here than the saved second, and
        it cannot leave a worker running against stale settings.
        """
        with self._lock:
            workers, self._workers = self._workers, []
        for worker in workers:
            worker.stop()
        with self._lock:
            self.cfg = cfg
            self._detector.set_settings(_settings(cfg))
            self._detector.set_baselines(_baselines(cfg))
            self._alerts.set_settings(_alert_settings(cfg))
            self._start_locked()

    # -- the detector ------------------------------------------------------

    def _ingest(self, sample: met.MetricSample) -> None:
        """Called from a worker thread for every sample it computes.

        Driving the detector from the producing thread means every sample is
        seen exactly once, at the moment it was taken. Polling for samples
        instead would drop them when a camera runs faster than the poll and
        repeat them when it runs slower, and adaptive sampling guarantees both
        happen.
        """
        with self._lock:
            verdict = self._detector.update([sample])
            self._verdict = verdict
            # Score and history are derived views of the verdict, updated here
            # so every sample is reflected exactly once. Both are read-only to
            # everything downstream.
            self._score = scoring.score_verdict(
                verdict, self._detector.settings.bad_fraction)
            self.history.record(self._score.value, verdict.state, verdict)
            self._alert_state = self._alerts.update(verdict)
            events = self._alerts.drain()
            workers = list(self._workers)
            sampling = self.cfg.sampling

        # Effects run outside the lock: a notification shells out and the
        # overlay talks to another thread, and neither should be able to stall
        # the capture loop that called us.
        self._apply_alert_effects(events)

        if not sampling.adaptive or self._calibrating:
            return
        hz = det.choose_rate(
            verdict.state, verdict.worst_ratio,
            peak_hz=sampling.fps, idle_hz=sampling.idle_fps,
            away_hz=sampling.away_fps, active_ratio=sampling.active_ratio,
        )
        for worker in workers:
            worker.set_rate(hz)

    def latest_preview(self, camera: int) -> tuple:
        """Newest (image, landmarks, role) for one camera, or () if unavailable.

        Empty means all of: no such camera, preview switched off, or the camera
        has started but not yet produced a frame. The last one is a real window
        of a second or so on every start, so it has to be a clean "nothing yet"
        rather than a partly-filled tuple -- appending the role to an empty
        preview would produce a truthy 1-tuple that the caller then fails to
        unpack.
        """
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            if worker.cam.index == camera:
                preview = worker.latest_preview()
                if len(preview) != 2:
                    return ()
                return preview + (worker.cam.role,)
        return ()

    @property
    def verdict(self) -> det.PostureVerdict:
        with self._lock:
            return self._verdict

    @property
    def score(self) -> scoring.PostureScore:
        with self._lock:
            return self._score

    @property
    def alert(self) -> al.AlertState:
        with self._lock:
            return self._alert_state

    # -- alert effects -----------------------------------------------------

    def _apply_alert_effects(self, events: list) -> None:
        cfg = self.cfg.alerts
        state = self._alert_state

        for event in events:
            # Every transition is recorded, not just escalations: the
            # daily alert count is more honest when a snooze is visible
            # in the same table as the alert it silenced.
            if self.store is not None:
                try:
                    self.store.add_alert(time.time(), event.level, event.kind,
                                         event.state.offenders)
                except Exception:
                    log.debug('could not record an alert', exc_info=True)
            if (event.kind == "escalate" and event.level == al.LEVEL_NOTIFY
                    and cfg.os_notifications):
                if self._notifier is None:
                    self._notifier = Notifier()
                self._notifier.send(event.state.headline or "Fix your posture",
                                    event.state.detail)
            elif event.kind == "stood_down" and cfg.os_notifications:
                # The window vanishing with no explanation would read as a
                # glitch, and the thing worth saying is not "you are fine" --
                # it is that the app was asking for something unreachable and
                # has stopped. Silence here would leave the underlying problem
                # to be rediscovered the next time it escalates.
                if self._notifier is None:
                    self._notifier = Notifier()
                self._notifier.send("Posture alert stood down",
                                    self._stand_down_detail())

        if not cfg.fullscreen_overlay:
            if self._overlay is not None:
                self._overlay.hide()
            return

        want = state.level >= al.LEVEL_OVERLAY
        if want and self._overlay is None:
            self._overlay = OverlayWindow(on_snooze=self.snooze)
        if self._overlay is None:
            return
        if want and not self._overlay.visible:
            self._overlay.show(state.headline, state.detail,
                               state.hold_required, state.hold_remaining,
                               self._overlay_view())
        elif want:
            self._overlay.update(state.headline, state.detail,
                                 state.hold_required, state.hold_remaining,
                                 self._overlay_view())
        elif self._overlay.visible:
            self._overlay.hide()

    def _stand_down_detail(self) -> str:
        """Why the overlay gave up, in terms the person can act on."""
        suspect = self._verdict.suspect
        if suspect:
            return (f"{', '.join(suspect)} cannot be satisfied by any normal "
                    "posture -- the baseline was captured somewhere you do not "
                    "sit. Recalibrate to start scoring it again.")
        return ("It was asking for a posture that never arrived in ten minutes. "
                "Check the camera still sees you the way it did at calibration, "
                "and recalibrate if it has moved.")

    def _overlay_view(self) -> OverlayView:
        """The figure and tracking state the overlay should draw.

        Landmarks only. Which camera they come from matters when there are two:
        the one whose role owns the metric being complained about is the one
        you have to get back in front of, so it wins even if it has lost sight
        of you -- that is exactly the state worth showing. Otherwise any camera
        that can currently see you will do.
        """
        with self._lock:
            workers = list(self._workers)
            verdict = self._verdict
            thresh = self.cfg.sampling.visibility_threshold
        if not workers:
            return OverlayView(state=verdict.state, reason=verdict.reason)

        roles = {met.SPEC_BY_KEY[k].role for k in verdict.offenders
                 if k in met.SPEC_BY_KEY}
        chosen = next((w for w in workers if w.cam.role in roles), None)
        if chosen is None:
            chosen = next((w for w in workers
                           if w.state.state in (TRACKING, PARTIAL)), workers[0])

        cam = chosen.state
        size = cam.size
        return OverlayView(
            landmarks=tuple(tuple(lm) for lm in cam.landmarks),
            thresh=thresh,
            aspect=(size[0] / size[1]) if size and size[1] else 4 / 3,
            state=verdict.state,
            reason=verdict.reason,
        )

    def snooze(self, seconds: float | None = None) -> dict:
        """Suppress alerts for a fixed stretch. Explicit and time-boxed."""
        with self._lock:
            self._alert_state = self._alerts.snooze(seconds)
        self._alerts.drain()
        if self._overlay is not None:
            self._overlay.hide()
        return self._alert_state.to_dict()

    def cancel_snooze(self) -> dict:
        with self._lock:
            self._alert_state = self._alerts.cancel_snooze()
        self._alerts.drain()
        return self._alert_state.to_dict()

    # -- calibration -------------------------------------------------------

    def start_calibration(self, camera: int | None = None) -> dict:
        """Begin baseline capture. With no camera named, calibrates all of them.

        Calibrating every camera at once is the normal case and the right
        default: you sit well once, and each camera records what that looks
        like from where it happens to stand.
        """
        with self._lock:
            workers = [w for w in self._workers
                       if camera is None or w.cam.index == camera]
        if not workers:
            return {"ok": False, "error": "no running camera to calibrate"}
        self._calibration_result = None
        self._calibrating = True
        for worker in workers:
            # Full rate for the whole capture, so ten seconds means the ~50
            # samples the statistics assume rather than whatever the adaptive
            # rate happened to be sitting at.
            worker.set_rate(self.cfg.sampling.fps)
            worker.start_calibration()
        return {"ok": True, "cameras": [w.cam.index for w in workers],
                "duration": self.cfg.calibration.duration}

    def cancel_calibration(self) -> dict:
        with self._lock:
            workers = list(self._workers)
        self._calibrating = False
        for worker in workers:
            worker.cancel_calibration()
        return {"ok": True}

    def collect_calibration(self) -> dict | None:
        """Finish any calibration whose capture window has elapsed.

        Polled from the status endpoint rather than run on a timer: the result
        only matters when someone is looking, and this keeps the whole flow on
        one thread instead of racing the workers for it.
        """
        with self._lock:
            workers = list(self._workers)
        pending = [w for w in workers if w.calibration is not None]
        if not pending:
            return self._calibration_result
        if any(w.calibration is not None and w.calibration.running for w in pending):
            return self._calibration_result

        results: list[dict] = []
        problems: list[str] = []
        baselines = dict(self.cfg.baselines)
        for worker in pending:
            session = worker.calibration
            if session is None:
                continue
            baseline, issues = session.result()
            worker.cancel_calibration()
            if baseline.metrics:
                baselines[str(baseline.camera)] = baseline.to_dict()
            results.append({
                "camera": baseline.camera,
                "name": worker.state.name,
                "role": baseline.role,
                "complete": baseline.complete,
                "missing": list(baseline.missing),
                "metrics": {k: v.to_dict() for k, v in baseline.metrics.items()},
            })
            problems.extend(f"{worker.state.name}: {p}" for p in issues)
            problems.extend(
                f"{worker.state.name}: {p}" for p in
                baseline.quality_notes(self.cfg.detection.tolerance_multiplier))

        self._calibrating = False
        with self._lock:
            self.cfg.baselines = baselines
            self._detector.set_baselines(_baselines(self.cfg))
        if self._on_config_change is not None:
            self._on_config_change(self.cfg)

        self._calibration_result = {
            "ok": any(r["metrics"] for r in results),
            "at": time.time(),
            "cameras": results,
            "problems": problems,
        }
        return self._calibration_result

    def clear_baselines(self) -> dict:
        """Throw the calibration away and go back to having no opinion."""
        with self._lock:
            self.cfg.baselines = {}
            self._detector.set_baselines({})
        self._calibration_result = None
        if self._on_config_change is not None:
            self._on_config_change(self.cfg)
        return {"ok": True}

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        calibration = self.collect_calibration()
        with self._lock:
            workers = list(self._workers)
            verdict = self._verdict
            score = self._score
            alert = self._alert_state
            baselines = dict(self.cfg.baselines)
            configured = len([c for c in self.cfg.cameras if c.enabled])
        cameras = [w.snapshot() for w in workers]
        # Metrics a running camera is producing right now that calibration has
        # no baseline for. Happens whenever a new metric ships after you last
        # calibrated, and would otherwise be invisible: the detector simply has
        # no window for it and it never appears anywhere.
        uncalibrated: set[str] = set()
        for worker in workers:
            sample = worker.state.sample
            if sample is None:
                continue
            base = baselines.get(str(worker.cam.index)) or {}
            have = set((base.get("metrics") or {}))
            for key in sample.values:
                if key not in have:
                    uncalibrated.add(met.SPEC_BY_KEY[key].label
                                     if key in met.SPEC_BY_KEY else key)
        return {
            "uptime": round(time.monotonic() - self.started_at, 1),
            "cameras": cameras,
            "roles": sorted({c["role"] for c in cameras}),
            "any_tracking": any(c["state"] in (TRACKING, PARTIAL) for c in cameras),
            "running": bool(workers),
            "configured_count": configured,
            "posture": verdict.to_dict(),
            "score": score.to_dict(),
            "alert": alert.to_dict(),
            "session_average": self.history.session_average,
            "state_since": round(max(0.0, time.monotonic() - verdict.since), 1),
            "calibrating": any(c.get("calibrating") for c in cameras),
            "calibration_result": calibration,
            "baselines": baselines,
            "uncalibrated_metrics": sorted(uncalibrated),
        }


def _settings(cfg: Config) -> det.DetectionSettings:
    """Translate the config section into detector settings."""
    d = cfg.detection
    return det.DetectionSettings(
        window_seconds=d.window_seconds, bad_fraction=d.bad_fraction,
        exit_ratio=d.exit_ratio, absence_seconds=d.absence_seconds,
        tolerance_multiplier=d.tolerance_multiplier,
        min_window_fill=d.min_window_fill, overrides=dict(d.overrides),
        min_confidence=d.min_confidence,
    )


def _open_store(cfg: Config) -> Store | None:
    """Open the stats database, or carry on without one.

    Stats are a nice-to-have; posture monitoring is not. A database that cannot
    be opened costs you history and nothing else.
    """
    if not cfg.stats.enabled:
        return None
    try:
        store = Store(cfg.stats.path)
        if cfg.stats.retain_days > 0:
            removed = store.prune(time.time() - cfg.stats.retain_days * 86400)
            if removed:
                log.info("pruned %d stats row(s) older than %d days",
                         removed, cfg.stats.retain_days)
        return store
    except Exception:
        log.warning("could not open the stats database; history will not persist",
                    exc_info=True)
        return None


def _alert_settings(cfg: Config) -> al.AlertSettings:
    """Translate the config section into engine settings."""
    a = cfg.alerts
    return al.AlertSettings(
        enabled=a.enabled, subtle_after=a.subtle_after,
        notify_after=a.notify_after, overlay_after=a.overlay_after,
        clear_hold=a.clear_hold, hold_grace=a.hold_grace,
        snooze_seconds=a.snooze_seconds,
        renotify_after=a.renotify_after,
    )


def _baselines(cfg: Config) -> dict[int, Baseline]:
    """Rebuild baselines from their stored form, skipping anything malformed.

    A corrupt entry costs one camera its calibration and says so in the log,
    rather than stopping the app from starting at all.
    """
    out: dict[int, Baseline] = {}
    for key, data in (cfg.baselines or {}).items():
        try:
            out[int(key)] = Baseline.from_dict(data)
        except (TypeError, ValueError, KeyError):
            log.warning("ignoring unreadable baseline for camera %s", key)
    return out
