"""Monitor wiring: adaptive rate, calibration gating, baseline persistence.

The camera workers are replaced with fakes, so this exercises the supervisor's
logic without opening a device or loading a model.
"""

from __future__ import annotations

import pytest

from posture import detector as det
from posture import metrics as met
from posture import monitor as mon
from posture.calibration import Baseline, MetricBaseline
from posture.config import CameraConfig, Config


class FakeWorker:
    """Stands in for CameraWorker: records rate changes, fakes calibration."""

    def __init__(self, index: int = 1, role: str = "side") -> None:
        self.cam = CameraConfig(index=index, role=role, name=f"cam{index}")
        self.state = mon.CameraState(index=index, role=role, name=f"cam{index}")
        self.rates: list[float] = []
        self.calibration = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def set_rate(self, hz: float) -> None:
        self.rates.append(hz)

    def latest_preview(self) -> tuple:
        """Empty by default: a freshly started camera has no frame yet."""
        return ()

    def start_calibration(self):
        self.calibration = FakeSession(self.cam.index, self.cam.role)
        return self.calibration

    def cancel_calibration(self) -> None:
        self.calibration = None

    def snapshot(self) -> dict:
        return {"index": self.cam.index, "role": self.cam.role, "state": "tracking"}


class FakeSession:
    def __init__(self, camera: int, role: str) -> None:
        self.camera, self.role = camera, role
        self.running = True
        self.duration = 10.0

    def finish(self, spread: float = 1.0) -> None:
        self.running = False
        self._spread = spread

    def result(self):
        baseline = Baseline(
            camera=self.camera, role=self.role,
            metrics={"neck_flexion": MetricBaseline("neck_flexion", 170.0,
                                                    getattr(self, "_spread", 1.0), 50)},
        )
        return baseline, []


def make_monitor(**sampling) -> tuple[mon.Monitor, list[FakeWorker]]:
    cfg = Config()
    # Explicit as well as the conftest guard: a Monitor opens a stats database,
    # and a default config points that at the user's real one.
    cfg.stats.path = ":memory:"
    cfg.cameras = [CameraConfig(index=1, role="side", enabled=True)]
    for key, value in sampling.items():
        setattr(cfg.sampling, key, value)
    monitor = mon.Monitor(cfg)
    workers = [FakeWorker(1)]
    monitor._workers = workers  # noqa: SLF001 - substituting the device layer
    return monitor, workers


def sample(values: dict[str, float], camera: int = 1) -> met.MetricSample:
    return met.MetricSample(role="side", t=0.0, person=True, values=dict(values),
                            camera_index=camera)


class TestAdaptiveRate:
    def test_rate_follows_the_verdict(self):
        monitor, workers = make_monitor(fps=5.0, idle_fps=2.0, away_fps=0.5)
        monitor.cfg.baselines = {"1": Baseline(
            camera=1, role="side",
            metrics={"neck_flexion": MetricBaseline("neck_flexion", 170.0, 1.0, 50)},
        ).to_dict()}
        monitor._detector.set_baselines(mon._baselines(monitor.cfg))  # noqa: SLF001
        monitor._ingest(sample({"neck_flexion": 170.0}))  # noqa: SLF001
        assert workers[0].rates[-1] == 2.0
        monitor._ingest(sample({"neck_flexion": 120.0}))  # noqa: SLF001
        assert workers[0].rates[-1] == 5.0

    def test_adaptive_can_be_switched_off(self):
        monitor, workers = make_monitor(adaptive=False)
        monitor._ingest(sample({"neck_flexion": 170.0}))  # noqa: SLF001
        assert workers[0].rates == []


class TestCalibrationGating:
    def test_calibration_pins_the_full_rate(self):
        """Adaptive sampling must not throttle the capture it depends on.

        An uncalibrated detector reports GOOD or AWAY, so the adaptive rule
        would drop to the idle rate -- and a ten-second calibration measured
        only 32 frames instead of ~50 before this was fixed, which was not
        enough to build a baseline at all.
        """
        monitor, workers = make_monitor(fps=5.0, idle_fps=2.0)
        result = monitor.start_calibration()
        assert result["ok"] and result["cameras"] == [1]
        assert workers[0].rates == [5.0]

        # Samples arriving mid-calibration must not move the rate.
        for _ in range(5):
            monitor._ingest(sample({"neck_flexion": 170.0}))  # noqa: SLF001
        assert workers[0].rates == [5.0]

    def test_rate_control_resumes_after_calibration(self):
        monitor, workers = make_monitor(fps=5.0, idle_fps=2.0)
        monitor.start_calibration()
        workers[0].calibration.finish()
        monitor.collect_calibration()
        monitor._ingest(sample({"neck_flexion": 170.0}))  # noqa: SLF001
        assert workers[0].rates[-1] in (2.0, 5.0)
        assert len(workers[0].rates) > 1

    def test_cancelling_releases_the_pin(self):
        monitor, workers = make_monitor(fps=5.0, idle_fps=2.0)
        monitor.start_calibration()
        monitor.cancel_calibration()
        monitor._ingest(sample({"neck_flexion": 170.0}))  # noqa: SLF001
        assert len(workers[0].rates) > 1

    def test_calibration_with_no_cameras_fails_cleanly(self):
        monitor, _ = make_monitor()
        monitor._workers = []  # noqa: SLF001
        result = monitor.start_calibration()
        assert result["ok"] is False and "no running camera" in result["error"]


class TestBaselinePersistence:
    def test_finished_calibration_is_stored_and_applied(self):
        saved: list[Config] = []
        monitor, workers = make_monitor()
        monitor.set_config_change_hook(saved.append)
        monitor.start_calibration()
        workers[0].calibration.finish()
        result = monitor.collect_calibration()
        assert result["ok"]
        assert "1" in monitor.cfg.baselines
        assert monitor._detector.calibrated  # noqa: SLF001
        assert saved, "a finished calibration must be persisted"

    def test_clearing_baselines_uncalibrates(self):
        monitor, workers = make_monitor()
        monitor.start_calibration()
        workers[0].calibration.finish()
        monitor.collect_calibration()
        monitor.clear_baselines()
        assert monitor.cfg.baselines == {}
        assert not monitor._detector.calibrated  # noqa: SLF001

    def test_noisy_calibration_is_reported(self):
        monitor, workers = make_monitor()
        monitor.start_calibration()
        workers[0].calibration.finish(spread=40.0)  # way past the 25 deg ceiling
        result = monitor.collect_calibration()
        assert any("Recalibrate" in p for p in result["problems"])

    def test_unreadable_baseline_is_skipped_not_fatal(self):
        cfg = Config()
        cfg.baselines = {"1": {"nonsense": True}, "oops": {"camera": 2}}
        assert mon._baselines(cfg) == {} or set(mon._baselines(cfg)) <= {1, 2}


class TestSnapshot:
    def test_snapshot_carries_the_posture_verdict(self):
        monitor, _ = make_monitor()
        snap = monitor.snapshot()
        assert snap["posture"]["state"] in (det.AWAY, det.UNKNOWN, det.GOOD)
        assert "calibrating" in snap and "baselines" in snap

    def test_settings_translate_from_config(self):
        cfg = Config()
        cfg.detection.window_seconds = 42.0
        cfg.detection.overrides = {"neck_flexion": 3.0}
        settings = mon._settings(cfg)
        assert settings.window_seconds == 42.0
        assert settings.overrides == {"neck_flexion": 3.0}


class TestPreview:
    """The window between a camera starting and its first frame is real."""

    def test_no_frame_yet_is_empty_not_a_partial_tuple(self):
        """Appending the role to an empty preview would make a truthy 1-tuple,
        which the caller unpacks into three names and crashes on -- a 500 for
        every request during the second or so each camera takes to start."""
        monitor, workers = make_monitor()
        assert monitor.latest_preview(1) == ()

    def test_unknown_camera_is_empty(self):
        monitor, _ = make_monitor()
        assert monitor.latest_preview(99) == ()

    def test_a_ready_camera_returns_image_landmarks_and_role(self):
        import numpy as np
        monitor, workers = make_monitor()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        workers[0].latest_preview = lambda: (image, None)
        preview = monitor.latest_preview(1)
        assert len(preview) == 3
        assert preview[0] is image and preview[2] == "side"


class FakeNotifier:
    def __init__(self, *a, **kw):
        self.sent = []
        self.backend = "fake"

    def send(self, title, message):
        self.sent.append((title, message))
        return True


class FakeOverlay:
    def __init__(self, on_snooze=None, **kw):
        self.on_snooze = on_snooze
        self.visible = False
        self.available = True
        self.shown = 0
        self.updates = 0
        self.hidden = 0
        self.stopped = False

    def show(self, *a):
        self.visible = True
        self.shown += 1

    def update(self, *a):
        self.updates += 1

    def hide(self):
        self.visible = False
        self.hidden += 1

    def stop(self):
        self.stopped = True
        self.visible = False


@pytest.fixture
def alerting(monkeypatch):
    """A monitor whose alert effects are fakes, with fast escalation."""
    from posture import alerts as al

    notifier = FakeNotifier()
    overlays = []

    def make_overlay(**kw):
        ov = FakeOverlay(**kw)
        overlays.append(ov)
        return ov

    monkeypatch.setattr(mon, "Notifier", lambda *a, **k: notifier)
    monkeypatch.setattr(mon, "OverlayWindow", make_overlay)

    monitor, workers = make_monitor()
    monitor.cfg.alerts.notify_after = 2.0
    monitor.cfg.alerts.overlay_after = 2.0
    monitor.cfg.alerts.clear_hold = 1.0
    monitor._alerts.set_settings(mon._alert_settings(monitor.cfg))  # noqa: SLF001
    return monitor, workers, notifier, overlays, al


def drive(monitor, state, seconds, t0=0.0, step=0.5):
    """Push a posture state through the monitor's alert engine."""
    t = t0
    while t < t0 + seconds:
        verdict = type("V", (), {"state": state, "offenders": ("neck_tilt",)
                                 if state == "bad" else ()})()
        monitor._alert_state = monitor._alerts.update(verdict, now=t)  # noqa: SLF001
        monitor._apply_alert_effects(monitor._alerts.drain())          # noqa: SLF001
        t += step
    return t


class TestAlertEffects:
    def test_notification_fires_once_at_the_notify_step(self, alerting):
        monitor, _w, notifier, _o, al = alerting
        drive(monitor, "bad", 3.0)
        assert len(notifier.sent) == 1
        drive(monitor, "bad", 3.0, t0=3.0)
        assert len(notifier.sent) == 1, "must not repeat while sitting at a level"

    def test_the_overlay_appears_at_the_overlay_step(self, alerting):
        monitor, _w, _n, overlays, al = alerting
        drive(monitor, "bad", 6.0)
        assert overlays and overlays[0].visible
        assert monitor.alert.level == al.LEVEL_OVERLAY

    def test_the_overlay_is_updated_while_it_is_up(self, alerting):
        monitor, _w, _n, overlays, _al = alerting
        drive(monitor, "bad", 8.0)
        assert overlays[0].updates > 1, "the countdown has to keep moving"

    def test_holding_a_good_posture_takes_it_down(self, alerting):
        monitor, _w, _n, overlays, al = alerting
        t = drive(monitor, "bad", 6.0)
        assert overlays[0].visible
        drive(monitor, "good", 2.0, t0=t)
        assert not overlays[0].visible
        assert monitor.alert.level == al.LEVEL_NONE

    def test_snooze_takes_the_overlay_down_immediately(self, alerting):
        monitor, _w, _n, overlays, _al = alerting
        drive(monitor, "bad", 6.0)
        assert overlays[0].visible
        monitor.snooze()
        assert not overlays[0].visible
        assert monitor.alert.snoozed

    def test_the_overlay_can_snooze_itself(self, alerting):
        """The button on the overlay is the only escape it offers."""
        monitor, _w, _n, overlays, _al = alerting
        drive(monitor, "bad", 6.0)
        overlays[0].on_snooze()
        assert monitor.alert.snoozed

    def test_notifications_can_be_switched_off(self, alerting):
        monitor, _w, notifier, _o, _al = alerting
        monitor.cfg.alerts.os_notifications = False
        drive(monitor, "bad", 6.0)
        assert notifier.sent == []

    def test_the_overlay_can_be_switched_off(self, alerting):
        monitor, _w, _n, overlays, al = alerting
        monitor.cfg.alerts.fullscreen_overlay = False
        drive(monitor, "bad", 6.0)
        assert not any(o.visible for o in overlays)
        assert monitor.alert.level == al.LEVEL_OVERLAY, "the level still escalates"

    def test_stopping_the_monitor_takes_the_overlay_down(self, alerting):
        monitor, _w, _n, overlays, _al = alerting
        drive(monitor, "bad", 6.0)
        monitor.stop()
        assert overlays[0].stopped

    def test_alert_state_reaches_the_snapshot(self, alerting):
        monitor, _w, _n, _o, _al = alerting
        drive(monitor, "bad", 6.0)
        payload = monitor.snapshot()["alert"]
        assert payload["level"] == 3 and payload["active"] is True
        assert "neck_tilt" in payload["offenders"]
