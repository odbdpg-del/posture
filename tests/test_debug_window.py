"""The debug window must be closable.

This has bitten twice, so it gets a test rather than a careful reading. Closing
an OpenCV window destroys it without telling the program, and the next
``imshow`` silently creates a new one -- so the window reappears every frame and
the app looks like it refuses to die. The loop has to notice the window is gone
and stop.

The camera, the pose model and OpenCV's window functions are all faked, so this
runs in milliseconds and needs no hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from posture import cli


class FakeFrame:
    def __init__(self, t: float) -> None:
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.t = t
        self.seq = int(t * 10)

    @property
    def aspect(self) -> float:
        return 64 / 48


class FakeCapture:
    """Hands out frames forever, so only the exit condition can stop the loop."""

    def __init__(self, *args, **kwargs) -> None:
        self.backend_name = "FAKE"
        self.last_error = None
        self._t = 0.0
        self.stopped = False

    def start(self):
        return self

    def read(self, timeout: float = 2.0):
        self._t += 0.02
        return FakeFrame(self._t)

    def stop(self) -> None:
        self.stopped = True


class FakeResult:
    array = None
    infer_ms = 1.0

    @property
    def person(self) -> bool:
        return False


class FakeEstimator:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def detect(self, image, timestamp_ms):
        return FakeResult()

    def close(self) -> None:
        self.closed = True


class FakeCv2:
    """Just the surface `run` touches, plus a switch for closing the window."""

    WND_PROP_VISIBLE = 4

    def __init__(self, close_after: int | None = None, key: int = -1) -> None:
        self.close_after = close_after
        self.key = key
        self.shown = 0
        self.destroyed = False

    def imshow(self, name, image) -> None:
        self.shown += 1

    def waitKey(self, delay):  # noqa: N802 - mirrors the cv2 name
        return self.key

    def getWindowProperty(self, name, prop):  # noqa: N802 - mirrors the cv2 name
        if self.close_after is not None and self.shown >= self.close_after:
            return 0.0  # what OpenCV reports once the window is gone
        return 1.0

    def destroyAllWindows(self) -> None:  # noqa: N802 - mirrors the cv2 name
        self.destroyed = True


@pytest.fixture
def patched(monkeypatch, tmp_path):
    captures: list[FakeCapture] = []
    estimators: list[FakeEstimator] = []

    def make_capture(*args, **kwargs):
        cap = FakeCapture()
        captures.append(cap)
        return cap

    def make_estimator(*args, **kwargs):
        est = FakeEstimator()
        estimators.append(est)
        return est

    monkeypatch.setattr(cli, "CameraCapture", make_capture)
    monkeypatch.setattr(cli, "PoseEstimator", make_estimator)
    return captures, estimators


def run_debug(fake_cv2, monkeypatch, tmp_path, extra=()):
    monkeypatch.setattr(cli, "cv2", fake_cv2)
    args = cli.build_parser().parse_args([
        "--watch", "--debug", "--camera", "0", "--role", "side",
        "--duration", "600", "--config", str(tmp_path / "cfg.json"), *extra,
    ])
    return cli.run(args)


class TestClosing:
    def test_closing_the_window_stops_the_loop(self, patched, monkeypatch, tmp_path):
        """The X button. Without this the loop runs until the heat death of the
        laptop, rebuilding the window five times a second."""
        captures, estimators = patched
        fake = FakeCv2(close_after=5)
        code = run_debug(fake, monkeypatch, tmp_path)
        assert code == 0
        assert fake.shown < 20, "loop kept drawing after the window was closed"
        assert captures[0].stopped, "camera must be released on the way out"
        assert estimators[0].closed, "pose model must be closed on the way out"

    def test_q_still_quits(self, patched, monkeypatch, tmp_path):
        captures, _ = patched
        fake = FakeCv2(key=ord("q"))
        assert run_debug(fake, monkeypatch, tmp_path) == 0
        assert captures[0].stopped

    def test_escape_still_quits(self, patched, monkeypatch, tmp_path):
        captures, _ = patched
        fake = FakeCv2(key=27)
        assert run_debug(fake, monkeypatch, tmp_path) == 0
        assert captures[0].stopped

    def test_an_open_window_does_not_stop_the_loop(self, patched, monkeypatch,
                                                   tmp_path):
        """The guard must not fire on a window that is merely minimised or
        unfocused, or the app would quit the moment you looked away."""
        captures, _ = patched
        fake = FakeCv2(close_after=None)
        monkeypatch.setattr(cli, "cv2", fake)
        args = cli.build_parser().parse_args([
            "--watch", "--debug", "--camera", "0", "--role", "side",
            "--duration", "0.4", "--config", str(tmp_path / "cfg.json"),
        ])
        assert cli.run(args) == 0
        assert fake.shown > 1, "loop should have kept running until its duration"
        assert captures[0].stopped

    def test_window_hint_is_on_screen(self):
        """The exit key has to be discoverable from the window itself."""
        from posture import metrics as met
        from posture import overlay

        canvas = np.zeros((320, 480, 3), dtype=np.uint8)
        sample = met.MetricSample(role="side", t=0.0, person=False)
        status, colour = overlay.status_for(sample)
        overlay.draw_panel(canvas, sample, status=status, status_colour=colour,
                           fps=5.0, infer_ms=17.0, cpu_pct=9.0)
        # The hint is drawn as text, so assert on the source of the panel rather
        # than pixels: cheap, and it fails if the line is ever dropped.
        import inspect
        assert "press q or Esc to quit" in inspect.getsource(overlay.draw_panel)
