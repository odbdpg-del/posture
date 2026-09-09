"""Landmark jitter must not reach anything that judges you.

Recorded from a live side camera at two samples a second, on a person the
detector called "good" throughout::

    +13.4  -0.1  +0.6  +19.4  +4.8  +17.8  +6.3  +28.1  +16.4  +11.3

That is neck tilt in degrees. A neck does not do that. Over 75 seconds the raw
signal moved 3.9 degrees between consecutive samples on average, and the
posture score -- which reads the newest sample -- wandered between 57 and 86
while the posture itself did not change band once.
"""

from __future__ import annotations

import statistics

from posture import metrics as met
from posture.smoothing import SampleSmoother

# The real recording, so the test is anchored to a measurement rather than to
# a guess about what jitter looks like.
LIVE_NECK_TILT = [13.4, -0.1, 0.6, 19.4, 4.8, 17.8, 6.3, 28.1, 16.4, 11.3,
                  4.9, -5.9, 14.8, 1.4, 14.3, 2.7, -6.6, 15.1]


def sample(t, role="side", **values):
    return met.MetricSample(role=role, t=t, person=True, camera_index=0,
                            values=dict(values))


def feed(smoother, values, step=0.5, key="neck_tilt"):
    return [smoother.add(sample(i * step, **{key: v})).values[key]
            for i, v in enumerate(values)]


def travel(xs):
    """Mean absolute change per sample: how much of the signal is movement the
    body could not actually have performed."""
    return statistics.mean(abs(b - a) for a, b in zip(xs, xs[1:]))


class TestItRemovesWhatTheBodyCannotDo:
    def test_the_live_trace_stops_flailing(self):
        out = feed(SampleSmoother(1.5), LIVE_NECK_TILT)
        assert travel(out) < travel(LIVE_NECK_TILT) / 2

    def test_the_real_movement_survives(self):
        """Smoothing that flattened the signal would be worse than the noise:
        the lean this trace contains has to still be there."""
        out = feed(SampleSmoother(1.5), LIVE_NECK_TILT)
        assert max(out) > 12.0, "the forward lean is real and must not be lost"
        assert min(out) < 6.0, "and so is coming back up from it"

    def test_a_single_wild_sample_does_not_move_it_much(self):
        """The failure mode a mean would have. One spike among steady readings
        is the shape jitter actually arrives in."""
        steady = [5.0] * 6
        out = feed(SampleSmoother(1.5), steady + [45.0] + steady)
        assert max(out) < 10.0

    def test_a_real_change_is_followed(self):
        """Sustained movement is posture, not noise, and must come through."""
        out = feed(SampleSmoother(1.5), [2.0] * 8 + [20.0] * 8)
        assert out[-1] == 20.0


class TestItDoesNotInventData:
    def test_a_metric_absent_from_a_frame_stays_absent(self):
        """"I can see you but not your hips" must keep meaning that. A
        remembered value is not a measurement."""
        s = SampleSmoother()
        s.add(sample(0.0, neck_tilt=5.0, torso_lean=3.0))
        out = s.add(sample(0.2, neck_tilt=5.0))
        assert "torso_lean" not in out.values

    def test_everything_but_the_values_is_passed_through(self):
        s = SampleSmoother()
        original = met.MetricSample(
            role="side", t=0.0, person=True, camera_index=2,
            values={"neck_tilt": 5.0}, missing=("left_hip",),
            notes=("hips not in frame; torso metrics unavailable",),
            near_side="right", facing=-1, scale_kind="neck")
        out = s.add(original)
        for field in ("role", "t", "person", "camera_index", "missing", "notes",
                      "near_side", "facing", "scale_kind"):
            assert getattr(out, field) == getattr(original, field), field

    def test_an_empty_sample_is_returned_untouched(self):
        s = SampleSmoother()
        empty = met.MetricSample(role="side", t=0.0, person=False)
        assert s.add(empty) is empty

    def test_a_stale_history_is_not_reused_after_a_gap(self):
        """Away for a minute, then back. The old readings describe a posture
        from before you left and must not be blended into the new one."""
        s = SampleSmoother(1.5)
        for i in range(10):
            s.add(sample(i * 0.2, neck_tilt=30.0))
        out = s.add(sample(120.0, neck_tilt=4.0))
        assert out.values["neck_tilt"] == 4.0

    def test_the_window_is_time_not_sample_count(self):
        """Adaptive sampling changes the rate underneath this. At the idle
        rate a fixed count would silently become a twelve-second window."""
        fast = feed(SampleSmoother(1.5), [0.0] * 5 + [10.0] * 5, step=0.1)
        slow = feed(SampleSmoother(1.5), [0.0] * 5 + [10.0] * 5, step=2.0)
        assert slow[-1] == 10.0, "at 0.5 Hz each sample stands alone"
        assert fast[-1] < 10.0, "at 10 Hz the window still holds the older ones"

    def test_reset_forgets(self):
        s = SampleSmoother(1.5)
        for i in range(5):
            s.add(sample(i * 0.2, neck_tilt=30.0))
        s.reset()
        assert s.add(sample(1.2, neck_tilt=4.0)).values["neck_tilt"] == 4.0


class TestItIsAppliedBeforeAnythingJudges:
    def test_the_worker_smooths_before_the_detector_and_calibration(self):
        """Both have to see the same signal. A tolerance derived from the
        spread of a raw stream and applied to a smooth one would be too loose,
        and the reverse too tight."""
        import inspect

        from posture import monitor as mon

        src = inspect.getsource(mon.CameraWorker._run)
        assert "self._smoother.add(" in src
        smoothed = src.index("self._smoother.add(")
        assert smoothed < src.index("self._on_sample(sample)")
        assert smoothed < src.index("session.add(sample)")
