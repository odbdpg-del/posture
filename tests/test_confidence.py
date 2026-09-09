"""Per-metric confidence, and what the app is allowed to do with a shaky one.

Confidence is per metric rather than per frame because it is not one question
for the whole frame. On a desk camera the shoulders are solid and the hips are
extrapolated, so in a single frame neck tilt can be worth acting on while trunk
lean is guesswork -- and a frame-level number would average that away, which is
the distinction that matters most.
"""

from __future__ import annotations

from posture import detector as det
from posture import landmarks as lmk
from posture import metrics as met
from posture.calibration import Baseline, MetricBaseline


def seated(vis=0.97, **overrides):
    """A side view with every landmark the metrics need."""
    arr = lmk.make_synthetic({
        lmk.NOSE: (0.60, 0.24),
        lmk.RIGHT_EAR: (0.50, 0.25), lmk.RIGHT_SHOULDER: (0.48, 0.45),
        lmk.RIGHT_HIP: (0.46, 0.72),
        lmk.LEFT_EAR: (0.49, 0.25), lmk.LEFT_SHOULDER: (0.47, 0.45),
        lmk.LEFT_HIP: (0.45, 0.72),
    }, visibility=vis)
    for name, v in overrides.items():
        arr[getattr(lmk, name.upper()), 3] = v
    return arr


def side(arr, thresh=0.6):
    return met.compute("side", arr, aspect=4 / 3, vis_thresh=thresh, t=0.0)


class TestItIsPerMetric:
    def test_a_shaky_hip_lowers_only_the_metrics_that_need_one(self):
        """The whole reason it is not one number for the frame."""
        s = side(seated(right_hip=0.64, left_hip=0.64))
        assert s.confidence["neck_tilt"] > 0.9
        for key in ("torso_lean", "neck_flexion", "forward_head"):
            assert s.confidence[key] < 0.7, key

    def test_it_is_the_weakest_landmark_not_the_average(self):
        """Averaging a confident shoulder against an invented hip reports a
        comfortable number for a geometry that is guesswork at one end."""
        s = side(seated(right_hip=0.62, left_hip=0.62))
        assert s.confidence["torso_lean"] == 0.62

    def test_every_reported_metric_has_one(self):
        s = side(seated())
        assert set(s.confidence) == set(s.values)

    def test_a_metric_that_was_not_computed_has_none(self):
        s = side(seated(right_hip=0.05, left_hip=0.05))
        assert "torso_lean" not in s.values
        assert "torso_lean" not in s.confidence

    def test_the_front_role_reports_its_own(self):
        arr = lmk.make_synthetic({
            lmk.LEFT_SHOULDER: (0.40, 0.44), lmk.RIGHT_SHOULDER: (0.60, 0.45),
            lmk.LEFT_EAR: (0.455, 0.24), lmk.RIGHT_EAR: (0.545, 0.25),
        }, visibility=0.9)
        arr[lmk.LEFT_EAR, 3] = 0.7
        s = met.compute("front", arr, 4 / 3, 0.6, 0.0)
        assert s.confidence["shoulder_tilt"] == 0.9
        assert s.confidence["head_roll"] == 0.7


# Baselines chosen to be both breached by the synthetic pose and reachable
# from a neutral one -- an unreachable baseline is excluded from judgement by a
# different guard, which would make these tests pass for the wrong reason.
def detector_with(min_confidence=0.75, **centres):
    d = det.PostureDetector(det.DetectionSettings(
        min_window_fill=0.0, min_confidence=min_confidence))
    d.set_baselines({0: Baseline(camera=0, role="side", metrics={
        k: MetricBaseline(k, c, 1.0, 50) for k, c in centres.items()})})
    return d


def drive(d, seconds, arr, step=0.5):
    verdict = None
    t = 0.0
    while t < seconds:
        sample = met.compute("side", arr, 4 / 3, 0.6, t, camera_index=0)
        verdict = d.update([sample], now=t)
        t += step
    return verdict


class TestWhatTheAppDoesWithALowOne:
    """Measured badly is not the same as measured. The reading is reported
    either way; only the alert is withheld."""

    def test_a_confident_deviation_is_flagged(self):
        d = detector_with(torso_lean=-5.0)
        v = drive(d, 90.0, seated())
        assert "torso_lean" in v.offenders

    def test_the_same_deviation_unseen_is_not(self):
        d = detector_with(torso_lean=-5.0)
        v = drive(d, 90.0, seated(right_hip=0.62, left_hip=0.62))
        assert "torso_lean" not in v.offenders
        assert v.state == det.GOOD

    def test_but_it_is_still_reported(self):
        """Hiding it would be worse than nagging: the panel would silently
        shrink and nothing would say the camera had stopped seeing you well."""
        d = detector_with(torso_lean=-5.0)
        v = drive(d, 90.0, seated(right_hip=0.62, left_hip=0.62))
        row = next(m for m in v.metrics if m.key == "torso_lean")
        assert row.value is not None
        assert row.low_confidence is True
        assert row.confidence == 0.62
        assert row.to_dict()["low_confidence"] is True

    def test_a_confident_metric_beside_a_shaky_one_still_works(self):
        """The gate must not silence the camera, only the reading it cannot
        stand behind."""
        d = detector_with(torso_lean=-5.0, neck_tilt=-5.0)
        v = drive(d, 90.0, seated(right_hip=0.62, left_hip=0.62))
        assert v.offenders == ("neck_tilt",)

    def test_the_threshold_is_configurable(self):
        d = detector_with(min_confidence=0.5, torso_lean=-5.0)
        v = drive(d, 90.0, seated(right_hip=0.62, left_hip=0.62))
        assert "torso_lean" in v.offenders, "0.62 clears a 0.5 bar"

    def test_it_is_stricter_than_the_visibility_threshold(self):
        """Landmarks already have to clear visibility before a metric exists at
        all. This is the second gate, and it has to actually be second."""
        from posture.config import Config

        assert (det.DetectionSettings().min_confidence
                > Config().sampling.visibility_threshold)
