"""Two ways the posture score went absurdly low on a person sitting normally.

Both were found from a live session rather than imagined, and both produced the
same symptom -- a headline score near zero with nothing visibly wrong -- by
completely different routes. They are tested together because the symptom is
what a user reports, and whoever comes back to this next will be looking for
the symptom, not the cause.
"""

from __future__ import annotations

import numpy as np

from posture import landmarks as lmk
from posture import metrics as met
from posture import score as scoring
from posture.calibration import build_baseline


def turned(deg: float, shoulder_dy: float = 0.030, ear_dy: float = 0.010):
    """A seated person facing the camera, rotated ``deg`` about the vertical.

    Rotation foreshortens the horizontal spread of the shoulder and ear pairs
    by cos(deg) while leaving the vertical offsets -- the small asymmetries any
    real body has -- untouched. That combination is the whole bug.
    """
    c = float(np.cos(np.radians(deg)))
    half_shoulder, half_ear = 0.10 * c, 0.045 * c
    return lmk.make_synthetic({
        lmk.LEFT_SHOULDER: (0.50 - half_shoulder, 0.44),
        lmk.RIGHT_SHOULDER: (0.50 + half_shoulder, 0.44 + shoulder_dy),
        lmk.LEFT_EAR: (0.50 - half_ear, 0.24),
        lmk.RIGHT_EAR: (0.50 + half_ear, 0.24 + ear_dy),
    }, visibility=0.95)


def front(arr):
    return met.compute("front", arr, aspect=4 / 3, vis_thresh=0.5, t=0.0)


class TestTurningAwayFromTheFrontCamera:
    """Live session: shoulder tilt -52.3 and head roll -59.4 against a
    4-degree tolerance, from someone who had turned to talk to somebody. The
    score read 19 while the detector's own state was "good"."""

    def test_a_hard_turn_reports_nothing_rather_than_nonsense(self):
        sample = front(turned(85))
        assert sample.values == {}
        assert any("facing" in n for n in sample.notes)

    def test_the_old_scale_guard_would_not_have_caught_it(self):
        """MIN_SCALE guards the straight-line distance between the shoulders,
        which stays healthy precisely because of the vertical offset causing
        the trouble. Pinning this stops the guard being 'simplified' back."""
        sample = front(turned(85))
        assert sample.scale is not None and sample.scale > met.MIN_SCALE

    def test_the_reported_tilt_is_bounded_by_the_gate(self):
        """Whatever the turn, a reading that survives is within a range the
        metric's own tolerances can make sense of."""
        for deg in range(0, 90, 5):
            value = front(turned(deg)).values.get("shoulder_tilt")
            if value is not None:
                assert abs(value) < 30.0, f"{deg} degrees produced {value}"

    def test_facing_the_camera_still_measures_normally(self):
        values = front(turned(0)).values
        assert set(values) == {"shoulder_tilt", "head_roll", "lateral_offset"}

    def test_a_genuine_tilt_while_facing_is_still_reported(self):
        """The gate must not buy its safety by suppressing the thing the
        metric exists to catch. max_tolerance for shoulder tilt is 12."""
        arr = lmk.make_synthetic({
            lmk.LEFT_SHOULDER: (0.40, 0.44), lmk.RIGHT_SHOULDER: (0.60, 0.485),
            lmk.LEFT_EAR: (0.455, 0.24), lmk.RIGHT_EAR: (0.545, 0.25),
        }, visibility=0.95)
        tilt = front(arr).values.get("shoulder_tilt")
        assert tilt is not None
        assert abs(tilt) > met.SPEC_BY_KEY["shoulder_tilt"].max_tolerance / 2

    def test_lateral_offset_goes_too_and_not_just_the_angles(self):
        """It divides by the same shoulder span the turn collapses, so leaving
        it in would be the same error one step further on."""
        assert "lateral_offset" not in front(turned(85)).values


class TestABaselineNeutralPostureCannotReach:
    """Live session: a torso lean baseline of -15.9 degrees, captured while
    reclining. Sitting upright then read 15.9 past baseline against a 7-degree
    tolerance, so the score sat near zero all day and blamed torso lean while
    the person sat perfectly straight."""

    def samples(self, torso_lean: float, n: int = 60):
        return [met.MetricSample(
            role="side", t=i * 0.2, person=True, camera_index=0,
            values={"neck_tilt": 6.5, "neck_flexion": 157.3,
                    "forward_head": 0.05, "torso_lean": torso_lean + (i % 5) * 0.4},
        ) for i in range(n)]

    def test_calibrating_while_reclined_is_reported_as_a_problem(self):
        _baseline, problems = build_baseline(0, "side", self.samples(-15.9))
        assert any("Torso lean" in p and "Recalibrate" in p for p in problems)

    def test_calibrating_upright_is_not(self):
        _baseline, problems = build_baseline(0, "side", self.samples(-2.0))
        assert problems == []

    def test_the_warning_is_about_reach_not_about_spread(self):
        """A tight spread is what made this look like a high-quality baseline.
        The problem has to be detected despite it, not because of it."""
        _baseline, problems = build_baseline(
            0, "side", [met.MetricSample(
                role="side", t=i * 0.2, person=True, camera_index=0,
                values={"torso_lean": -15.9}) for i in range(60)])
        assert any("Torso lean" in p for p in problems)

    def test_a_one_sided_metric_measured_from_180_is_handled(self):
        """neck_flexion is LOW_IS_BAD around a neutral of 180, so the same
        check has to work without assuming neutral is zero."""
        spec = met.SPEC_BY_KEY["neck_flexion"]
        assert spec.neutral == 180.0
        # A baseline *above* neutral is unreachable-in-the-bad-direction for a
        # LOW_IS_BAD metric, so a stacked-upright neck must not be a problem.
        _baseline, problems = build_baseline(0, "side", [met.MetricSample(
            role="side", t=i * 0.2, person=True, camera_index=0,
            values={"neck_flexion": 178.0}) for i in range(60)])
        assert not any("Neck flexion" in p for p in problems)


class TestTheSymptomItself:
    def test_a_wild_reading_no_longer_reaches_the_score(self):
        """End to end, in the units the user sees: the score is what was
        absurd, so the test says so in those terms."""
        sample = front(turned(85))
        assert not sample.values, "nothing to score means nothing to score wrongly"

        class FakeMetric:
            key, label = "shoulder_tilt", "Shoulder tilt"
            ratio, out_fraction, flagged = 13.132, 0.518, False

        class FakeVerdict:
            state, calibrated = "good", True
            metrics = (FakeMetric(),)

        # Belt and braces. Fed the live numbers directly, the score no longer
        # calls it "needs correction" either: the detector's state was "good"
        # and it had not flagged the metric, so the score is capped out of the
        # bad band. Both defences would have to fail to get 19 back.
        s = scoring.score_verdict(FakeVerdict(), 0.70)
        assert s.band != "needs correction", s.value
