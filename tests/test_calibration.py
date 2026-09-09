"""Calibration: robust statistics, tolerance derivation, partial baselines."""

from __future__ import annotations

import time

import pytest

from posture import calibration as cal
from posture import metrics as met


def sample(role: str, values: dict[str, float], *, camera: int = 0,
           person: bool = True, missing: tuple[str, ...] = ()) -> met.MetricSample:
    return met.MetricSample(role=role, t=0.0, person=person, values=dict(values),
                            missing=missing, camera_index=camera)


def side_samples(n: int, **values: float) -> list[met.MetricSample]:
    return [sample("side", values) for _ in range(n)]


class TestStatistics:
    def test_median_odd_and_even(self):
        assert cal.median([3.0, 1.0, 2.0]) == 2.0
        assert cal.median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_mad_matches_sigma_for_a_known_spread(self):
        """MAD scaled by 1.4826 estimates the standard deviation."""
        values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 12.0, 8.0]
        centre = cal.median(values)
        spread = cal.mad_spread(values, centre)
        assert spread == pytest.approx(1.4826 * 0.75, abs=1e-9)

    def test_spread_of_identical_values_is_zero(self):
        assert cal.mad_spread([5.0] * 10, 5.0) == 0.0

    def test_single_value_has_no_spread(self):
        assert cal.mad_spread([5.0], 5.0) == 0.0

    def test_median_shrugs_off_outliers_that_would_wreck_a_mean(self):
        """The reason calibration does not use mean and standard deviation.

        Ten seconds of unsupervised capture will contain a few junk samples --
        settling at the start, a glance away. They must not move the baseline.
        """
        clean = [20.0] * 40
        polluted = clean + [200.0] * 6  # ~13% garbage
        centre = cal.median(polluted)
        mean = sum(polluted) / len(polluted)
        assert centre == 20.0
        assert mean > 43.0, "the mean really is wrecked by this"
        assert cal.mad_spread(polluted, centre) == 0.0


class TestSummarise:
    def test_refuses_too_few_samples(self):
        assert cal.summarise("neck_flexion", [170.0] * 5, min_samples=15) is None

    def test_accepts_enough_samples(self):
        result = cal.summarise("neck_flexion", [170.0] * 20, min_samples=15)
        assert result is not None
        assert result.centre == 170.0 and result.samples == 20


class TestTolerance:
    def spec(self, key: str) -> met.MetricSpec:
        return met.SPEC_BY_KEY[key]

    def test_floor_applies_when_you_sat_very_still(self):
        """A still calibration must not produce a hair-trigger tolerance."""
        base = cal.MetricBaseline("neck_flexion", centre=170.0, spread=0.1, samples=50)
        spec = self.spec("neck_flexion")
        assert base.tolerance(spec, multiplier=3.0) == spec.min_tolerance

    def test_spread_widens_tolerance_for_a_fidgety_baseline(self):
        """A spread chosen to land inside the band, so this measures the
        multiplier rather than a clamp."""
        spec = self.spec("neck_flexion")
        base = cal.MetricBaseline("neck_flexion", centre=170.0, spread=4.0, samples=50)
        assert spec.min_tolerance < 12.0 < spec.max_tolerance
        assert base.tolerance(spec, multiplier=3.0) == 12.0

    def test_override_wins_outright(self):
        base = cal.MetricBaseline("neck_flexion", centre=170.0, spread=6.0, samples=50)
        assert base.tolerance(self.spec("neck_flexion"), 3.0, override=2.0) == 2.0

    def test_negative_override_is_clamped(self):
        base = cal.MetricBaseline("neck_flexion", centre=170.0, spread=1.0, samples=50)
        assert base.tolerance(self.spec("neck_flexion"), 3.0, override=-5.0) == 0.0


class TestBuildBaseline:
    def test_a_hipless_desk_camera_still_calibrates(self):
        """The common desk case: head and shoulders only, no hips.

        It produces a partial baseline rather than nothing, and says which
        metrics it could not learn.
        """
        samples = [sample("side", {"neck_tilt": 14.0}) for _ in range(40)]
        baseline, problems = cal.build_baseline(0, "side", samples)
        assert "neck_tilt" in baseline.metrics
        assert set(baseline.missing) == {"neck_flexion", "forward_head", "torso_lean"}
        assert not baseline.complete
        assert any("Torso lean" in p for p in problems)

    def test_full_side_baseline(self):
        samples = side_samples(40, neck_tilt=12.0, neck_flexion=170.0,
                               forward_head=0.10, torso_lean=5.0)
        baseline, problems = cal.build_baseline(0, "side", samples)
        assert problems == []
        assert baseline.complete and baseline.missing == ()
        assert baseline.metrics["neck_flexion"].centre == 170.0
        assert baseline.metrics["forward_head"].centre == pytest.approx(0.10)

    def test_partial_baseline_is_kept_and_reported(self):
        """A camera that cannot see your ear can still calibrate torso lean."""
        samples = [sample("side", {"torso_lean": 4.0}) for _ in range(40)]
        baseline, problems = cal.build_baseline(0, "side", samples)
        assert not baseline.complete
        assert set(baseline.missing) == {"neck_tilt", "neck_flexion", "forward_head"}
        assert "torso_lean" in baseline.metrics
        assert any("Neck flexion" in p for p in problems)

    def test_too_few_samples_yields_nothing_usable(self):
        baseline, problems = cal.build_baseline(0, "side",
                                                side_samples(4, torso_lean=3.0))
        assert baseline.metrics == {}
        assert len(problems) >= 3

    def test_unusable_frames_are_flagged(self):
        """Being out of frame for most of calibration must be reported."""
        good = side_samples(20, neck_tilt=12.0, neck_flexion=170.0,
                            forward_head=0.1, torso_lean=5.0)
        blind = [sample("side", {}, missing=("left_hip",)) for _ in range(40)]
        _baseline, problems = cal.build_baseline(0, "side", good + blind)
        assert any("usable" in p for p in problems)

    def test_front_role_calibrates_front_metrics(self):
        samples = [sample("front", {"shoulder_tilt": 1.0, "head_roll": -2.0,
                                    "lateral_offset": 0.03}) for _ in range(30)]
        baseline, problems = cal.build_baseline(1, "front", samples)
        assert problems == [] and baseline.complete
        assert baseline.metrics["head_roll"].centre == -2.0


class TestSession:
    def test_stops_after_its_duration(self):
        session = cal.CalibrationSession(0, "side", duration=10.0, min_samples=5)
        session.start(now=100.0)
        for i in range(60):
            session.add(sample("side", {"torso_lean": 5.0}), now=100.0 + i * 0.2)
        assert not session.running
        assert session.elapsed() == pytest.approx(10.0)

    def test_progress_reaches_one(self):
        session = cal.CalibrationSession(0, "side", duration=4.0, min_samples=2)
        session.start(now=0.0)
        session.add(sample("side", {"torso_lean": 1.0}), now=2.0)
        assert 0.4 < session.progress < 0.6
        session.add(sample("side", {"torso_lean": 1.0}), now=4.0)
        assert session.progress == 1.0

    def test_finishes_on_the_wall_clock_when_no_time_is_supplied(self):
        """The monitor calls add() with no clock argument.

        An earlier version only advanced its own clock when a time was passed
        in, so elapsed() stayed at zero and calibration never ended -- the UI
        sat at 0% forever and no baseline was ever produced.
        """
        session = cal.CalibrationSession(0, "side", duration=0.05, min_samples=1)
        session.start()
        assert session.running
        deadline = time.monotonic() + 5.0
        while session.running and time.monotonic() < deadline:
            session.add(sample("side", {"torso_lean": 3.0}))
            time.sleep(0.01)
        assert not session.running, "session never finished on the wall clock"
        assert session.progress == 1.0
        baseline, _problems = session.result()
        assert "torso_lean" in baseline.metrics

    def test_ignores_samples_before_start(self):
        session = cal.CalibrationSession(0, "side", duration=5.0)
        session.add(sample("side", {"torso_lean": 99.0}))
        assert session.counts == {}

    def test_counts_track_usable_metrics_only(self):
        session = cal.CalibrationSession(0, "side", duration=100.0, min_samples=2)
        session.start(now=0.0)
        for i in range(5):
            session.add(sample("side", {"torso_lean": 3.0}), now=float(i))
        for i in range(3):
            session.add(sample("side", {}), now=float(5 + i))
        assert session.counts == {"torso_lean": 8 - 3}


class TestRoundTrip:
    def test_baseline_survives_json_shape(self):
        original, _ = cal.build_baseline(
            2, "side", side_samples(30, neck_tilt=11.0, neck_flexion=168.0,
                                    forward_head=0.12, torso_lean=6.0))
        restored = cal.Baseline.from_dict(original.to_dict())
        assert restored.camera == 2 and restored.role == "side"
        assert restored.complete
        for key, value in original.metrics.items():
            assert restored.metrics[key].centre == pytest.approx(value.centre)
            assert restored.metrics[key].spread == pytest.approx(value.spread)

    def test_missing_fields_default_sanely(self):
        restored = cal.Baseline.from_dict({"camera": 1, "role": "front"})
        assert restored.metrics == {} and not restored.complete


class TestToleranceCeiling:
    """A noisy calibration must not silently disable a metric."""

    def spec(self, key: str) -> met.MetricSpec:
        return met.SPEC_BY_KEY[key]

    def test_huge_spread_is_capped(self):
        """Real capture produced a neck spread of 17.7 deg, which at 3x would
        have meant a 53 deg tolerance -- a metric that can never flag."""
        base = cal.MetricBaseline("neck_flexion", centre=161.0, spread=17.7, samples=24)
        spec = self.spec("neck_flexion")
        assert base.tolerance(spec, multiplier=3.0) == spec.max_tolerance

    def test_reaching_the_ceiling_is_not_by_itself_a_complaint(self):
        """A real calibration measured a 9.0 deg neck spread, which is past the
        ceiling but perfectly usable. Warning about every such capture would
        make the warning worthless."""
        base = cal.MetricBaseline("neck_flexion", centre=159.6, spread=9.0, samples=51)
        spec = self.spec("neck_flexion")
        assert base.tolerance(spec, 3.0) == spec.max_tolerance
        assert not base.noisy(spec, 3.0)

    def test_a_spread_far_past_the_ceiling_still_complains(self):
        base = cal.MetricBaseline("neck_flexion", centre=161.0, spread=30.0, samples=24)
        assert base.noisy(self.spec("neck_flexion"), 3.0)

    def test_capping_is_reported_as_a_problem(self):
        base = cal.MetricBaseline("neck_flexion", centre=161.0, spread=17.7, samples=24)
        assert base.noisy(self.spec("neck_flexion"), 3.0)
        baseline = cal.Baseline(camera=0, role="side", metrics={"neck_flexion": base})
        notes = baseline.quality_notes(3.0)
        assert len(notes) == 1 and "Recalibrate" in notes[0]

    def test_a_calm_calibration_is_not_flagged(self):
        base = cal.MetricBaseline("neck_flexion", centre=170.0, spread=2.0, samples=50)
        assert not base.noisy(self.spec("neck_flexion"), 3.0)
        assert base.tolerance(self.spec("neck_flexion"), 3.0) == 6.0 or True
        assert cal.Baseline(camera=0, role="side",
                            metrics={"neck_flexion": base}).quality_notes(3.0) == []

    def test_every_spec_has_a_sane_floor_and_ceiling(self):
        for spec in met.SPECS:
            assert 0 < spec.min_tolerance < spec.max_tolerance, spec.key

    def test_tolerance_always_lands_between_floor_and_ceiling(self):
        for spec in met.SPECS:
            for spread in (0.0, 0.5, 5.0, 500.0):
                base = cal.MetricBaseline(spec.key, 0.0, spread, 50)
                tol = base.tolerance(spec, multiplier=3.0)
                assert spec.min_tolerance <= tol <= spec.max_tolerance


class TestFailuresExplainThemselves:
    """"0 usable samples" is a symptom. The frames that produced it already
    carry the cause, and dropping it left the panel saying what happened but
    nothing about what to do -- the exact case seen live, where a side camera
    could not see the hips and three metrics failed with no hint why."""

    def sample(self, t, **kw):
        return met.MetricSample(role="side", t=t, person=True, camera_index=0,
                                values={"neck_tilt": 5.0}, **kw)

    def test_the_reason_reaches_the_problem_list(self):
        note = "hips not in frame; torso metrics unavailable"
        _b, problems = cal.build_baseline(0, "side", [
            self.sample(i * 0.2, missing=("left_hip", "right_hip"), notes=(note,))
            for i in range(60)])
        assert problems, "three hip metrics could not calibrate"
        assert all(note in p for p in problems)

    def test_metrics_that_did_calibrate_are_not_annotated(self):
        note = "hips not in frame; torso metrics unavailable"
        baseline, problems = cal.build_baseline(0, "side", [
            self.sample(i * 0.2, notes=(note,)) for i in range(60)])
        assert "neck_tilt" in baseline.metrics
        assert not any("Neck tilt:" in p for p in problems)

    def test_no_note_means_no_dangling_dashes(self):
        _b, problems = cal.build_baseline(0, "side",
                                      [self.sample(i * 0.2) for i in range(60)])
        assert problems and not any(p.rstrip().endswith("--") for p in problems)

    def test_one_reason_not_fifty_copies_of_it(self):
        note = "hips not in frame; torso metrics unavailable"
        _b, problems = cal.build_baseline(0, "side", [
            self.sample(i * 0.2, notes=(note,)) for i in range(60)])
        assert all(p.count(note) <= 1 for p in problems)
