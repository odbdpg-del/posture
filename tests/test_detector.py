"""Detection: rolling window, hysteresis, presence, and metric-level fusion.

Time is passed in explicitly rather than slept through, so a sixty-second
window is exercised in microseconds and the thresholds can be checked exactly.
"""

from __future__ import annotations

import pytest

from posture import calibration as cal
from posture import detector as det
from posture import metrics as met

HZ = 5.0
STEP = 1.0 / HZ


def sample(role: str, values: dict[str, float], *, camera: int = 0,
           person: bool = True, missing: tuple[str, ...] = ()) -> met.MetricSample:
    return met.MetricSample(role=role, t=0.0, person=person, values=dict(values),
                            missing=missing, camera_index=camera)


def baseline(camera: int = 0, role: str = "side", spread: float = 1.0,
             **centres: float) -> cal.Baseline:
    return cal.Baseline(
        camera=camera, role=role,
        metrics={k: cal.MetricBaseline(k, centre=v, spread=spread, samples=50)
                 for k, v in centres.items()},
    )


SIDE = baseline(0, "side", neck_flexion=170.0, forward_head=0.10, torso_lean=5.0)


def make(**overrides) -> det.PostureDetector:
    settings = det.DetectionSettings(**overrides)
    return det.PostureDetector(settings, {0: SIDE})


def feed(detector: det.PostureDetector, values: dict[str, float] | None,
         seconds: float, *, t0: float = 0.0, person: bool = True,
         camera: int = 0, missing: tuple[str, ...] = ()) -> det.PostureVerdict:
    """Feed a steady posture for a stretch of simulated time."""
    verdict = None
    t = t0
    end = t0 + seconds
    while t < end:
        samples = [sample("side", values or {}, camera=camera, person=person,
                          missing=missing)]
        verdict = detector.update(samples, now=t)
        t += STEP
    assert verdict is not None
    return verdict


class TestWindow:
    def test_trims_samples_older_than_the_window(self):
        window = det.MetricWindow(10.0)
        for i in range(30):
            window.add(float(i), 0.0)
        assert len(window) == 11  # t=19 back to t=9 inclusive
        assert window.span == pytest.approx(10.0)

    def test_out_fraction_counts_only_what_exceeds(self):
        window = det.MetricWindow(100.0)
        for i, excess in enumerate([0.0, 5.0, 0.0, 5.0]):
            window.add(float(i), excess)
        assert window.out_fraction(1.0) == 0.5
        assert window.out_fraction(9.0) == 0.0

    def test_empty_window_is_never_out(self):
        assert det.MetricWindow(10.0).out_fraction(0.0) == 0.0

    def test_clear_empties_it(self):
        window = det.MetricWindow(10.0)
        window.add(0.0, 1.0)
        window.clear()
        assert len(window) == 0


class TestPresence:
    def test_starts_away(self):
        detector = make()
        verdict = detector.update([], now=0.0)
        assert verdict.state == det.AWAY

    def test_empty_chair_suspends_judgement(self):
        """Terrible posture readings mean nothing if nobody is there."""
        detector = make()
        verdict = feed(detector, None, 40.0, person=False)
        assert verdict.state == det.AWAY
        assert "no person" in verdict.reason

    def test_brief_absence_does_not_trip_away(self):
        detector = make(absence_seconds=15.0)
        feed(detector, {"neck_flexion": 170.0}, 30.0)
        verdict = feed(detector, None, 10.0, t0=30.0, person=False)
        assert verdict.state != det.AWAY

    def test_absence_beyond_the_limit_does(self):
        detector = make(absence_seconds=15.0)
        feed(detector, {"neck_flexion": 170.0}, 30.0)
        verdict = feed(detector, None, 20.0, t0=30.0, person=False)
        assert verdict.state == det.AWAY

    def test_returning_after_a_long_absence_does_not_alert_instantly(self):
        """Coming back to your desk must not inherit a verdict from an hour ago."""
        detector = make()
        feed(detector, {"neck_flexion": 140.0}, 90.0)
        feed(detector, None, 60.0, t0=90.0, person=False)
        verdict = feed(detector, {"neck_flexion": 140.0}, 2.0, t0=3600.0)
        assert verdict.state != det.BAD


class TestVisibility:
    def test_person_without_usable_landmarks_is_unknown(self):
        detector = make()
        verdict = feed(detector, {}, 30.0, missing=("left_hip",))
        assert verdict.state == det.UNKNOWN
        assert "left_hip" in verdict.reason

    def test_uncalibrated_is_unknown_not_good(self):
        """Never claim posture is fine before there is anything to compare to."""
        detector = det.PostureDetector(det.DetectionSettings(), {})
        verdict = feed(detector, {"neck_flexion": 170.0}, 30.0)
        assert verdict.state == det.UNKNOWN
        assert "not calibrated" in verdict.reason
        assert verdict.calibrated is False


class TestVerdicts:
    def test_baseline_posture_is_good(self):
        detector = make()
        verdict = feed(detector, {"neck_flexion": 170.0, "forward_head": 0.10,
                                  "torso_lean": 5.0}, 90.0)
        assert verdict.state == det.GOOD
        assert verdict.offenders == ()

    def test_sustained_slouch_is_bad(self):
        detector = make()
        verdict = feed(detector, {"neck_flexion": 140.0}, 90.0)
        assert verdict.state == det.BAD
        assert "neck_flexion" in verdict.offenders
        assert verdict.worst_ratio > 1.0

    def test_occasional_slouch_is_not_bad(self):
        """Half a window out of tolerance is under the 70% bar."""
        detector = make(bad_fraction=0.70)
        t = 0.0
        verdict = None
        for _ in range(15):
            verdict = feed(detector, {"neck_flexion": 140.0}, 4.0, t0=t)
            t += 4.0
            verdict = feed(detector, {"neck_flexion": 170.0}, 4.0, t0=t)
            t += 4.0
        assert verdict.state == det.GOOD

    def test_does_not_judge_before_the_window_fills(self):
        """Three seconds of data is not evidence of anything."""
        detector = make(window_seconds=60.0, min_window_fill=0.30)
        verdict = feed(detector, {"neck_flexion": 120.0}, 3.0)
        assert verdict.state == det.GOOD
        assert "filling" in verdict.reason

    def test_judges_once_enough_time_has_passed(self):
        detector = make(window_seconds=60.0, min_window_fill=0.30)
        verdict = feed(detector, {"neck_flexion": 120.0}, 25.0)
        assert verdict.state == det.BAD


class TestDirection:
    def test_improving_a_one_sided_metric_never_flags(self):
        """Neck flexion above baseline is better posture, not worse."""
        detector = make()
        verdict = feed(detector, {"neck_flexion": 179.0}, 90.0)
        assert verdict.state == det.GOOD

    def test_lowering_a_one_sided_metric_flags(self):
        detector = make()
        assert feed(detector, {"neck_flexion": 140.0}, 90.0).state == det.BAD

    def test_leaning_back_does_not_flag_torso_lean(self):
        detector = make()
        verdict = feed(detector, {"torso_lean": -30.0}, 90.0)
        assert verdict.state == det.GOOD

    def test_two_sided_metric_flags_either_way(self):
        front = baseline(1, "front", shoulder_tilt=0.0, head_roll=0.0,
                         lateral_offset=0.0)
        for value in (-25.0, 25.0):
            detector = det.PostureDetector(det.DetectionSettings(), {1: front})
            t = 0.0
            verdict = None
            while t < 90.0:
                verdict = detector.update(
                    [sample("front", {"head_roll": value}, camera=1)], now=t)
                t += STEP
            assert verdict.state == det.BAD, value


class TestHysteresis:
    def test_recovering_only_to_the_entry_threshold_stays_bad(self):
        """The whole point: sitting at the boundary must not flap the state.

        Entry tolerance is 8 deg (the spec floor beats 3 * spread here), so
        the exit tolerance is 0.7 * 8 = 5.6. An excess of 6.5 sits between the
        two: it would not have flagged the metric in the first place, but it is
        not enough to clear it once flagged.
        """
        detector = make(tolerance_multiplier=3.0, exit_ratio=0.70)
        assert feed(detector, {"neck_flexion": 160.0}, 90.0).state == det.BAD
        verdict = feed(detector, {"neck_flexion": 163.5}, 90.0, t0=90.0)
        assert verdict.state == det.BAD

    def test_recovering_past_the_exit_threshold_clears(self):
        detector = make(tolerance_multiplier=3.0, exit_ratio=0.70)
        assert feed(detector, {"neck_flexion": 160.0}, 90.0).state == det.BAD
        verdict = feed(detector, {"neck_flexion": 170.0}, 90.0, t0=90.0)
        assert verdict.state == det.GOOD

    def test_without_hysteresis_the_same_posture_would_clear(self):
        """Confirms the previous test is actually measuring hysteresis."""
        detector = make(tolerance_multiplier=3.0, exit_ratio=1.0)
        assert feed(detector, {"neck_flexion": 160.0}, 90.0).state == det.BAD
        verdict = feed(detector, {"neck_flexion": 163.5}, 90.0, t0=90.0)
        assert verdict.state == det.GOOD


class TestFusion:
    def test_worst_camera_wins_per_metric(self):
        """Two side cameras, one seeing a slouch: that is enough."""
        second = baseline(1, "side", neck_flexion=170.0)
        detector = det.PostureDetector(det.DetectionSettings(),
                                       {0: SIDE, 1: second})
        t = 0.0
        verdict = None
        while t < 90.0:
            verdict = detector.update([
                sample("side", {"neck_flexion": 170.0}, camera=0),
                sample("side", {"neck_flexion": 140.0}, camera=1),
            ], now=t)
            t += STEP
        assert verdict.state == det.BAD

    def test_a_camera_without_a_baseline_is_ignored(self):
        detector = make()
        t = 0.0
        verdict = None
        while t < 90.0:
            verdict = detector.update([
                sample("side", {"neck_flexion": 170.0}, camera=0),
                sample("side", {"neck_flexion": 100.0}, camera=9),
            ], now=t)
            t += STEP
        assert verdict.state == det.GOOD

    def test_single_camera_still_works(self):
        detector = make()
        assert feed(detector, {"neck_flexion": 170.0}, 90.0).state == det.GOOD

    def test_cameras_reporting_at_different_rates(self):
        """Cameras are not synchronised, and adaptive sampling makes their
        rates differ. A slow camera must still contribute rather than looking
        unmeasurable every time the fast one reports first."""
        second = baseline(1, "side", neck_flexion=170.0)
        detector = det.PostureDetector(det.DetectionSettings(),
                                       {0: SIDE, 1: second})
        t = 0.0
        verdict = None
        while t < 90.0:
            # Camera 0 at 5 Hz sees good posture; camera 1 at 1 Hz sees a slouch.
            detector.update([sample("side", {"neck_flexion": 172.0}, camera=0)], now=t)
            if abs(t % 1.0) < 1e-9:
                verdict = detector.update(
                    [sample("side", {"neck_flexion": 140.0}, camera=1)], now=t)
            t += STEP
        assert verdict.state == det.BAD
        assert "neck_flexion" in verdict.offenders

    def test_a_slow_camera_alone_does_not_look_blind(self):
        detector = make()
        t = 0.0
        verdict = None
        while t < 120.0:
            verdict = detector.update(
                [sample("side", {"neck_flexion": 172.0}, camera=0)], now=t)
            t += 2.0  # 0.5 Hz, the away/idle rate
        assert verdict.state == det.GOOD


class TestReconfiguration:
    def test_recalibrating_discards_the_window(self):
        """Old verdicts say nothing about a new baseline."""
        detector = make()
        assert feed(detector, {"neck_flexion": 140.0}, 90.0).state == det.BAD
        detector.set_baselines({0: baseline(0, "side", neck_flexion=140.0)})
        verdict = feed(detector, {"neck_flexion": 140.0}, 5.0, t0=90.0)
        assert verdict.state == det.GOOD

    def test_settings_change_resets_windows(self):
        detector = make()
        feed(detector, {"neck_flexion": 140.0}, 90.0)
        detector.set_settings(det.DetectionSettings(window_seconds=30.0))
        verdict = feed(detector, {"neck_flexion": 140.0}, 2.0, t0=90.0)
        assert verdict.state == det.GOOD

    def test_override_replaces_the_derived_tolerance(self):
        detector = make(overrides={"neck_flexion": 50.0})
        verdict = feed(detector, {"neck_flexion": 140.0}, 90.0)
        assert verdict.state == det.GOOD

    def test_verdict_serializes(self):
        detector = make()
        payload = feed(detector, {"neck_flexion": 140.0}, 90.0).to_dict()
        assert payload["state"] == det.BAD
        assert payload["calibrated"] is True
        keys = {m["key"] for m in payload["metrics"]}
        assert "neck_flexion" in keys


class TestAdaptiveRate:
    RATES = dict(peak_hz=5.0, idle_hz=2.0, away_hz=0.5, active_ratio=0.5)

    def test_empty_chair_drops_to_the_slow_rate(self):
        assert det.choose_rate(det.AWAY, 0.0, **self.RATES) == 0.5

    def test_comfortably_good_posture_idles(self):
        assert det.choose_rate(det.GOOD, 0.1, **self.RATES) == 2.0

    def test_approaching_a_tolerance_returns_to_full_rate(self):
        """The rate must rise before the verdict does, not after."""
        assert det.choose_rate(det.GOOD, 0.6, **self.RATES) == 5.0

    def test_bad_posture_runs_at_full_rate(self):
        """Phase 3's hold-to-clear needs to see recovery promptly."""
        assert det.choose_rate(det.BAD, 2.0, **self.RATES) == 5.0

    def test_unreadable_landmarks_run_at_full_rate(self):
        assert det.choose_rate(det.UNKNOWN, 0.0, **self.RATES) == 5.0

    def test_boundary_is_inclusive(self):
        assert det.choose_rate(det.GOOD, 0.5, **self.RATES) == 5.0

    def test_a_realistic_day_averages_under_the_budget(self):
        """Rough sanity check on the reason adaptive sampling exists at all.

        Inference measured ~13.3 ms of CPU per sample, so a rate maps directly
        to a share of one core. A day that is mostly good posture with the
        chair empty a third of the time has to land under 10%.
        """
        ms_per_sample = 13.3
        day = [(det.AWAY, 0.0, 0.33), (det.GOOD, 0.1, 0.52),
               (det.GOOD, 0.7, 0.10), (det.BAD, 1.5, 0.05)]
        hz = sum(det.choose_rate(s, r, **self.RATES) * share for s, r, share in day)
        percent_of_core = hz * ms_per_sample / 10.0
        assert percent_of_core < 10.0
        # And confirm the fixed-rate version really would have been over.
        assert 5.0 * ms_per_sample / 10.0 > 6.0


class TestToleranceProvenance:
    """The panel has to be able to say where a tolerance came from."""

    def verdict_for(self, spread: float, **settings):
        base = cal.Baseline(camera=0, role="side", metrics={
            "neck_flexion": cal.MetricBaseline("neck_flexion", 170.0, spread, 50)})
        detector = det.PostureDetector(det.DetectionSettings(**settings), {0: base})
        v = feed(detector, {"neck_flexion": 170.0}, 30.0)
        return next(m for m in v.metrics if m.key == "neck_flexion")

    def test_a_spread_inside_the_band_is_learned(self):
        m = self.verdict_for(4.0)
        assert m.source == "learned"
        assert m.tolerance == pytest.approx(12.0)
        assert m.derived == pytest.approx(12.0)

    def test_a_tiny_spread_is_floored(self):
        m = self.verdict_for(0.1)
        assert m.source == "floored"
        assert m.tolerance == m.floor

    def test_a_wide_spread_is_capped(self):
        """The user's real calibration: 9.0 deg spread, 27 derived, 15 ceiling."""
        m = self.verdict_for(9.0)
        assert m.source == "capped"
        assert m.tolerance == m.ceiling == 15.0
        assert m.derived == pytest.approx(27.0)

    def test_an_override_is_reported_as_manual(self):
        m = self.verdict_for(4.0, overrides={"neck_flexion": 6.5})
        assert m.source == "manual"
        assert m.tolerance == pytest.approx(6.5)

    def test_lowering_the_ceiling_makes_the_metric_complain_sooner(self):
        """The correction that prompted this: a lower ceiling is more strict.

        With a 9 deg spread the tolerance is whatever the ceiling is, so a 25
        deg ceiling needs a 25 deg slouch and a 15 deg ceiling needs 15.
        """
        spec = met.SPEC_BY_KEY["neck_flexion"]
        base = cal.MetricBaseline("neck_flexion", 159.6, 9.0, 51)
        assert base.tolerance(spec, 3.0) == 15.0
        assert spec.max_tolerance == 15.0, "neck flexion ceiling was lowered to 15"
