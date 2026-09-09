"""The posture score: derived from real measurements, absent when there are none."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from posture import score as sc


@dataclass
class FakeMetric:
    key: str
    label: str
    ratio: float | None
    out_fraction: float
    flagged: bool = False


@dataclass
class FakeVerdict:
    state: str = "good"
    calibrated: bool = True
    reason: str = ""
    metrics: tuple = ()


def metric(ratio, out=0.0, key="neck_tilt", label="Neck tilt", flagged=False):
    return FakeMetric(key, label, ratio, out, flagged)


class TestAbsence:
    """A score is a claim about posture. Do not make one without evidence."""

    def test_uncalibrated_has_no_score(self):
        s = sc.score_verdict(FakeVerdict(calibrated=False))
        assert s.value is None and s.tone == "idle"
        assert "calibrate" in s.reason.lower()

    def test_empty_chair_has_no_score(self):
        s = sc.score_verdict(FakeVerdict(state="away"))
        assert s.value is None
        assert "nobody" in s.reason.lower()

    def test_unreadable_landmarks_have_no_score(self):
        s = sc.score_verdict(FakeVerdict(state="unknown", reason="cannot see left_hip"))
        assert s.value is None
        assert "left_hip" in s.reason

    def test_no_measurable_metric_has_no_score(self):
        assert sc.score_verdict(FakeVerdict(state="good", metrics=())).value is None

    def test_an_empty_chair_is_never_reported_as_excellent(self):
        """The failure that matters: 100/100 for furniture."""
        for state in ("away", "unknown"):
            assert sc.score_verdict(FakeVerdict(state=state)).value is None


class TestScoring:
    def test_sitting_at_baseline_is_a_hundred(self):
        s = sc.score_verdict(FakeVerdict(metrics=(metric(0.0, 0.0),)))
        assert s.value == 100 and s.band == "excellent" and s.tone == "good"
        assert s.worst is None

    def test_a_momentary_deviation_dents_but_does_not_wreck_it(self):
        """Reaching for a mug should not read as bad posture.

        It has to dent the score -- a number that ignored the present moment
        would be useless -- but with nothing in the window supporting it, the
        dent stops at the floor of "good".
        """
        s = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 0.0),)))
        assert s.value < 100
        assert s.band == "good"

    def test_the_same_deviation_sustained_scores_far_worse(self):
        momentary = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 0.0),))).value
        sustained = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 1.0),))).value
        assert sustained < momentary - 30

    def test_score_falls_as_deviation_grows(self):
        """While the window still vouches for it. Past that see
        TestSustainedEvidence, which is where it stops falling."""
        values = [sc.score_verdict(FakeVerdict(metrics=(metric(r, 0.9),))).value
                  for r in (0.0, 0.5, 1.0, 1.5)]
        assert values == sorted(values, reverse=True)
        assert len(set(values)) == len(values)

    def test_it_bottoms_out_rather_than_going_negative(self):
        s = sc.score_verdict(FakeVerdict(metrics=(metric(99.0, 1.0),)))
        assert s.value == 0 and s.tone == "bad"

    def test_the_worst_axis_decides(self):
        """Three good metrics must not hide one badly forward head."""
        s = sc.score_verdict(FakeVerdict(metrics=(
            metric(0.0, 0.0, "shoulder_tilt", "Shoulder tilt"),
            metric(0.0, 0.0, "head_roll", "Head roll"),
            metric(1.4, 0.9, "neck_tilt", "Neck tilt"),
        )))
        assert s.value < 30
        assert s.worst == "Neck tilt"

    def test_every_metric_is_reported_for_explanation(self):
        s = sc.score_verdict(FakeVerdict(metrics=(
            metric(0.2, 0.0, "a", "A"), metric(0.9, 0.5, "b", "B"))))
        assert [m.key for m in s.metrics] == ["a", "b"]
        assert s.metrics[0].score > s.metrics[1].score

    def test_a_missing_ratio_is_treated_as_no_deviation(self):
        s = sc.score_verdict(FakeVerdict(metrics=(metric(None, 0.0),)))
        assert s.value == 100


class TestBands:
    @pytest.mark.parametrize("value,band,tone", [
        (100, "excellent", "good"), (85, "excellent", "good"),
        (84, "good", "ok"), (70, "good", "ok"),
        (69, "fair", "warn"), (50, "fair", "warn"),
        (49, "needs correction", "bad"), (0, "needs correction", "bad"),
    ])
    def test_band_boundaries(self, value, band, tone):
        assert sc.band_for(value) == (band, tone)

    def test_serializes_for_the_panel(self):
        payload = sc.score_verdict(FakeVerdict(metrics=(metric(0.3, 0.1),))).to_dict()
        assert set(payload) == {"value", "band", "tone", "reason", "worst", "metrics"}
        assert isinstance(payload["metrics"], list)


class TestSustainedEvidence:
    """The score may not contradict the verdict it is derived from.

    Measured live: a metric out of tolerance for half its window scored 19 --
    "needs correction" -- while the detector's own state was "good" and no
    alert was firing. The detector will not call a metric bad until it has been
    out for ``bad_fraction`` of the window; the score gave 60 of those points
    away on the instantaneous reading alone.
    """

    def test_the_live_case_no_longer_reads_as_needing_correction(self):
        """The exact numbers off the running app."""
        s = sc.score_verdict(FakeVerdict(metrics=(metric(13.132, 0.518),)))
        assert s.band != "needs correction"
        assert s.value >= 50

    def test_an_unsupported_spike_cannot_leave_the_good_band(self):
        for ratio in (1.5, 5.0, 50.0, 1000.0):
            s = sc.score_verdict(FakeVerdict(metrics=(metric(ratio, 0.0),)))
            assert s.band in ("good", "excellent"), f"ratio {ratio} gave {s.value}"

    def test_it_plateaus_once_the_deviation_outruns_its_evidence(self):
        """The point of the cap: how far out you are right now is worth
        something, but not more than the window will vouch for."""
        values = [sc.score_verdict(FakeVerdict(metrics=(metric(r, 0.2),))).value
                  for r in (2.0, 10.0, 100.0)]
        assert len(set(values)) == 1

    def test_a_confirmed_deviation_may_still_bottom_out(self):
        """The cap must not defang the score. Past the detector's own
        threshold it lifts entirely."""
        s = sc.score_verdict(FakeVerdict(metrics=(metric(99.0, 1.0),)))
        assert s.value == 0

    def test_the_threshold_is_where_needs_correction_becomes_possible(self):
        below = sc.score_verdict(FakeVerdict(metrics=(metric(99.0, 0.69),))).value
        above = sc.score_verdict(FakeVerdict(metrics=(metric(99.0, 0.95),))).value
        assert below >= 50, "the detector calls this good; the score must not disagree"
        assert above < 50, "the detector calls this bad; the score must agree"

    def test_a_flagged_metric_is_never_capped_into_fair(self):
        """Hysteresis keeps a metric flagged while its out-fraction falls back
        under the threshold. An alert on screen beside a reassuring score would
        be the same contradiction the other way round."""
        s = sc.score_verdict(FakeVerdict(metrics=(metric(2.0, 0.4, flagged=True),)))
        assert s.value < 50

    def test_the_ceiling_is_continuous(self):
        """A jump would show up as the headline number lurching while you sat
        still. Both joints are checked, not just the obvious one."""
        for bad_fraction in (0.4, 0.7, 0.9):
            values = [sc.severity_ceiling(f / 500.0, bad_fraction)
                      for f in range(501)]
            steps = [b - a for a, b in zip(values, values[1:])]
            assert max(steps) < 0.02, f"bad_fraction={bad_fraction}"
            assert min(steps) >= 0.0, "the ceiling must never tighten as evidence grows"

    def test_the_anchors_land_on_real_band_boundaries(self):
        """A cap that stopped short of a label for no visible reason would be
        a number nobody could account for."""
        assert sc.band_for(int(round(100 * (1 - sc.UNSUPPORTED_CEILING))))[0] == "good"
        assert sc.band_for(int(round(100 * (1 - sc.THRESHOLD_CEILING))))[0] == "fair"

    def test_the_mirrored_threshold_matches_the_detector(self):
        """score.py stays a leaf by copying this constant. Pin the copy."""
        from posture.detector import DetectionSettings

        assert sc.DEFAULT_BAD_FRACTION == DetectionSettings().bad_fraction

    def test_the_monitor_passes_the_configured_threshold(self):
        """Not the mirrored default -- the user can change it."""
        import inspect

        from posture import monitor as mon

        src = inspect.getsource(mon.Monitor._ingest)
        assert "self._detector.settings.bad_fraction" in src
