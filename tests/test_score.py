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


@dataclass
class FakeVerdict:
    state: str = "good"
    calibrated: bool = True
    reason: str = ""
    metrics: tuple = ()


def metric(ratio, out=0.0, key="neck_tilt", label="Neck tilt"):
    return FakeMetric(key, label, ratio, out)


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
        """Reaching for a mug should not read as bad posture."""
        s = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 0.0),)))
        assert 50 < s.value < 70

    def test_the_same_deviation_sustained_scores_far_worse(self):
        momentary = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 0.0),))).value
        sustained = sc.score_verdict(FakeVerdict(metrics=(metric(1.0, 1.0),))).value
        assert sustained < momentary - 30

    def test_score_falls_as_deviation_grows(self):
        values = [sc.score_verdict(FakeVerdict(metrics=(metric(r, 0.5),))).value
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
