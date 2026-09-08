"""The history buffer: real samples, real episodes, no invented data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from posture.history import MIN_EPISODE_S, History


@dataclass
class FakeMetric:
    key: str = "neck_tilt"
    flagged: bool = False
    excess: float | None = None


@dataclass
class FakeVerdict:
    offenders: tuple = ()
    metrics: tuple = field(default_factory=tuple)


class TestSampling:
    def test_samples_are_rate_limited(self):
        """Recording on every frame must not fill the buffer at 5 Hz."""
        hist = History(window_s=600, interval_s=2.0)
        for i in range(50):
            hist.record(90, "good", FakeVerdict(), now=1000 + i * 0.2)
        assert 4 <= len(hist.samples()) <= 6

    def test_old_samples_fall_out_of_the_window(self):
        hist = History(window_s=10, interval_s=1.0)
        for i in range(40):
            hist.record(90, "good", FakeVerdict(), now=1000 + i)
        times = [t for t, _s, _st in hist.samples()]
        assert max(times) - min(times) <= 11

    def test_absent_scores_are_kept_as_gaps(self):
        """A gap must stay a gap, not be smoothed over — the timeline draws a
        break rather than implying a measurement that never happened."""
        hist = History(window_s=100, interval_s=1.0)
        hist.record(90, "good", FakeVerdict(), now=1000)
        hist.record(None, "away", FakeVerdict(), now=1001)
        hist.record(88, "good", FakeVerdict(), now=1002)
        assert [s[1] for s in hist.samples()] == [90, None, 88]

    def test_session_average_ignores_absent_scores(self):
        hist = History(window_s=100, interval_s=1.0)
        for i, score in enumerate([100, None, 80, None, 60]):
            hist.record(score, "good" if score else "away", FakeVerdict(), now=1000 + i)
        assert hist.session_average == 80

    def test_no_average_before_any_score(self):
        assert History().session_average is None

    def test_average_survives_samples_leaving_the_window(self):
        """It is a session figure, not a window figure."""
        hist = History(window_s=3, interval_s=1.0)
        for i in range(20):
            hist.record(50, "good", FakeVerdict(), now=1000 + i)
        assert len(hist.samples()) < 6
        assert hist.session_average == 50


class TestEpisodes:
    def bad(self, hist, start, seconds, offenders=("neck_tilt",), score=30):
        verdict = FakeVerdict(offenders=offenders,
                              metrics=(FakeMetric(flagged=True, excess=12.0),))
        for i in range(seconds):
            hist.record(score, "bad", verdict, now=start + i)

    def test_a_sustained_bad_stretch_is_recorded(self):
        hist = History(window_s=600, interval_s=1.0)
        self.bad(hist, 1000, 30)
        hist.record(90, "good", FakeVerdict(), now=1031)
        episodes = hist.episodes()
        assert len(episodes) == 1
        assert episodes[0].duration >= 29
        assert episodes[0].offenders == ("neck_tilt",)

    def test_a_momentary_blip_is_not(self):
        """Movement is not posture. Recording every twitch would bury the real
        episodes in noise."""
        hist = History(window_s=600, interval_s=1.0)
        short = MIN_EPISODE_S / 2.0
        self.bad(hist, 1000, 2)
        hist.record(90, "good", FakeVerdict(), now=1000 + short)
        assert short < MIN_EPISODE_S, "the fixture must be under the threshold"
        assert hist.episodes() == []

    def test_an_episode_exactly_at_the_threshold_is_kept(self):
        hist = History(window_s=600, interval_s=1.0)
        self.bad(hist, 1000, 2)
        hist.record(90, "good", FakeVerdict(), now=1000 + MIN_EPISODE_S)
        assert len(hist.episodes()) == 1

    def test_an_ongoing_episode_is_marked(self):
        hist = History(window_s=600, interval_s=1.0)
        self.bad(hist, 1000, 20)
        episode = hist.episodes()[-1]
        assert episode.ended is None
        assert episode.to_dict()["ongoing"] is True

    def test_peak_deviation_is_tracked(self):
        hist = History(window_s=600, interval_s=1.0)
        verdict = FakeVerdict(offenders=("neck_tilt",),
                              metrics=(FakeMetric(flagged=True, excess=8.0),))
        for i in range(10):
            hist.record(40, "bad", verdict, now=1000 + i)
        hist.record(40, "bad", FakeVerdict(
            offenders=("neck_tilt",),
            metrics=(FakeMetric(flagged=True, excess=19.0),)), now=1010)
        assert hist.episodes()[-1].peak["neck_tilt"] == 19.0

    def test_worst_score_is_the_low_point(self):
        hist = History(window_s=600, interval_s=1.0)
        for i, score in enumerate([40, 22, 35]):
            for j in range(4):
                hist.record(score, "bad", FakeVerdict(offenders=("a",)),
                            now=1000 + i * 4 + j)
        assert hist.episodes()[-1].worst_score == 22

    def test_separate_stretches_are_separate_episodes(self):
        hist = History(window_s=600, interval_s=1.0)
        self.bad(hist, 1000, 20)
        for i in range(20):
            hist.record(95, "good", FakeVerdict(), now=1020 + i)
        self.bad(hist, 1050, 20)
        assert len(hist.episodes()) == 2


class TestSerialization:
    def test_shape_the_timeline_expects(self):
        hist = History(window_s=600, interval_s=1.0)
        for i in range(5):
            hist.record(90, "good", FakeVerdict(), now=1000 + i)
        payload = hist.to_dict()
        assert set(payload) >= {"samples", "episodes", "session_average",
                                "count", "now", "window", "interval"}
        assert payload["samples"][0].keys() == {"t", "score", "state"}

    def test_window_argument_narrows_the_result(self):
        hist = History(window_s=600, interval_s=1.0)
        for i in range(100):
            hist.record(90, "good", FakeVerdict(), now=hist_now(i))
        wide = hist.to_dict()["count"]
        narrow = hist.to_dict(window_s=1)["count"]
        assert narrow <= wide

    def test_empty_history_serializes_cleanly(self):
        payload = History().to_dict()
        assert payload["samples"] == [] and payload["session_average"] is None


def hist_now(i):
    import time
    return time.time() - 100 + i
