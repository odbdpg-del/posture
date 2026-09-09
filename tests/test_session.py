"""Session tracks: duration, stretches, and movement.

The point of the module is the distinction between a moment and a pattern. The
score already says how you are sitting right now; none of it is interesting
unless it can also say for how long, in what stretches, and whether you moved
at all.
"""

from __future__ import annotations

from posture.session import (
    GAP_S,
    MIN_RUN_S,
    TRACK_INTERVAL_S,
    SessionAnalysis,
)


class Row:
    def __init__(self, key="neck_tilt", label="Neck tilt", unit="deg",
                 value=5.0, ratio=0.2, tolerance=6.0, baseline=5.0,
                 low_confidence=False):
        self.key, self.label, self.unit = key, label, unit
        self.value, self.ratio, self.tolerance = value, ratio, tolerance
        self.baseline, self.low_confidence = baseline, low_confidence


class Verdict:
    def __init__(self, *rows, state="good"):
        self.state, self.metrics = state, rows


def run(sa, seconds, t0=0.0, step=2.0, **row):
    t = t0
    while t < t0 + seconds:
        sa.record(Verdict(Row(**row)), now=t)
        t += step
    return t


def track(sa, key="neck_tilt"):
    return sa.tracks[key]


class TestDurationNotJustDeviation:
    def test_it_totals_the_time_past_tolerance(self):
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 60.0, value=5.0, ratio=0.2)
        run(sa, 100.0, t0=t, value=25.0, ratio=3.0)
        assert 95 <= track(sa).total_over_s <= 101

    def test_it_keeps_the_longest_continuous_stretch(self):
        """Twenty minutes in one sitting is a different thing from the same
        twenty minutes in forty scattered half-minutes, and a total alone
        cannot tell them apart."""
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        for _ in range(3):
            t = run(sa, 20.0, t0=t, value=25.0, ratio=3.0)
            t = run(sa, 20.0, t0=t, value=5.0, ratio=0.2)
        t = run(sa, 60.0, t0=t, value=25.0, ratio=3.0)
        assert track(sa).total_over_s > 100
        assert 55 <= track(sa).longest_over_s <= 61

    def test_a_share_of_the_session_is_reported(self):
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 100.0, value=5.0, ratio=0.2)
        run(sa, 100.0, t0=t, value=25.0, ratio=3.0)
        share = sa.to_dict()["tracks"][0]["share_over"]
        assert 0.45 <= share <= 0.55

    def test_a_flicker_is_not_counted_as_a_run(self):
        """Reaching for a mug is movement, not posture. Counting it would bury
        the stretches worth acting on."""
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        for _ in range(5):
            t = run(sa, 2.0, t0=t, value=25.0, ratio=3.0)
            t = run(sa, 20.0, t0=t, value=5.0, ratio=0.2)
        assert track(sa).runs == 0

    def test_a_real_stretch_is(self):
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 30.0, value=25.0, ratio=3.0)
        run(sa, 20.0, t0=t, value=5.0, ratio=0.2)
        assert track(sa).runs == 1
        assert MIN_RUN_S == 5.0


class TestGapsAreCreditedToNothing:
    def test_leaving_the_desk_does_not_extend_a_run(self):
        """Otherwise "you slouched for twenty minutes" and "you left" are the
        same reading."""
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 20.0, value=25.0, ratio=3.0)
        sa.record(Verdict(Row(value=25.0, ratio=3.0)), now=t + 600.0)
        assert track(sa).total_over_s < 30
        assert track(sa).longest_over_s < 30

    def test_the_session_clock_stops_too(self):
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 20.0)
        sa.record(Verdict(Row()), now=t + 600.0)
        assert sa.to_dict()["measured"] < 30

    def test_away_time_is_not_measured_time(self):
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        while t < 60.0:
            sa.record(Verdict(Row(), state="away"), now=t)
            t += 2.0
        assert sa.to_dict()["measured"] == 0.0


class TestConfidenceStopsTheClock:
    def test_an_untrusted_reading_counts_against_nothing(self):
        """The same treatment being out of frame gets: not evidence of a
        posture, so not evidence against one either."""
        sa = SessionAnalysis(started=0.0)
        run(sa, 100.0, value=25.0, ratio=3.0, low_confidence=True)
        assert track(sa).total_over_s == 0.0

    def test_a_trusted_one_does(self):
        sa = SessionAnalysis(started=0.0)
        run(sa, 100.0, value=25.0, ratio=3.0)
        assert track(sa).total_over_s > 90


class TestMovement:
    """Stillness is not the goal, so this describes rather than scores."""

    def test_settling_somewhere_new_counts_as_a_change(self):
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        for value in (5.0, 20.0, 5.0, 20.0):
            t = run(sa, 30.0, t0=t, value=value, ratio=0.2)
        assert track(sa).changes >= 3

    def test_wobbling_in_place_does_not(self):
        """A change has to move a full tolerance and hold there, or jitter
        would read as an active worker."""
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        for i in range(40):
            t = run(sa, 2.0, t0=t, value=5.0 + (i % 2) * 2.0, ratio=0.2)
        assert track(sa).changes == 0

    def test_a_brief_excursion_does_not(self):
        """Reaching for something and coming back is not a new position."""
        sa = SessionAnalysis(started=0.0)
        t = run(sa, 30.0, value=5.0, ratio=0.2)
        t = run(sa, 2.0, t0=t, value=30.0, ratio=0.2)
        run(sa, 30.0, t0=t, value=5.0, ratio=0.2)
        assert track(sa).changes == 0

    def test_an_hourly_rate_is_withheld_until_there_is_enough_session(self):
        """A rate extrapolated from ninety seconds is arithmetic, not a
        measurement."""
        sa = SessionAnalysis(started=0.0)
        run(sa, 60.0, value=5.0, ratio=0.2)
        assert sa.variability()["changes_per_hour"] is None

    def test_and_reported_once_there_is(self):
        sa = SessionAnalysis(started=0.0)
        t = 0.0
        for value in (5.0, 20.0, 5.0, 20.0, 5.0):
            t = run(sa, 120.0, t0=t, value=value, ratio=0.2)
        v = sa.variability()
        assert v["changes_per_hour"] is not None and v["changes_per_hour"] > 0

    def test_the_band_is_descriptive_and_never_a_verdict(self):
        sa = SessionAnalysis(started=0.0)
        run(sa, 120.0, value=5.0, ratio=0.1)
        assert sa.variability()["band"] in ("low", "moderate", "high")


class TestTracksForCharts:
    def test_points_are_never_closer_than_the_chart_interval(self):
        """The interval is a floor, not a cadence: samples arrive at whatever
        rate adaptive sampling has chosen, and a point is kept when the last
        one is old enough. A chart a few hundred pixels wide cannot show more,
        and the payload has to stay small enough to send whole."""
        sa = SessionAnalysis(started=0.0)
        run(sa, 300.0, step=2.0, value=5.0, ratio=0.2)
        points = sa.to_dict()["tracks"][0]["points"]
        assert len(points) > 20, "an hour of chart needs points in it"
        gaps = [b["t"] - a["t"] for a, b in zip(points, points[1:])]
        assert min(gaps) >= TRACK_INTERVAL_S

    def test_a_point_says_whether_it_was_past_tolerance(self):
        sa = SessionAnalysis(started=0.0)
        run(sa, 60.0, value=25.0, ratio=3.0)
        assert all(p["over"] for p in sa.to_dict()["tracks"][0]["points"])

    def test_a_row_carrying_only_a_key_still_gets_a_track(self):
        """Duck-typed like the score, so this stays a leaf module."""
        class Bare:
            key = "neck_tilt"
        sa = SessionAnalysis(started=0.0)
        sa.record(Verdict(Bare()), now=0.0)
        assert sa.tracks["neck_tilt"].label == "neck_tilt"

    def test_the_gap_constant_is_the_one_the_docstring_claims(self):
        assert GAP_S == 5.0
