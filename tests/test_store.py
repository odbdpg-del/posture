"""SQLite stats: what is stored, and how a day's time is attributed.

The schema is dull; the interesting part is turning a stream of samples into
"you sat well for four hours" without inventing time you were not there.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from posture.store import Store, day_bounds, today


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "stats.db", max_gap=6.0)
    yield s
    s.close()


def at(day: str, hour: float) -> float:
    """Epoch seconds at a local-time hour on a given day."""
    start, _end = day_bounds(day)
    return start + hour * 3600


DAY = "2026-03-04"


def fill(store, day, start_hour, seconds, state, *, score=90, step=2.0):
    """Write a run of samples in one state."""
    t = at(day, start_hour)
    end = t + seconds
    while t < end:
        store.add_sample(t, score, state)
        t += step
    return t


class TestSchema:
    def test_it_creates_itself(self, tmp_path):
        path = tmp_path / "nested" / "stats.db"
        s = Store(path)
        assert path.exists()
        s.close()

    def test_reopening_keeps_the_data(self, tmp_path):
        path = tmp_path / "stats.db"
        first = Store(path)
        first.add_sample(at(DAY, 9), 90, "good")
        first.close()
        second = Store(path)
        assert len(second.samples(*day_bounds(DAY))) == 1
        second.close()

    def test_the_schema_holds_no_image_columns(self, store):
        """The database arrived after "frames never reach disk"; it must not be
        the thing that breaks it."""
        rows = store._rows(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE type = 'table'")
        for row in rows:
            cols = store._rows(f"PRAGMA table_info({row['name']})")  # noqa: SLF001
            for col in cols:
                assert "BLOB" not in (col["type"] or "").upper(), col["name"]
                assert not any(word in col["name"].lower() for word in
                               ("image", "frame", "jpeg", "png", "pixel"))


class TestTimeAttribution:
    def test_good_time_is_counted(self, store):
        fill(store, DAY, 9, 600, "good")
        stats = store.day_stats(DAY)
        assert stats.good_seconds == pytest.approx(600, abs=4)
        assert stats.bad_seconds == 0

    def test_good_and_bad_split(self, store):
        end = fill(store, DAY, 9, 600, "good")
        store.add_sample(end, 20, "bad")
        fill(store, DAY, 9 + 600 / 3600, 300, "bad", score=20)
        stats = store.day_stats(DAY)
        assert stats.good_seconds > stats.bad_seconds > 0
        assert 60 < stats.good_percent_helper() < 75

    def test_a_gap_is_attributed_to_nothing(self, store):
        """The app was closed. Counting that as good posture would be a lie —
        it is the difference between "you sat well for six hours" and "the app
        was shut for six hours"."""
        fill(store, DAY, 9, 60, "good")
        fill(store, DAY, 15, 60, "good")     # six hours later
        stats = store.day_stats(DAY)
        assert stats.good_seconds == pytest.approx(120, abs=8)
        assert stats.measured_seconds < 200

    def test_away_time_is_separate_from_bad(self, store):
        fill(store, DAY, 9, 120, "good")
        fill(store, DAY, 9 + 120 / 3600, 120, "away", score=None)
        stats = store.day_stats(DAY)
        assert stats.away_seconds > 100
        assert stats.bad_seconds == 0
        assert stats.measured_seconds == pytest.approx(stats.good_seconds, abs=4)

    def test_unknown_time_is_separate_too(self, store):
        fill(store, DAY, 9, 120, "unknown", score=None)
        stats = store.day_stats(DAY)
        assert stats.unknown_seconds > 100
        assert stats.good_seconds == 0 and stats.bad_seconds == 0


class TestStreak:
    def test_longest_good_streak(self, store):
        fill(store, DAY, 9, 120, "good")
        fill(store, DAY, 9 + 120 / 3600, 60, "bad", score=20)
        fill(store, DAY, 9 + 180 / 3600, 400, "good")
        stats = store.day_stats(DAY)
        assert stats.longest_good_streak == pytest.approx(400, abs=6)

    def test_a_gap_breaks_the_streak(self, store):
        """Twenty minutes of good, the laptop shut, twenty more — that is not a
        forty-minute streak."""
        fill(store, DAY, 9, 300, "good")
        fill(store, DAY, 14, 300, "good")
        stats = store.day_stats(DAY)
        assert stats.longest_good_streak == pytest.approx(300, abs=6)

    def test_no_good_time_means_no_streak(self, store):
        fill(store, DAY, 9, 120, "bad", score=10)
        assert store.day_stats(DAY).longest_good_streak == 0


class TestCountsAndTimeline:
    def test_alerts_counted_from_the_notify_step_up(self, store):
        store.add_alert(at(DAY, 9), 1, "escalate")     # subtle: not an alert
        store.add_alert(at(DAY, 10), 2, "escalate")
        store.add_alert(at(DAY, 11), 3, "escalate")
        store.add_alert(at(DAY, 11.5), 0, "clear")
        assert store.day_stats(DAY).alerts == 2

    def test_episodes_counted(self, store):
        store.upsert_episode(at(DAY, 9), at(DAY, 9.1), ("neck_tilt",), 20, {})
        store.upsert_episode(at(DAY, 10), at(DAY, 10.2), ("torso_lean",), 30, {})
        assert store.day_stats(DAY).episodes == 2

    def test_an_episode_is_updated_not_duplicated(self, store):
        started = at(DAY, 9)
        store.upsert_episode(started, None, ("neck_tilt",), 40, {})
        store.upsert_episode(started, at(DAY, 9.1), ("neck_tilt",), 22, {"neck_tilt": 14})
        rows = store.episodes(*day_bounds(DAY))
        assert len(rows) == 1
        assert rows[0]["worst_score"] == 22 and rows[0]["peak"] == {"neck_tilt": 14}
        assert not rows[0]["ongoing"]

    def test_timeline_buckets_the_day(self, store):
        fill(store, DAY, 9, 600, "good")
        stats = store.day_stats(DAY, buckets=96)
        assert len(stats.timeline) == 96, "every bucket, so the day keeps its shape"
        assert any(b["score"] is not None for b in stats.timeline)

    def test_empty_buckets_keep_their_place(self):
        """Ten busy minutes in a day is a thin bar in the morning, not one bar
        filling the whole chart. Dropping empty buckets lost the *when*."""
        with_data = [b for b in _one_hour_of_samples().timeline
                     if b["good"] or b["bad"] or b["other"]]
        assert len(with_data) < 96, "the rest of the day must still be present"
        assert all(b["good"] == 0 for b in _one_hour_of_samples().timeline[:30])

    def test_empty_days_are_empty_not_absent(self, store):
        stats = store.day_stats("2020-01-01")
        assert stats.samples == 0 and stats.timeline == []
        assert stats.good_fraction is None
        assert stats.to_dict()["good_percent"] is None

    def test_average_score_ignores_missing_scores(self, store):
        store.add_sample(at(DAY, 9), 100, "good")
        store.add_sample(at(DAY, 9) + 2, None, "away")
        store.add_sample(at(DAY, 9) + 4, 80, "good")
        assert store.day_stats(DAY).average_score == 90


def _one_hour_of_samples():
    s = Store(":memory:", max_gap=6.0)
    fill(s, DAY, 9, 3600, "good")
    stats = s.day_stats(DAY, buckets=96)
    s.close()
    return stats


class TestDaysAndPruning:
    def test_days_lists_what_has_data(self, store):
        fill(store, DAY, 9, 20, "good")
        fill(store, "2026-03-05", 9, 20, "good")
        assert store.days()[:2] == ["2026-03-05", DAY]

    def test_pruning_drops_old_rows_only(self, store):
        fill(store, "2020-01-01", 9, 20, "good")
        fill(store, DAY, 9, 20, "good")
        removed = store.prune(at(DAY, 0))
        assert removed > 0
        assert store.days() == [DAY]

    def test_pruning_covers_alerts_and_episodes(self, store):
        store.add_alert(at("2020-01-01", 9), 2, "escalate")
        store.upsert_episode(at("2020-01-01", 9), at("2020-01-01", 9.1), (), 10, {})
        store.prune(at(DAY, 0))
        assert store.day_stats("2020-01-01").alerts == 0
        assert store.day_stats("2020-01-01").episodes == 0

    def test_today_is_a_valid_day_string(self):
        start, end = day_bounds(today())
        assert start <= time.time() < end


class TestBounds:
    def test_day_bounds_are_local_midnight(self):
        start, end = day_bounds(DAY)
        assert datetime.fromtimestamp(start).hour == 0
        assert end - start == pytest.approx(86400, abs=3700)  # allows a DST day

    def test_a_bad_date_raises(self):
        with pytest.raises(ValueError):
            day_bounds("not-a-date")


# Small helper so the split test reads as a percentage rather than a ratio.
def _good_percent(self) -> float:
    fraction = self.good_fraction
    return 0.0 if fraction is None else fraction * 100


from posture.store import DayStats  # noqa: E402

DayStats.good_percent_helper = _good_percent
