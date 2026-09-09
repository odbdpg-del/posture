"""Alert escalation, hold-to-clear, and the ways out of an alert.

Time is passed in explicitly, so a ninety-second escalation is exercised in
microseconds and every threshold is checked exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from posture import alerts as al


@dataclass
class FakeVerdict:
    state: str = al.GOOD
    offenders: tuple = ()


def engine(**overrides) -> al.AlertEngine:
    return al.AlertEngine(al.AlertSettings(**overrides))


def run(eng, state, seconds, *, t0=0.0, step=0.5, offenders=("neck_tilt",)):
    """Hold one posture state for a stretch. Returns the final AlertState."""
    t = t0
    out = None
    while t < t0 + seconds:
        out = eng.update(FakeVerdict(state, offenders if state == al.BAD else ()), now=t)
        t += step
    return out, t


class TestEscalation:
    def test_good_posture_never_alerts(self):
        eng = engine()
        state, _ = run(eng, al.GOOD, 300)
        assert state.level == al.LEVEL_NONE
        assert not state.headline

    def test_subtle_comes_first(self):
        eng = engine()
        state, _ = run(eng, al.BAD, 5)
        assert state.level == al.LEVEL_SUBTLE

    def test_notification_at_thirty_seconds(self):
        eng = engine()
        before, t = run(eng, al.BAD, 29)
        assert before.level == al.LEVEL_SUBTLE
        after, _ = run(eng, al.BAD, 4, t0=t)
        assert after.level == al.LEVEL_NOTIFY

    def test_overlay_at_ninety_seconds(self):
        """Thirty to notify, sixty more to the overlay."""
        eng = engine()
        before, t = run(eng, al.BAD, 89)
        assert before.level == al.LEVEL_NOTIFY
        after, _ = run(eng, al.BAD, 3, t0=t)
        assert after.level == al.LEVEL_OVERLAY

    def test_it_does_not_climb_past_the_overlay(self):
        eng = engine()
        state, _ = run(eng, al.BAD, 300)
        assert state.level == al.LEVEL_OVERLAY

    def test_timings_are_configurable(self):
        eng = engine(subtle_after=2.0, notify_after=3.0, overlay_after=4.0)
        assert run(eng, al.BAD, 1)[0].level == al.LEVEL_NONE
        assert run(eng, al.BAD, 2.5)[0].level == al.LEVEL_SUBTLE
        assert run(eng, al.BAD, 6)[0].level == al.LEVEL_NOTIFY
        assert run(eng, al.BAD, 11)[0].level == al.LEVEL_OVERLAY

    def test_disabled_never_escalates(self):
        eng = engine(enabled=False)
        assert run(eng, al.BAD, 300)[0].level == al.LEVEL_NONE


class TestClearing:
    def test_a_low_alert_clears_as_soon_as_posture_is_good(self):
        eng = engine()
        bad, t = run(eng, al.BAD, 40)
        assert bad.level == al.LEVEL_NOTIFY
        good, _ = run(eng, al.GOOD, 1, t0=t)
        assert good.level == al.LEVEL_NONE

    def test_the_overlay_needs_a_sustained_hold(self):
        """The core mechanic: a moment of good posture is not enough."""
        eng = engine(clear_hold=5.0)
        _, t = run(eng, al.BAD, 95)
        brief, t2 = run(eng, al.GOOD, 3, t0=t)
        assert brief.level == al.LEVEL_OVERLAY, "3s must not dismiss a 5s hold"
        held, _ = run(eng, al.GOOD, 3, t0=t2)
        assert held.level == al.LEVEL_NONE

    def test_slipping_during_the_hold_restarts_it(self):
        eng = engine(clear_hold=5.0)
        _, t = run(eng, al.BAD, 95)
        _, t = run(eng, al.GOOD, 4, t0=t)
        _, t = run(eng, al.BAD, 1, t0=t)         # one bad reading
        nearly, t = run(eng, al.GOOD, 4, t0=t)
        assert nearly.level == al.LEVEL_OVERLAY, "the hold must have restarted"
        done, _ = run(eng, al.GOOD, 2, t0=t)
        assert done.level == al.LEVEL_NONE

    def test_looking_away_does_not_dismiss_the_overlay(self):
        """"Not bad" includes "I cannot see you". Leaning out of frame must not
        be a way to get rid of it."""
        eng = engine(clear_hold=5.0)
        _, t = run(eng, al.BAD, 95)
        hidden, _ = run(eng, al.UNKNOWN, 60, t0=t)
        assert hidden.level == al.LEVEL_OVERLAY

    def test_hold_progress_is_reported(self):
        eng = engine(clear_hold=5.0)
        _, t = run(eng, al.BAD, 95)
        part, _ = run(eng, al.GOOD, 2, t0=t)
        assert part.hold_remaining == pytest.approx(3.0, abs=0.6)
        assert part.hold_required == 5.0


class TestAbsence:
    def test_leaving_the_desk_cancels_the_alert(self):
        """An alert that survives you leaving would ambush you on return."""
        eng = engine()
        bad, t = run(eng, al.BAD, 95)
        assert bad.level == al.LEVEL_OVERLAY
        away, _ = run(eng, al.AWAY, 5, t0=t)
        assert away.level == al.LEVEL_NONE

    def test_returning_starts_from_nothing(self):
        eng = engine()
        _, t = run(eng, al.BAD, 95)
        _, t = run(eng, al.AWAY, 30, t0=t)
        back, _ = run(eng, al.BAD, 5, t0=t)
        assert back.level == al.LEVEL_SUBTLE

    def test_an_unreadable_frame_freezes_rather_than_escalates(self):
        eng = engine()
        _, t = run(eng, al.BAD, 20)
        frozen, _ = run(eng, al.UNKNOWN, 300, t0=t)
        assert frozen.level == al.LEVEL_SUBTLE, "must not climb while blind"

    def test_an_unreadable_frame_does_not_clear_either(self):
        eng = engine()
        _, t = run(eng, al.BAD, 40)
        frozen, _ = run(eng, al.UNKNOWN, 300, t0=t)
        assert frozen.level == al.LEVEL_NOTIFY


class TestSnooze:
    def test_snoozing_clears_and_suppresses(self):
        eng = engine(snooze_seconds=60.0)
        _, t = run(eng, al.BAD, 95)
        eng.snooze(now=t)
        during, _ = run(eng, al.BAD, 50, t0=t)
        assert during.level == al.LEVEL_NONE
        assert during.snoozed

    def test_it_is_time_boxed_and_resumes(self):
        """A snooze must never quietly become "off forever"."""
        eng = engine(snooze_seconds=60.0)
        _, t = run(eng, al.BAD, 95)
        eng.snooze(now=t)
        _, t = run(eng, al.BAD, 61, t0=t)
        after, _ = run(eng, al.BAD, 5, t0=t)
        assert not after.snoozed
        assert after.level == al.LEVEL_SUBTLE, "escalation restarts from nothing"

    def test_it_does_not_resume_mid_way_up(self):
        eng = engine(snooze_seconds=30.0)
        _, t = run(eng, al.BAD, 95)
        eng.snooze(now=t)
        _, t = run(eng, al.BAD, 31, t0=t)
        resumed, _ = run(eng, al.BAD, 25, t0=t)
        assert resumed.level == al.LEVEL_SUBTLE

    def test_snooze_length_is_configurable(self):
        eng = engine(snooze_seconds=5.0)
        eng.snooze(now=0.0)
        assert run(eng, al.BAD, 4, t0=0.0)[0].snoozed
        assert not run(eng, al.BAD, 3, t0=5.5)[0].snoozed

    def test_an_explicit_length_overrides_the_default(self):
        eng = engine(snooze_seconds=600.0)
        state = eng.snooze(seconds=30.0, now=0.0)
        assert state.snoozed_until == 30.0

    def test_it_can_be_cancelled(self):
        eng = engine(snooze_seconds=600.0)
        eng.snooze(now=0.0)
        eng.cancel_snooze(now=10.0)
        assert not eng.state(10.0).snoozed
        assert run(eng, al.BAD, 5, t0=10.0)[0].level == al.LEVEL_SUBTLE

    def test_there_is_no_one_click_dismiss(self):
        """The only ways out are fixing your posture or an explicit, timed
        snooze. Assert the engine exposes nothing else."""
        public = {n for n in dir(al.AlertEngine) if not n.startswith("_")}
        assert "dismiss" not in public and "clear" not in public
        assert {"snooze", "cancel_snooze", "update"} <= public


class TestEvents:
    def test_each_escalation_is_announced_once(self):
        eng = engine()
        run(eng, al.BAD, 95)
        kinds = [(e.kind, e.level) for e in eng.drain()]
        assert kinds == [("escalate", al.LEVEL_SUBTLE),
                         ("escalate", al.LEVEL_NOTIFY),
                         ("escalate", al.LEVEL_OVERLAY)]

    def test_no_repeat_events_while_sitting_at_a_level(self):
        """The notification must fire once, not once per sample."""
        eng = engine()
        run(eng, al.BAD, 45)
        eng.drain()
        run(eng, al.BAD, 40, t0=45)
        assert eng.drain() == []

    def test_clearing_is_announced(self):
        eng = engine()
        _, t = run(eng, al.BAD, 40)
        eng.drain()
        run(eng, al.GOOD, 2, t0=t)
        events = eng.drain()
        assert [e.kind for e in events] == ["clear"]
        assert events[0].previous == al.LEVEL_NOTIFY

    def test_optional_renotify_repeats(self):
        eng = engine(renotify_after=20.0)
        run(eng, al.BAD, 35)
        eng.drain()
        run(eng, al.BAD, 25, t0=35)
        assert any(e.kind == "escalate" for e in eng.drain())


class TestReporting:
    def test_headline_names_the_offenders(self):
        eng = engine()
        state, _ = run(eng, al.BAD, 5, offenders=("neck_tilt", "torso_lean"))
        head = state.headline.lower()
        assert "neck tilt" in head and "torso lean" in head

    def test_the_headline_reports_rather_than_diagnoses(self):
        """The app measures you against a baseline you set. It does not know
        whether your posture is healthy, and naming a body part as a fault
        claims knowledge it does not have."""
        eng = engine()
        state, _ = run(eng, al.BAD, 5, offenders=("neck_tilt",))
        head = state.headline.lower()
        assert "baseline" in head
        for word in ("fix your", "bad posture", "correct your", "poor", "wrong"):
            assert word not in head, head

    def test_the_headline_says_how_long(self):
        """A measurement without a duration cannot distinguish a stretch from a
        moment, which is the distinction the whole escalation rests on."""
        eng = engine()
        state, _ = run(eng, al.BAD, 95, offenders=("neck_tilt",))
        assert "1m" in state.headline

    def test_detail_explains_the_hold_once_the_overlay_is_up(self):
        eng = engine()
        state, _ = run(eng, al.BAD, 95)
        assert "baseline" in state.detail and "closes this" in state.detail

    def test_serializes_for_the_panel(self):
        eng = engine()
        state, _ = run(eng, al.BAD, 95)
        payload = state.to_dict(now=95.0)
        assert payload["level"] == al.LEVEL_OVERLAY and payload["name"] == "overlay"
        assert payload["active"] is True
        assert set(payload) >= {"level", "name", "active", "bad_for", "hold_remaining",
                                "snoozed", "snooze_remaining", "offenders", "headline"}

    def test_an_idle_state_serializes_too(self):
        payload = al.AlertEngine().state(0.0).to_dict(now=0.0)
        assert payload["active"] is False and payload["level"] == 0


class TestHoldGrace:
    """"Continuously within tolerance" cannot be claimed across a blind gap."""

    def test_a_long_blind_gap_voids_a_part_completed_hold(self):
        eng = engine(clear_hold=5.0, hold_grace=1.5)
        _, t = run(eng, al.BAD, 95)
        _, t = run(eng, al.GOOD, 3, t0=t)      # 3 of the 5 seconds
        _, t = run(eng, al.UNKNOWN, 20, t0=t)  # then out of frame
        nearly, t = run(eng, al.GOOD, 3, t0=t)
        assert nearly.level == al.LEVEL_OVERLAY, "the gap must have voided it"
        done, _ = run(eng, al.GOOD, 3, t0=t)
        assert done.level == al.LEVEL_NONE

    def test_a_brief_flicker_does_not_punish_you(self):
        """A dropped landmark for a moment is not leaving the desk."""
        eng = engine(clear_hold=5.0, hold_grace=1.5)
        _, t = run(eng, al.BAD, 95)
        _, t = run(eng, al.GOOD, 4, t0=t)
        _, t = run(eng, al.UNKNOWN, 1.0, t0=t)   # inside the grace
        done, _ = run(eng, al.GOOD, 1.5, t0=t)
        assert done.level == al.LEVEL_NONE

    def test_the_grace_is_configurable(self):
        eng = engine(clear_hold=5.0, hold_grace=10.0)
        _, t = run(eng, al.BAD, 95)
        _, t = run(eng, al.GOOD, 4, t0=t)
        _, t = run(eng, al.UNKNOWN, 8, t0=t)     # still inside a 10s grace
        done, _ = run(eng, al.GOOD, 1.5, t0=t)
        assert done.level == al.LEVEL_NONE


class TestOverlaySafety:
    """The screen-blocking window must not be able to trap you."""

    def test_it_takes_itself_down_when_updates_stop(self):
        """Unplug the camera while the overlay is up and nothing can dismiss
        it — no posture reading is arriving. It has to give up on its own."""
        import time as _time

        from posture.overlay_window import OverlayWindow

        ov = OverlayWindow(stale_after=0.4)
        if not ov.available:                      # headless CI
            pytest.skip("no display")
        ov.show("Fix your posture", "detail", 5.0, 5.0)
        assert ov.visible
        deadline = _time.monotonic() + 5.0
        while ov.visible and _time.monotonic() < deadline:
            _time.sleep(0.1)
        assert not ov.visible, "a stale overlay must hide itself"
        ov.stop()

    def test_updates_keep_it_alive(self):
        import time as _time

        from posture.overlay_window import OverlayWindow

        ov = OverlayWindow(stale_after=0.6)
        if not ov.available:
            pytest.skip("no display")
        ov.show("Fix your posture", "detail", 5.0, 5.0)
        for _ in range(6):
            _time.sleep(0.15)
            ov.update("Fix your posture", "detail", 5.0, 3.0)
        assert ov.visible, "a live overlay must not time out"
        ov.stop()
        assert not ov.visible


class TestNotifier:
    def test_it_picks_a_backend_and_never_raises(self):
        from posture.notify import Notifier

        n = Notifier()
        assert n.backend in ("toast", "banner")

    def test_a_broken_toast_falls_back_rather_than_failing(self, monkeypatch):
        """A reminder that silently fails to appear is worse than a plain one."""
        from posture.notify import Notifier

        n = Notifier()
        monkeypatch.setattr(n, "_toast", lambda *a: (_ for _ in ()).throw(RuntimeError))
        used = []
        monkeypatch.setattr(n, "_banner", lambda *a: used.append(a) or True)
        assert n.send("t", "m") is True
        assert used, "the fallback must have been used"

    def test_it_returns_false_rather_than_raising_when_all_routes_fail(self, monkeypatch):
        from posture.notify import Notifier

        n = Notifier()
        boom = lambda *a: (_ for _ in ()).throw(RuntimeError)  # noqa: E731
        monkeypatch.setattr(n, "_toast", boom)
        monkeypatch.setattr(n, "_banner", boom)
        assert n.send("t", "m") is False
