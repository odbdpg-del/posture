"""The full-screen overlay must always be escapable by doing what it asks.

Reported from a live session: "very often this screen is almost impossible to
get back out of". It was not almost impossible, it was impossible. A torso lean
baseline of -15.9 degrees -- captured while reclining -- meant the metric
demanded a torso reclined at least 8.9 degrees past vertical before it counted
as within tolerance. The overlay's hold only advances on a GOOD reading, one
permanently-flagged metric pins the whole state to BAD, so sitting up straight
-- the thing the window was asking for -- could never clear it. Snooze was the
only way out.

Two defences, because they fail differently. The detector refuses to judge you
against a baseline no neutral posture can reach, which fixes the diagnosable
case at the root. The alert engine stands the overlay down when the hold has
never once started, which covers the cases nothing can diagnose -- a camera
nudged since calibration, a landmark being read off the wrong thing.
"""

from __future__ import annotations

from posture import alerts as al
from posture import detector as det
from posture import metrics as met
from posture.calibration import Baseline, MetricBaseline


def baseline(camera: int, **metrics: float) -> Baseline:
    return Baseline(camera=camera, role="side", metrics={
        key: MetricBaseline(key, centre, 1.0, 50) for key, centre in metrics.items()})


def sample(camera: int = 0, **values: float) -> met.MetricSample:
    return met.MetricSample(role="side", t=0.0, person=True, camera_index=camera,
                            values=dict(values))


def detector_with(**metrics: float) -> det.PostureDetector:
    d = det.PostureDetector(det.DetectionSettings(min_window_fill=0.0))
    d.set_baselines({0: baseline(0, **metrics)})
    return d


def drive(d: det.PostureDetector, seconds: float, step: float = 0.5, **values: float):
    verdict = None
    t = 0.0
    while t < seconds:
        verdict = d.update([sample(**values)], now=t)
        t += step
    return verdict


class TestABaselineNothingCanReach:
    def test_the_reclined_torso_baseline_is_not_judged(self):
        """The live numbers. Sitting upright must not read as bad against a
        baseline captured while reclining."""
        d = detector_with(torso_lean=-15.904)
        verdict = drive(d, 90.0, torso_lean=13.62)
        assert verdict.state == det.GOOD
        assert verdict.offenders == ()
        assert "Torso lean" in verdict.suspect

    def test_a_reachable_baseline_is_judged_normally(self):
        """The guard must not become a way for real deviations to escape."""
        d = detector_with(torso_lean=-2.0)
        verdict = drive(d, 90.0, torso_lean=20.0)
        assert verdict.state == det.BAD
        assert "torso_lean" in verdict.offenders
        assert verdict.suspect == ()

    def test_a_good_metric_beside_a_suspect_one_still_works(self):
        """The point is that one broken reference must not silence or pin
        everything else."""
        d = detector_with(torso_lean=-15.904, neck_tilt=5.0)
        verdict = drive(d, 90.0, torso_lean=13.62, neck_tilt=40.0)
        assert verdict.state == det.BAD
        assert verdict.offenders == ("neck_tilt",)
        assert verdict.suspect == ("Torso lean",)

    def test_the_row_is_marked_for_the_panel(self):
        d = detector_with(torso_lean=-15.904)
        verdict = drive(d, 90.0, torso_lean=13.62)
        row = next(m for m in verdict.metrics if m.key == "torso_lean")
        assert row.unreachable is True
        assert row.to_dict()["unreachable"] is True

    def test_it_is_reported_rather_than_silently_dropped(self):
        """A metric that stops being judged with no explanation is worse than
        one that nags: nothing would ever tell you to recalibrate."""
        d = detector_with(torso_lean=-15.904)
        assert "suspect" in drive(d, 90.0, torso_lean=13.62).to_dict()


def engine(**settings) -> al.AlertEngine:
    base = dict(subtle_after=0.0, notify_after=1.0, overlay_after=1.0,
                clear_hold=5.0, overlay_giveup_after=60.0)
    base.update(settings)
    return al.AlertEngine(al.AlertSettings(**base))


def run(eng: al.AlertEngine, state: str, seconds: float, t0: float = 0.0,
        step: float = 0.5):
    class V:
        def __init__(self, s):
            self.state, self.offenders = s, ("torso_lean",) if s == al.BAD else ()
    t = t0
    last = eng.state(t0)
    while t < t0 + seconds:
        last = eng.update(V(state), now=t)
        t += step
    return last, t


class TestTheOverlayStandsDown:
    def test_an_unclearable_overlay_gives_up(self):
        eng = engine()
        state, t = run(eng, al.BAD, 30.0)
        assert state.level == al.LEVEL_OVERLAY
        state, _ = run(eng, al.BAD, 45.0, t0=t)
        assert state.level == al.LEVEL_NONE
        assert state.snoozed, "clearing without snoozing would re-escalate at once"

    def test_it_says_so_rather_than_just_vanishing(self):
        eng = engine()
        run(eng, al.BAD, 90.0)
        assert any(e.kind == "stood_down" for e in eng.drain())

    def test_reaching_a_good_posture_once_disarms_it(self):
        """The discriminator. Someone who has managed a good reading has shown
        the target is reachable, and is being asked to hold it -- which is the
        mechanic, not a trap. The valve must never become a way to wait out
        the nag by sitting still and refusing."""
        eng = engine()
        _state, t = run(eng, al.BAD, 30.0)
        _state, t = run(eng, al.GOOD, 1.0, t0=t)       # a moment of good posture
        state, _ = run(eng, al.BAD, 200.0, t0=t)
        assert state.level == al.LEVEL_OVERLAY
        assert not state.snoozed

    def test_holding_long_enough_still_clears_it_normally(self):
        eng = engine()
        _state, t = run(eng, al.BAD, 30.0)
        state, _ = run(eng, al.GOOD, 6.0, t0=t)
        assert state.level == al.LEVEL_NONE
        assert not state.snoozed, "a hold that was met is not a stand-down"

    def test_it_can_be_switched_off(self):
        eng = engine(overlay_giveup_after=0.0)
        state, _ = run(eng, al.BAD, 300.0)
        assert state.level == al.LEVEL_OVERLAY

    def test_the_default_is_a_last_resort_not_a_nag_timeout(self):
        """Long enough that a person who intends to comply has had every
        chance, so this only ever fires on something the app got wrong."""
        assert al.AlertSettings().overlay_giveup_after >= 300.0

    def test_being_unable_to_see_you_does_not_count_as_progress(self):
        """UNKNOWN freezes the hold, so it never starts. That is exactly the
        stuck case the valve is for -- it must not be mistaken for a person
        making progress."""
        eng = engine()
        _state, t = run(eng, al.BAD, 30.0)
        state, _ = run(eng, al.UNKNOWN, 90.0, t0=t)
        assert state.level == al.LEVEL_NONE and state.snoozed
