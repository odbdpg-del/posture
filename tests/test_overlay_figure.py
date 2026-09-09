"""The figure on the full-screen overlay, and what the hold readout says.

The overlay's job at this point is no longer just to nag: it has to explain why
it is still up. Two things it says are tested here, both without a display --
the geometry is pure, and the readout was pulled out of the Tk loop so it could
be.

The bug behind all of this: when the landmarks the metrics need drop below the
visibility threshold, the detector reports UNKNOWN and the hold timer freezes.
The window went on showing a countdown that could not move, which is
indistinguishable from an app that has decided your posture is wrong. Sitting
up straighter does nothing; only getting back in frame does.
"""

from __future__ import annotations

from posture.overlay_window import (
    BAD,
    GOOD,
    WARN,
    OverlayView,
    figure,
    hold_message,
)

# Payload order is monitor.PREVIEW_LANDMARKS, not raw MediaPipe indices.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_HIP, R_HIP, L_KNEE, R_KNEE = 9, 10, 11, 12


def seated(vis: float = 0.9, **override: float):
    """A plausible seated figure, with per-landmark visibility overrideable."""
    points = {
        NOSE: (0.50, 0.24), L_EYE: (0.47, 0.22), R_EYE: (0.53, 0.22),
        L_EAR: (0.44, 0.24), R_EAR: (0.56, 0.24),
        L_SHOULDER: (0.40, 0.44), R_SHOULDER: (0.60, 0.44),
        L_ELBOW: (0.36, 0.62), R_ELBOW: (0.64, 0.62),
        L_HIP: (0.43, 0.74), R_HIP: (0.57, 0.74),
        L_KNEE: (0.42, 0.95), R_KNEE: (0.58, 0.95),
    }
    return [[x, y, override.get(str(i), vis)] for i, (x, y) in points.items()]


def kinds(prims, shape=None):
    return [p[-1] for p in prims if shape is None or p[0] == shape]


class TestFigure:
    def test_nothing_to_draw_without_landmarks(self):
        assert figure([], 0.5) == []

    def test_a_seated_figure_draws_the_measured_segments(self):
        prims = figure(seated(), 0.5)
        lines = [p for p in prims if p[0] == "line"]
        assert kinds(lines).count("key") == 3, (
            "shoulder line, ear-to-shoulder and shoulder-to-hip")
        assert "ref" in kinds(lines), "the vertical to line up against"
        assert "faint" in kinds(lines), "limbs carry the framing"

    def test_faint_landmarks_are_marked_lost_rather_than_dropped(self):
        """"Not tracked" and "tracked but not trusted" are different, and the
        second is the one you can fix by moving."""
        prims = figure(seated(vis=0.2), 0.5)
        dots = [p for p in prims if p[0] == "dot"]
        assert kinds(dots) and set(kinds(dots)) == {"lost"}

    def test_invisible_landmarks_are_not_drawn_at_all(self):
        prims = figure(seated(vis=0.0), 0.5)
        assert prims == []

    def test_no_torso_line_when_the_hips_are_out_of_frame(self):
        """The desk case: hips under the desk. Drawing a line to a landmark the
        app is ignoring would imply a measurement that is not happening."""
        hidden = {str(L_HIP): 0.05, str(R_HIP): 0.05}
        prims = figure(seated(**hidden), 0.5)
        lines = [p for p in prims if p[0] == "line"]
        assert kinds(lines).count("key") == 2, "shoulders and neck only"

    def test_the_neck_segment_survives_without_hips(self):
        """Neck tilt is the metric that works from head and shoulders alone, so
        its segment has to be drawn when the hips are gone."""
        hidden = {str(L_HIP): 0.05, str(R_HIP): 0.05}
        prims = figure(seated(**hidden), 0.5)
        ear = (0.44 + 0.56) / 2
        assert any(p[0] == "line" and p[-1] == "key" and abs(p[1] - ear) < 1e-9
                   for p in prims)

    def test_one_visible_shoulder_still_anchors_the_figure(self):
        """Turned slightly away is the common case, not an error."""
        prims = figure(seated(**{str(R_SHOULDER): 0.05}), 0.5)
        assert "key" in kinds([p for p in prims if p[0] == "line"])

    def test_coordinates_stay_normalized(self):
        for prim in figure(seated(), 0.5):
            for value in prim[1:-1]:
                assert 0.0 <= value <= 1.0


class TestHoldMessage:
    def test_a_good_posture_counts_down(self):
        text, colour = hold_message(OverlayView(state="good"), 3.0)
        assert "3s to go" in text and colour == GOOD

    def test_a_bad_posture_says_what_to_do(self):
        text, colour = hold_message(OverlayView(state="bad"), None)
        assert "Sit back" in text and colour == BAD

    def test_an_unmeasurable_reading_says_the_hold_is_paused(self):
        """The whole point: a frozen countdown must not look like a verdict."""
        view = OverlayView(state="unknown", reason="cannot see left_hip")
        text, colour = hold_message(view, 5.0)
        assert "paused" in text
        assert colour == WARN

    def test_landmark_names_are_readable_rather_than_identifiers(self):
        view = OverlayView(state="unknown", reason="cannot see left_hip")
        text, _ = hold_message(view, 5.0)
        assert "left hip" in text and "_" not in text

    def test_it_never_shows_a_countdown_it_cannot_honour(self):
        """A remaining time is still passed while unknown -- it just froze. The
        readout must not repeat it as though it were ticking."""
        view = OverlayView(state="unknown", reason="")
        text, _ = hold_message(view, 4.0)
        assert "4s" not in text

    def test_measuring_is_exactly_the_two_states_that_move_the_hold(self):
        assert OverlayView(state="good").measuring
        assert OverlayView(state="bad").measuring
        assert not OverlayView(state="unknown").measuring
        assert not OverlayView(state="away").measuring


class TestTheOverlayNeverSeesAFrame:
    def test_the_view_carries_positions_only(self):
        """The overlay covers the whole screen, including anything being
        screen-shared. It gets landmark coordinates and nothing else."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(OverlayView)}
        assert fields == {"landmarks", "thresh", "aspect", "state", "reason"}
        for banned in ("image", "frame", "jpeg", "pixels"):
            assert not any(banned in name for name in fields)
