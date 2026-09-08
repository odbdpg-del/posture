"""Metric tests against synthetic landmarks.

The invariance tests matter more than the exact-value ones. A posture metric
that changes when you move the camera closer, switch it to the other side of
the desk, or turn on mirroring is useless no matter how precise it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from posture import landmarks as lmk
from posture import metrics as met

from . import synthetic as syn

VIS = 0.6
T = 100.0


def side(arr, aspect=4 / 3):
    return met.compute_side(arr, aspect, VIS, T)


def front(arr, aspect=4 / 3):
    return met.compute_front(arr, aspect, VIS, T)


class TestSideBaseline:
    def test_perfectly_stacked_posture(self):
        s = side(syn.side_pose())
        assert s.person and s.complete
        assert s.values["neck_flexion"] == pytest.approx(180.0)
        assert s.values["forward_head"] == pytest.approx(0.0, abs=1e-9)
        assert s.values["torso_lean"] == pytest.approx(0.0, abs=1e-9)
        assert s.near_side == "left"
        assert s.facing == 1
        assert s.scale_kind == "torso"
        assert s.scale == pytest.approx(0.40)

    def test_forward_head_is_a_ratio_of_torso_length(self):
        s = side(syn.side_pose(torso=0.40, head_forward=0.08))
        assert s.values["forward_head"] == pytest.approx(0.20)

    def test_forward_head_reduces_neck_flexion(self):
        upright = side(syn.side_pose(head_forward=0.0)).values["neck_flexion"]
        slouched = side(syn.side_pose(head_forward=0.08)).values["neck_flexion"]
        assert slouched < upright
        # ear-shoulder is (0.08, 0.16), hip-shoulder is (0, -0.40).
        # The deviation from straight is atan(0.08/0.16) = 26.57 degrees.
        assert slouched == pytest.approx(180.0 - 26.565, abs=0.01)

    @pytest.mark.parametrize("lean", [0.0, 5.0, 15.0, -8.0])
    def test_torso_lean_matches_the_pose(self, lean):
        s = side(syn.side_pose(lean_deg=lean))
        assert s.values["torso_lean"] == pytest.approx(lean)

    def test_lean_toward_the_camera_is_positive(self):
        assert side(syn.side_pose(lean_deg=12.0)).values["torso_lean"] > 0


class TestSideInvariance:
    """Properties that must hold for the metrics to be worth alerting on."""

    def test_camera_on_the_other_side_of_the_desk(self):
        left = side(syn.side_pose(lean_deg=11.0, head_forward=0.07))
        right = side(syn.side_pose(lean_deg=11.0, head_forward=0.07, mirrored=True))
        assert left.facing == 1 and right.facing == -1
        for key in ("neck_flexion", "forward_head", "torso_lean"):
            assert left.values[key] == pytest.approx(right.values[key]), key

    @pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 1.0, 2.35])
    def test_frame_shape_does_not_matter(self, aspect):
        """A 4:3 and a 16:9 camera looking at the same body must agree.

        They only do because the metric frame undoes the separate width and
        height normalization; without it every angle would be off by a factor
        of the aspect ratio and a baseline taken on one camera would be
        meaningless on another.
        """
        pose = dict(lean_deg=9.0, head_forward=0.06)
        reference = side(syn.side_pose(aspect=4 / 3, **pose), 4 / 3)
        s = side(syn.side_pose(aspect=aspect, **pose), aspect)
        assert s.values == pytest.approx(reference.values)
        assert s.values["torso_lean"] == pytest.approx(9.0)
        assert s.values["forward_head"] == pytest.approx(0.15)

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.6])
    def test_distance_from_the_camera_does_not_matter(self, factor):
        """Scaling the whole body leaves angles and torso-normalized ratios
        untouched. This is what makes the baseline survive rolling your chair
        back."""
        s = side(syn.side_pose(
            torso=0.40 * factor, neck_len=0.16 * factor,
            head_forward=0.06 * factor, nose_ahead=0.06 * factor,
            origin=(0.55, -0.90 + 0.15 * factor),
        ))
        assert s.values["forward_head"] == pytest.approx(0.15)
        assert s.values["neck_flexion"] == pytest.approx(159.44, abs=0.01)
        assert s.values["torso_lean"] == pytest.approx(0.0, abs=1e-9)

    def test_either_body_side_works(self):
        left = side(syn.side_pose(side="left", head_forward=0.05))
        right = side(syn.side_pose(side="right", head_forward=0.05))
        assert left.near_side == "left" and right.near_side == "right"
        assert left.values == pytest.approx(right.values)

    def test_picks_the_more_visible_body_side(self):
        """With both sides present, the one with better visibility wins."""
        arr = syn.side_pose(side="left", head_forward=0.05)
        ear, shoulder, hip = lmk.SIDE_CHAIN["right"]
        for i in (ear, shoulder, hip):
            arr[i] = (0.5, 0.5, 0.0, 0.99)
        arr[list(lmk.SIDE_CHAIN["left"]), 3] = 0.7
        assert side(arr).near_side == "right"


class TestSideVisibility:
    def test_no_person_is_not_the_same_as_no_landmarks(self):
        absent = met.compute("side", None, 4 / 3, VIS, T)
        assert absent.person is False and absent.usable is False

        # Losing the shoulder is what actually blinds a side camera; it is the
        # vertex every side metric measures from.
        present = side(syn.side_pose(faint=(lmk.LEFT_SHOULDER,)))
        assert present.person is True and present.usable is False
        assert "left_shoulder" in present.missing

    def test_missing_nose_leaves_signed_metrics_unclaimed(self):
        s = side(syn.side_pose(lean_deg=10.0, faint=(lmk.NOSE,)))
        assert s.facing is None
        assert "forward_head" not in s.values
        assert any("nose" in n for n in s.notes)

    def test_upside_down_torso_discards_only_the_torso_metrics(self):
        """Shoulders below hips means the landmarks are wrong, not that you are
        upside down.

        The pose model extrapolates hips that are out of frame and reports the
        guess with high visibility, which produced torso lean readings of -90
        and -178 degrees on a real camera. The torso metrics have to go -- but
        neck tilt was measured from an ear and a shoulder the bad hip never
        touched, so throwing it away too would be superstition.
        """
        s = side(syn.side_pose(lean_deg=175.0))
        assert s.person and any("implausible" in n for n in s.notes)
        assert "torso_lean" not in s.values
        assert "neck_flexion" not in s.values
        assert "forward_head" not in s.values
        assert "neck_tilt" in s.values

    @pytest.mark.parametrize("lean", [-95.0, -70.0, 70.0, 120.0])
    def test_leans_beyond_the_plausible_limit_lose_the_torso(self, lean):
        s = side(syn.side_pose(lean_deg=lean))
        assert "torso_lean" not in s.values
        assert s.usable, "neck tilt should survive a bad hip"

    @pytest.mark.parametrize("lean", [-45.0, 0.0, 30.0, 55.0])
    def test_plausible_leans_still_work(self, lean):
        s = side(syn.side_pose(lean_deg=lean))
        assert s.usable
        assert s.values["torso_lean"] == pytest.approx(lean)

    def test_collapsed_landmarks_refuse_to_normalize(self):
        s = side(syn.side_pose(torso=0.005, neck_len=0.005, head_forward=0.001))
        assert not s.usable
        assert any("too close together" in n for n in s.notes)


class TestSideWithoutHips:
    """A desk camera usually cannot see hips. It must still work.

    Hips end up under the desk, below the frame, or behind an armrest. Making
    them a requirement meant the side camera reported nothing for most of the
    day, which is worse than reporting less.
    """

    def test_hips_out_of_frame_still_tracks(self):
        s = side(syn.side_pose(head_forward=0.06, faint=(lmk.LEFT_HIP,)))
        assert s.usable, "losing the hips must not blind the side camera"
        assert "neck_tilt" in s.values
        assert any("hips not in frame" in n for n in s.notes)

    def test_hip_metrics_are_absent_but_nothing_else_breaks(self):
        s = side(syn.side_pose(faint=(lmk.LEFT_HIP,)))
        for key in ("neck_flexion", "forward_head", "torso_lean"):
            assert key not in s.values
        assert s.scale_kind == "neck"
        assert not s.complete

    def test_neck_tilt_measures_the_head_forward_angle(self):
        """head_forward across neck_len is the tangent of the reported angle."""
        import math
        s = side(syn.side_pose(neck_len=0.16, head_forward=0.08,
                               faint=(lmk.LEFT_HIP,)))
        expected = math.degrees(math.atan2(0.08, 0.16))
        assert s.values["neck_tilt"] == pytest.approx(expected)

    def test_upright_head_reads_near_zero(self):
        s = side(syn.side_pose(head_forward=0.0, faint=(lmk.LEFT_HIP,)))
        assert s.values["neck_tilt"] == pytest.approx(0.0, abs=1e-9)

    def test_neck_tilt_survives_the_camera_moving_sides(self):
        left = side(syn.side_pose(head_forward=0.07, faint=(lmk.LEFT_HIP,)))
        right = side(syn.side_pose(head_forward=0.07, mirrored=True,
                                   faint=(lmk.RIGHT_HIP,)))
        assert left.values["neck_tilt"] == pytest.approx(right.values["neck_tilt"])

    def test_faint_ear_still_yields_torso_lean(self):
        """The other half of the same idea: an ear-less sample keeps the torso."""
        s = side(syn.side_pose(lean_deg=10.0, faint=(lmk.LEFT_EAR,)))
        assert s.values["torso_lean"] == pytest.approx(10.0)
        assert "neck_tilt" not in s.values
        assert "neck_flexion" not in s.values

    def test_an_ear_below_the_shoulder_is_rejected(self):
        """A seated head is above its shoulders. Anything else is the model
        fitting a person to a chair or a coat.

        This one fails flatteringly rather than obviously: neck tilt only counts
        upward deviation as bad, so a wildly negative reading produces zero
        excess and a confident posture score of 100. Seen on an empty chair.
        """
        s = side(syn.side_pose(neck_len=-0.16, head_forward=0.02,
                               faint=(lmk.LEFT_HIP,)))
        assert "neck_tilt" not in s.values
        assert any("implausible" in n for n in s.notes)

    @pytest.mark.parametrize("tilt", [-80.0, 95.0, 149.0])
    def test_impossible_head_angles_are_rejected(self, tilt):
        import math
        # Place the ear at the given angle from vertical above the shoulder.
        rad = math.radians(tilt)
        arr = syn.side_pose(neck_len=0.16 * math.cos(rad),
                            head_forward=0.16 * math.sin(rad),
                            faint=(lmk.LEFT_HIP,))
        assert "neck_tilt" not in side(arr).values

    def test_a_normal_head_angle_is_kept(self):
        import math
        rad = math.radians(25.0)
        arr = syn.side_pose(neck_len=0.16 * math.cos(rad),
                            head_forward=0.16 * math.sin(rad),
                            faint=(lmk.LEFT_HIP,))
        assert side(arr).values["neck_tilt"] == pytest.approx(25.0, abs=0.01)

    def test_only_a_shoulder_is_truly_required(self):
        s = side(syn.side_pose(faint=(lmk.LEFT_EAR, lmk.LEFT_HIP)))
        assert not s.usable
        assert s.near_side == "left", "it still knows which side it was looking at"


class TestFront:
    def test_square_on_baseline(self):
        f = front(syn.front_pose())
        assert f.person and f.complete
        assert f.values["shoulder_tilt"] == pytest.approx(0.0, abs=1e-9)
        assert f.values["head_roll"] == pytest.approx(0.0, abs=1e-9)
        assert f.values["lateral_offset"] == pytest.approx(0.0, abs=1e-9)
        assert f.scale_kind == "shoulders"
        assert f.scale == pytest.approx(0.30)

    @pytest.mark.parametrize("tilt", [0.0, 4.0, 12.0, -7.0])
    def test_shoulder_tilt_matches_the_pose(self, tilt):
        f = front(syn.front_pose(shoulder_tilt_deg=tilt))
        assert f.values["shoulder_tilt"] == pytest.approx(tilt)

    def test_right_shoulder_higher_is_positive(self):
        assert front(syn.front_pose(shoulder_tilt_deg=10.0)).values["shoulder_tilt"] > 0

    @pytest.mark.parametrize("roll", [0.0, 6.0, -11.0])
    def test_head_roll_matches_the_pose(self, roll):
        assert front(syn.front_pose(head_roll_deg=roll)).values["head_roll"] == pytest.approx(roll)

    def test_lateral_offset_is_a_ratio_of_shoulder_width(self):
        f = front(syn.front_pose(shoulder_width=0.30, lateral=0.06))
        assert f.values["lateral_offset"] == pytest.approx(0.20)

    def test_lean_toward_your_right_is_positive(self):
        assert front(syn.front_pose(lateral=0.05)).values["lateral_offset"] > 0
        assert front(syn.front_pose(lateral=-0.05)).values["lateral_offset"] < 0


class TestFrontInvariance:
    def test_mirrored_feed_reads_identically(self):
        """Whether the preview is mirrored is a display choice; it must not
        flip which shoulder the app thinks is drooping."""
        normal = front(syn.front_pose(shoulder_tilt_deg=8.0, head_roll_deg=5.0,
                                      lateral=0.04))
        mirrored = front(syn.front_pose(shoulder_tilt_deg=8.0, head_roll_deg=5.0,
                                        lateral=0.04, mirrored=True))
        for key in ("shoulder_tilt", "head_roll", "lateral_offset"):
            assert normal.values[key] == pytest.approx(mirrored.values[key]), key

    @pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 1.0, 2.35])
    def test_frame_shape_does_not_matter(self, aspect):
        pose = dict(shoulder_tilt_deg=6.0, head_roll_deg=4.0, lateral=0.045)
        reference = front(syn.front_pose(aspect=4 / 3, **pose), 4 / 3)
        f = front(syn.front_pose(aspect=aspect, **pose), aspect)
        assert f.values == pytest.approx(reference.values)
        assert f.values["shoulder_tilt"] == pytest.approx(6.0)
        assert f.values["head_roll"] == pytest.approx(4.0)
        assert f.values["lateral_offset"] == pytest.approx(0.15)

    @pytest.mark.parametrize("factor", [0.6, 1.0, 1.5])
    def test_distance_from_the_camera_does_not_matter(self, factor):
        f = front(syn.front_pose(
            shoulder_width=0.30 * factor, ear_width=0.13 * factor,
            neck_len=0.18 * factor, lateral=0.045 * factor,
            shoulder_tilt_deg=6.0, origin=(0.70, -0.75 + 0.13 * factor),
        ))
        assert f.values["shoulder_tilt"] == pytest.approx(6.0)
        assert f.values["lateral_offset"] == pytest.approx(0.15)


class TestFrontVisibility:
    def test_missing_shoulders_blocks_everything(self):
        f = front(syn.front_pose(faint=(lmk.RIGHT_SHOULDER,)))
        assert f.person and not f.usable
        assert "right_shoulder" in f.missing

    def test_missing_ears_still_gives_shoulder_tilt(self):
        f = front(syn.front_pose(shoulder_tilt_deg=7.0, faint=(lmk.LEFT_EAR,)))
        assert f.usable and not f.complete
        assert f.values["shoulder_tilt"] == pytest.approx(7.0)
        assert "head_roll" not in f.values
        assert "left_ear" in f.missing

    def test_turned_head_refuses_to_report_roll(self):
        """Ear-to-ear foreshortens to nothing when you look sideways, and the
        roll angle then swings on noise. Refuse rather than report garbage."""
        f = front(syn.front_pose(ear_width=0.02, shoulder_width=0.30))
        assert "head_roll" not in f.values
        assert any("turned" in n for n in f.notes)
        # The offset metric does not depend on ear separation, so it survives.
        assert "lateral_offset" in f.values

    def test_side_on_body_refuses_the_front_metrics(self):
        f = front(syn.front_pose(shoulder_width=0.01))
        assert not f.usable
        assert any("side-on" in n for n in f.notes)


class TestDispatch:
    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError):
            met.compute("diagonal", np.zeros((33, 4)), 4 / 3, VIS, T)

    def test_roles_cover_every_spec(self):
        assert sum(len(v) for v in met.SPECS_BY_ROLE.values()) == len(met.SPECS)

    def test_each_role_produces_exactly_its_own_metrics(self):
        s = side(syn.side_pose())
        f = front(syn.front_pose())
        assert set(s.values) == {sp.key for sp in met.SPECS_BY_ROLE["side"]}
        assert set(f.values) == {sp.key for sp in met.SPECS_BY_ROLE["front"]}
