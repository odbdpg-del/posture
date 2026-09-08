"""Geometry tests: angle conventions, aspect correction, degenerate inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posture import geometry as geo


class TestMetricFrame:
    def test_flips_y_and_scales_x(self):
        out = geo.to_metric_frame(np.array([0.25, 0.75]), aspect=2.0)
        assert out == pytest.approx([0.5, -0.75])

    def test_works_on_arrays(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = geo.to_metric_frame(pts, aspect=1.5)
        assert out.shape == (2, 2)
        assert out.ravel() == pytest.approx([0.0, 0.0, 1.5, -1.0])

    def test_corrects_aspect_distortion(self):
        """A physically 45-degree line reads as 45 degrees on any frame shape.

        This is the failure mode the metric frame exists to prevent: with
        separate normalization by width and height, a 45-degree line on a 16:9
        frame has normalized slope 16/9, and naive trigonometry calls it 61
        degrees.
        """
        for aspect in (4 / 3, 16 / 9, 1.0):
            # A line at 45 degrees in real space: equal pixel run and rise.
            # In normalized coordinates the run shrinks by the aspect ratio.
            v_normalized = np.array([0.1 / aspect, -0.1])
            v = geo.to_metric_frame(v_normalized + 0.5, aspect) - geo.to_metric_frame(
                np.array([0.5, 0.5]), aspect
            )
            assert geo.signed_angle_from_horizontal(v) == pytest.approx(45.0)

        # And confirm the uncorrected version really is wrong, so this test
        # would notice if to_metric_frame quietly became a no-op.
        naive = math.degrees(math.atan2(0.1, 0.1 / (16 / 9)))
        assert naive == pytest.approx(60.6, abs=0.2)


class TestAngleAt:
    def test_collinear_is_180(self):
        assert geo.angle_at([0, 2], [0, 1], [0, 0]) == pytest.approx(180.0)

    def test_right_angle(self):
        assert geo.angle_at([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0)

    def test_is_unsigned(self):
        """Mirroring the outer points cannot change an interior angle."""
        left = geo.angle_at([-1, 1], [0, 0], [0, -1])
        right = geo.angle_at([1, 1], [0, 0], [0, -1])
        assert left == pytest.approx(right)

    def test_degenerate_vertex_returns_none(self):
        assert geo.angle_at([0, 0], [0, 0], [1, 0]) is None


class TestSignedAngles:
    @pytest.mark.parametrize("vector,expected", [
        ((0, 1), 0.0),
        ((1, 0), 90.0),
        ((-1, 0), -90.0),
        ((1, 1), 45.0),
        ((-1, 1), -45.0),
        ((0, -1), 180.0),
    ])
    def test_from_vertical(self, vector, expected):
        assert geo.signed_angle_from_vertical(np.array(vector)) == pytest.approx(expected)

    @pytest.mark.parametrize("vector,expected", [
        ((1, 0), 0.0),
        ((0, 1), 90.0),
        ((0, -1), -90.0),
        ((1, 1), 45.0),
    ])
    def test_from_horizontal(self, vector, expected):
        assert geo.signed_angle_from_horizontal(np.array(vector)) == pytest.approx(expected)

    def test_zero_vector_returns_none(self):
        assert geo.signed_angle_from_vertical(np.zeros(2)) is None
        assert geo.signed_angle_from_horizontal(np.zeros(2)) is None
        assert geo.unit(np.zeros(2)) is None


class TestHelpers:
    def test_angle_between_clamps_rounding(self):
        """Parallel unit vectors must not trip arccos out of its domain."""
        v = np.array([0.3, 0.4])
        assert geo.angle_between(v, v * 3) == pytest.approx(0.0, abs=1e-6)
        assert geo.angle_between(v, -v) == pytest.approx(180.0)

    def test_midpoint(self):
        assert geo.midpoint([0, 0], [2, 4]) == pytest.approx([1.0, 2.0])

    def test_norm(self):
        assert geo.norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
