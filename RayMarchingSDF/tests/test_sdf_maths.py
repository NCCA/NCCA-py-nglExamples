"""Headless tests for the signed-distance-field maths used by both the GLSL
and WGSL ray marchers. These are numpy mirrors of the shader functions —
see sdf_maths.py for the mapping onto RayMarchFragment.glsl / RayMarch.wgsl.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_maths import (
    estimate_normal,
    scene,
    sd_box,
    sd_plane,
    sd_sphere,
    sd_torus,
    smooth_min,
)


# ----------------------------------------------------------------------
# sd_sphere
# ----------------------------------------------------------------------
class TestSdSphere:
    def test_point_on_surface_is_zero(self):
        p = np.array([1.0, 0.0, 0.0])
        assert sd_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0) == pytest.approx(0.0)

    def test_point_inside_is_negative(self):
        p = np.array([0.2, 0.0, 0.0])
        assert sd_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0) < 0.0

    def test_point_outside_is_positive(self):
        p = np.array([3.0, 0.0, 0.0])
        assert sd_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0) > 0.0

    def test_respects_centre_offset(self):
        p = np.array([5.0, 0.0, 0.0])
        centre = np.array([4.0, 0.0, 0.0])
        assert sd_sphere(p, centre, 1.0) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# sd_box
# ----------------------------------------------------------------------
class TestSdBox:
    def test_centre_is_negative(self):
        assert sd_box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])) < 0.0

    def test_face_centre_is_zero(self):
        p = np.array([1.0, 0.0, 0.0])
        assert sd_box(p, np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)

    def test_corner_outside_is_positive(self):
        p = np.array([2.0, 2.0, 2.0])
        assert sd_box(p, np.array([1.0, 1.0, 1.0])) > 0.0


# ----------------------------------------------------------------------
# sd_torus
# ----------------------------------------------------------------------
class TestSdTorus:
    def test_on_ring_centreline_is_zero(self):
        # major radius 1, minor radius 0.25: a point on the ring centreline
        # sits exactly minor_radius away from the tube surface at 0.
        p = np.array([1.0, 0.0, 0.0])
        assert sd_torus(p, 1.0, 0.25) == pytest.approx(-0.25)

    def test_on_tube_surface_is_zero(self):
        p = np.array([1.25, 0.0, 0.0])
        assert sd_torus(p, 1.0, 0.25) == pytest.approx(0.0, abs=1e-6)

    def test_far_away_is_positive(self):
        p = np.array([10.0, 0.0, 0.0])
        assert sd_torus(p, 1.0, 0.25) > 0.0


# ----------------------------------------------------------------------
# sd_plane
# ----------------------------------------------------------------------
class TestSdPlane:
    def test_point_on_plane_is_zero(self):
        p = np.array([3.0, 0.0, -2.0])
        assert sd_plane(p, np.array([0.0, 1.0, 0.0]), 0.0) == pytest.approx(0.0)

    def test_point_above_plane_is_positive(self):
        p = np.array([0.0, 2.0, 0.0])
        assert sd_plane(p, np.array([0.0, 1.0, 0.0]), 0.0) == pytest.approx(2.0)

    def test_point_below_plane_is_negative(self):
        p = np.array([0.0, -1.0, 0.0])
        assert sd_plane(p, np.array([0.0, 1.0, 0.0]), 0.0) == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# smooth_min
# ----------------------------------------------------------------------
class TestSmoothMin:
    def test_never_exceeds_hard_min(self):
        a, k = 1.0, 0.3
        for b in np.linspace(-2.0, 3.0, 20):
            assert smooth_min(a, b, k) <= min(a, b) + 1e-9

    def test_converges_to_hard_min_when_far_apart(self):
        # once |a - b| >> k the blend has no effect
        assert smooth_min(0.0, 10.0, 0.1) == pytest.approx(0.0, abs=1e-3)
        assert smooth_min(10.0, 0.0, 0.1) == pytest.approx(0.0, abs=1e-3)

    def test_is_symmetric(self):
        assert smooth_min(1.0, 2.0, 0.5) == pytest.approx(smooth_min(2.0, 1.0, 0.5))

    def test_zero_k_matches_hard_min(self):
        assert smooth_min(1.0, 2.0, 0.0) == pytest.approx(min(1.0, 2.0))


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
class TestScene:
    def test_returns_finite_distance(self):
        d = scene(np.array([0.0, 0.5, 0.0]), time=0.0, k=0.3)
        assert np.isfinite(d)

    def test_time_changes_the_scene(self):
        # sit right on the moving sphere's orbit path (radius 1.6, height
        # 1.1) so the distance field there is dominated by its position.
        p = np.array([1.6, 1.1, 0.0])
        d0 = scene(p, time=0.0, k=0.3)
        d1 = scene(p, time=1.5, k=0.3)
        # the moving sphere melts through the rest of the scene over time,
        # so the distance field at a fixed point should change.
        assert d0 != pytest.approx(d1)


# ----------------------------------------------------------------------
# estimate_normal
# ----------------------------------------------------------------------
class TestEstimateNormal:
    def test_sphere_normal_is_radial(self):
        def sphere_only(p):
            return sd_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0)

        p = np.array([1.0, 0.0, 0.0]) / np.linalg.norm([1.0, 0.0, 0.0])
        n = estimate_normal(sphere_only, p)
        np.testing.assert_allclose(n, p, atol=1e-3)

    def test_normal_is_unit_length(self):
        def sphere_only(p):
            return sd_sphere(p, np.array([0.0, 0.0, 0.0]), 1.0)

        p = np.array([0.0, 1.0, 0.0])
        n = estimate_normal(sphere_only, p)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-3)
