"""Headless tests for the ShadowMapping light-space maths (numpy-only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_maths import (  # noqa: E402
    light_space_matrix,
    look_at,
    ortho,
    project_to_shadow_uv,
)


class TestLookAt:
    def test_orthonormal_basis(self):
        """The 3x3 rotation part of a look_at matrix must be orthonormal."""
        m = look_at(
            np.array([0.0, 5.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0, 1, 0])
        )
        rot = m[:3, :3]
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-8)

    def test_eye_maps_to_origin(self):
        """Transforming the eye position itself should land at the view-space origin."""
        eye = np.array([2.0, 3.0, 4.0])
        m = look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        eye_h = np.array([*eye, 1.0])
        view_space = eye_h @ m
        np.testing.assert_allclose(view_space[:3], [0.0, 0.0, 0.0], atol=1e-8)


class TestOrtho:
    def test_centre_of_frustum_maps_to_ndc_origin(self):
        m = ortho(-5, 5, -5, 5, 0.1, 20.0)
        centre = np.array([0.0, 0.0, -(0.1 + 20.0) / 2.0, 1.0])
        ndc = centre @ m
        np.testing.assert_allclose(ndc[:2], [0.0, 0.0], atol=1e-8)

    def test_near_far_map_to_minus_one_plus_one(self):
        m = ortho(-5, 5, -5, 5, 0.1, 20.0)
        near_point = np.array([0.0, 0.0, -0.1, 1.0]) @ m
        far_point = np.array([0.0, 0.0, -20.0, 1.0]) @ m
        np.testing.assert_allclose(near_point[2], -1.0, atol=1e-8)
        np.testing.assert_allclose(far_point[2], 1.0, atol=1e-8)


class TestLightSpaceMatrix:
    def test_points_inside_frustum_project_into_unit_square(self):
        """A directional light orbiting above the origin with a generous
        ortho extent -- points near the origin (well inside the light's
        frustum) must project into [0, 1]^2, which is the classic
        shadow-mapping bug to catch (get the composition order wrong and
        this silently fails)."""
        light_pos = np.array([3.0, 6.0, 3.0])
        target = np.array([0.0, 0.0, 0.0])
        ls = light_space_matrix(
            light_pos, target, ortho_extents=8.0, near=0.1, far=30.0
        )

        for point in (
            (0.0, 0.0, 0.0),
            (1.0, 0.5, -1.0),
            (-1.5, 0.0, 1.5),
            (0.0, 2.0, 0.0),
        ):
            u, v, depth = project_to_shadow_uv(point, ls)
            assert 0.0 <= u <= 1.0
            assert 0.0 <= v <= 1.0
            assert 0.0 <= depth <= 1.0

    def test_point_outside_extent_falls_outside_unit_square(self):
        light_pos = np.array([3.0, 6.0, 3.0])
        target = np.array([0.0, 0.0, 0.0])
        ls = light_space_matrix(
            light_pos, target, ortho_extents=2.0, near=0.1, far=20.0
        )
        # well outside the +/-2 world-unit half-extent of the frustum
        u, v, _ = project_to_shadow_uv((10.0, 0.0, 10.0), ls)
        assert not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0)

    def test_matches_direct_view_then_project_composition(self):
        """Sanity check that light_space_matrix really is view-then-project
        (row-vector composition), not the reverse."""
        light_pos = np.array([2.0, 4.0, 2.0])
        target = np.array([0.0, 0.0, 0.0])
        view = look_at(light_pos, target, np.array([0.0, 1.0, 0.0]))
        proj = ortho(-5, 5, -5, 5, 0.1, 20.0)
        expected = view @ proj
        actual = light_space_matrix(
            light_pos, target, ortho_extents=5.0, near=0.1, far=20.0
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10)
