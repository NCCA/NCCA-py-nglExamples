"""Headless tests for the ray-cast / screen-space picking maths."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from picking_maths import (  # noqa: E402
    bounding_sphere,
    intersect_sphere,
    intersect_triangles,
    point_polyline_distance,
    point_segment_distance,
    ray_from_screen,
    transform_ray,
)


# ----------------------------------------------------------------------
# ray_from_screen
# ----------------------------------------------------------------------
class TestRayFromScreen:
    def test_identity_mvp_centre_of_screen(self):
        """With an identity MVP the screen centre maps to NDC (0,0): the ray
        starts on the near plane at z=-1 and points down +z."""
        origin, direction = ray_from_screen(400, 300, 800, 600, np.eye(4))
        np.testing.assert_allclose(origin, [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-6)

    def test_identity_mvp_corner(self):
        """Top-left pixel maps to NDC (-1, +1)."""
        origin, _ = ray_from_screen(0, 0, 800, 600, np.eye(4))
        np.testing.assert_allclose(origin, [-1.0, 1.0, -1.0], atol=1e-6)

    def test_direction_is_normalised(self):
        # arbitrary invertible matrix
        rng = np.random.default_rng(1)
        mvp = np.eye(4) + 0.1 * rng.standard_normal((4, 4))
        _, direction = ray_from_screen(123, 456, 800, 600, mvp)
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# transform_ray
# ----------------------------------------------------------------------
class TestTransformRay:
    def test_translation_moves_origin_not_direction(self):
        # row-vector convention: translation lives in row 3
        matrix = np.eye(4)
        matrix[3, :3] = [5.0, -2.0, 1.0]
        origin, direction = transform_ray(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), matrix
        )
        np.testing.assert_allclose(origin, [6.0, -2.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-6)

    def test_scale_stretches_direction(self):
        """Direction is deliberately not re-normalised so t stays comparable."""
        matrix = np.diag([2.0, 2.0, 2.0, 1.0])
        _, direction = transform_ray(np.zeros(3), np.array([1.0, 0.0, 0.0]), matrix)
        np.testing.assert_allclose(direction, [2.0, 0.0, 0.0], atol=1e-6)


# ----------------------------------------------------------------------
# intersect_sphere
# ----------------------------------------------------------------------
class TestIntersectSphere:
    def test_hit_through_centre(self):
        assert intersect_sphere(
            np.array([0.0, 0.0, -5.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            1.0,
        )

    def test_miss(self):
        assert not intersect_sphere(
            np.array([0.0, 3.0, -5.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            1.0,
        )

    def test_sphere_behind_origin(self):
        assert not intersect_sphere(
            np.array([0.0, 0.0, 5.0]),
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            1.0,
        )

    def test_origin_inside_sphere(self):
        assert intersect_sphere(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), 1.0
        )

    def test_unnormalised_direction(self):
        assert intersect_sphere(
            np.array([0.0, 0.0, -5.0]),
            np.array([0.0, 0.0, 10.0]),
            np.zeros(3),
            1.0,
        )


# ----------------------------------------------------------------------
# intersect_triangles (Moller-Trumbore)
# ----------------------------------------------------------------------
class TestIntersectTriangles:
    # a unit-ish triangle in the z=0 plane around the origin
    TRI = np.array([[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])

    def test_direct_hit(self):
        t = intersect_triangles(
            np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]), self.TRI
        )
        assert t == pytest.approx(5.0, abs=1e-6)

    def test_double_sided(self):
        """Hit from behind too: picking must not backface-cull."""
        t = intersect_triangles(
            np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), self.TRI
        )
        assert t == pytest.approx(5.0, abs=1e-6)

    def test_miss_outside_triangle(self):
        t = intersect_triangles(
            np.array([2.0, 2.0, -5.0]), np.array([0.0, 0.0, 1.0]), self.TRI
        )
        assert t is None

    def test_triangle_behind_ray(self):
        t = intersect_triangles(
            np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]), self.TRI
        )
        assert t is None

    def test_parallel_ray(self):
        t = intersect_triangles(
            np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]), self.TRI
        )
        assert t is None

    def test_nearest_of_many(self):
        tris = np.concatenate(
            [self.TRI, self.TRI + np.array([0.0, 0.0, 3.0])]  # copy 3 behind
        )
        t = intersect_triangles(
            np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]), tris
        )
        assert t == pytest.approx(5.0, abs=1e-6)

    def test_degenerate_triangle_ignored(self):
        degenerate = np.zeros((1, 3, 3))
        t = intersect_triangles(
            np.array([0.0, 0.0, -5.0]),
            np.array([0.0, 0.0, 1.0]),
            np.concatenate([degenerate, self.TRI]),
        )
        assert t == pytest.approx(5.0, abs=1e-6)


# ----------------------------------------------------------------------
# bounding_sphere
# ----------------------------------------------------------------------
class TestBoundingSphere:
    def test_unit_cube(self):
        corners = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            dtype=np.float64,
        )
        centre, radius = bounding_sphere(corners)
        np.testing.assert_allclose(centre, [0.0, 0.0, 0.0], atol=1e-6)
        assert radius == pytest.approx(np.sqrt(3.0), abs=1e-5)

    def test_offset_points(self):
        centre, radius = bounding_sphere(np.array([[9.0, 0.0, 0.0], [11.0, 0.0, 0.0]]))
        np.testing.assert_allclose(centre, [10.0, 0.0, 0.0], atol=1e-6)
        assert radius == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# 2D distances
# ----------------------------------------------------------------------
class TestScreenDistances:
    def test_point_on_segment(self):
        assert point_segment_distance([5, 0], [0, 0], [10, 0]) == pytest.approx(0.0)

    def test_perpendicular_distance(self):
        assert point_segment_distance([5, 3], [0, 0], [10, 0]) == pytest.approx(3.0)

    def test_clamped_to_endpoint(self):
        assert point_segment_distance([13, 4], [0, 0], [10, 0]) == pytest.approx(5.0)

    def test_zero_length_segment(self):
        assert point_segment_distance([3, 4], [0, 0], [0, 0]) == pytest.approx(5.0)

    def test_polyline_closed_uses_wrap_segment(self):
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        # point next to the left edge, which only exists when closed
        assert point_polyline_distance([-2, 5], square, closed=True) == pytest.approx(
            2.0
        )
        # open polyline: nearest is the corner point (0,0) or (0,10)
        assert point_polyline_distance([-2, 5], square, closed=False) == pytest.approx(
            np.hypot(2, 5)
        )
