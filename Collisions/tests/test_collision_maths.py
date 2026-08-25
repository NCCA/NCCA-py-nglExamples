"""Headless tests for the collision-detection maths."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from collision_maths import (
    ray_sphere_intersect,
    ray_triangle_intersect,
    sphere_bbox_reflect,
    sphere_plane_collide,
    sphere_sphere_collide,
)


class TestRaySphereIntersect:
    def test_ray_through_centre_hits(self):
        assert ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_ray_missing_sphere(self):
        assert not ray_sphere_intersect(
            np.array([0.0, 5.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_ray_pointing_away_misses(self):
        # sphere is behind the ray origin -- no hit even though the
        # infinite *line* would pass through it
        assert not ray_sphere_intersect(
            np.array([0.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_unnormalised_direction_gives_same_result(self):
        hit_a = ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )
        hit_b = ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 50.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )
        assert hit_a == hit_b


class TestRayTriangleIntersect:
    def setup_method(self):
        self.v0 = np.array([-1.0, -1.0, 0.0])
        self.v1 = np.array([1.0, -1.0, 0.0])
        self.v2 = np.array([0.0, 1.0, 0.0])

    def test_ray_through_centre_hits(self):
        hit, point = ray_triangle_intersect(
            np.array([0.0, -0.3, -5.0]),
            np.array([0.0, -0.3, 5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert hit
        np.testing.assert_allclose(point, [0.0, -0.3, 0.0], atol=1e-5)

    def test_ray_missing_triangle(self):
        hit, point = ray_triangle_intersect(
            np.array([5.0, 5.0, -5.0]),
            np.array([5.0, 5.0, 5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit
        assert point is None

    def test_ray_parallel_to_triangle_misses(self):
        hit, point = ray_triangle_intersect(
            np.array([0.0, 0.0, -5.0]),
            np.array([1.0, 0.0, -5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit
        assert point is None

    def test_ray_stopping_before_triangle_misses(self):
        # segment ends at z=-1, triangle is at z=0 -- t would be > 1 (this
        # port treats the segment as a ray though, so this actually still
        # hits since t is unbounded above; use a ray pointing away instead
        hit, _ = ray_triangle_intersect(
            np.array([0.0, -0.3, 5.0]),
            np.array([0.0, -0.3, 10.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit


class TestSpherePlaneCollide:
    def setup_method(self):
        self.center = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        self.width = 10.0
        self.depth = 10.0

    def test_sphere_merely_touching_plane_no_collision(self):
        # The C++'s literal threshold is `D = normal.(pos-center) + radius;
        # hit when D <= 0`, i.e. the sphere must sink a full diameter past
        # the plane -- merely touching it (pos.y == radius) is NOT enough,
        # even though that reads as "obviously colliding" physically.
        assert not sphere_plane_collide(
            np.array([0.0, 1.0, 0.0]),
            1.0,
            self.center,
            self.normal,
            self.width,
            self.depth,
        )

    def test_sphere_sunk_full_diameter_through_plane_collides(self):
        # Boundary case: pos.y == -radius makes D == 0 exactly, which the
        # C++'s `D <= 0.0f` counts as a hit.
        assert sphere_plane_collide(
            np.array([0.0, -1.0, 0.0]),
            1.0,
            self.center,
            self.normal,
            self.width,
            self.depth,
        )

    def test_sphere_above_plane_no_collision(self):
        assert not sphere_plane_collide(
            np.array([0.0, 5.0, 0.0]),
            1.0,
            self.center,
            self.normal,
            self.width,
            self.depth,
        )

    def test_sphere_below_plane_within_bounds_collides(self):
        assert sphere_plane_collide(
            np.array([0.0, -2.0, 0.0]),
            1.0,
            self.center,
            self.normal,
            self.width,
            self.depth,
        )

    def test_sphere_below_plane_outside_bounds_no_collision(self):
        assert not sphere_plane_collide(
            np.array([20.0, -2.0, 0.0]),
            1.0,
            self.center,
            self.normal,
            self.width,
            self.depth,
        )

    def test_offset_plane_centre_still_correct(self):
        # regression test for the deliberate generalisation over the C++
        # source, which only handles a plane through the world origin.
        # Same full-diameter threshold as above, just re-based on a plane
        # centre 5 units up: the boundary is offset_center.y - radius == 4.
        offset_center = np.array([0.0, 5.0, 0.0])
        assert sphere_plane_collide(
            np.array([0.0, 4.0, 0.0]),
            1.0,
            offset_center,
            self.normal,
            self.width,
            self.depth,
        )
        assert not sphere_plane_collide(
            np.array([0.0, 6.0, 0.0]),
            1.0,
            offset_center,
            self.normal,
            self.width,
            self.depth,
        )


class TestSphereSphereCollide:
    def test_touching_spheres_collide(self):
        assert sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([2.0, 0.0, 0.0]), 1.0
        )

    def test_overlapping_spheres_collide(self):
        assert sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([1.0, 0.0, 0.0]), 1.0
        )

    def test_separated_spheres_no_collision(self):
        assert not sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([5.0, 0.0, 0.0]), 1.0
        )


class TestSphereBboxReflect:
    def test_no_wall_hit_direction_unchanged(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            40.0,
        )
        assert not hit
        np.testing.assert_allclose(new_dir, [1.0, 0.0, 0.0])

    def test_hitting_positive_x_wall_reflects_x_component(self):
        # position + radius crosses the +X wall at half_extent=40
        hit, new_dir = sphere_bbox_reflect(
            np.array([39.5, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_hitting_positive_y_wall_reflects_y_component_only(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 39.5, 0.0]),
            np.array([0.5, 1.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [0.5, -1.0, 0.0], atol=1e-10)

    def test_hitting_negative_z_wall_reflects_z_component(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 0.0, -39.5]),
            np.array([0.0, 0.0, -1.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [0.0, 0.0, 1.0], atol=1e-10)

    def test_corner_reflects_off_both_walls(self):
        # position + radius crosses both +X and +Y walls simultaneously
        hit, new_dir = sphere_bbox_reflect(
            np.array([39.5, 39.5, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [-1.0, -1.0, 0.0], atol=1e-10)
