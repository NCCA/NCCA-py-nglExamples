"""Headless tests for the GeometryTessellation grid/LOD maths (numpy-only)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tess_grid import (  # noqa: E402
    build_patch_grid,
    patch_count,
    tess_level_from_distance,
)


class TestBuildPatchGrid:
    def test_vertex_count_is_four_per_patch(self):
        verts = build_patch_grid(resolution=4, size=8.0)
        assert verts.shape == (4 * 4 * 4, 3)

    def test_grid_is_centred_on_origin(self):
        verts = build_patch_grid(resolution=8, size=10.0)
        assert verts[:, 0].min() == pytest.approx(-5.0)
        assert verts[:, 0].max() == pytest.approx(5.0)
        assert verts[:, 2].min() == pytest.approx(-5.0)
        assert verts[:, 2].max() == pytest.approx(5.0)

    def test_all_control_points_start_flat(self):
        """y must be 0 everywhere -- displacement happens entirely in the
        tessellation evaluation shader, not on the CPU-side control mesh."""
        verts = build_patch_grid(resolution=6, size=6.0)
        np.testing.assert_allclose(verts[:, 1], 0.0)

    def test_patch_corners_are_a_counter_clockwise_quad(self):
        """First patch (bottom-left of the grid) should wind
        (x0,z0)->(x1,z0)->(x1,z1)->(x0,z0) i.e. a simple quad loop, so the
        TES's bilinear interpolation of gl_in[0..3] lines up with a
        continuous surface rather than a bowtie."""
        verts = build_patch_grid(resolution=2, size=4.0)
        p0, p1, p2, p3 = verts[0], verts[1], verts[2], verts[3]
        assert p0[0] == p3[0]  # left edge shares x
        assert p1[0] == p2[0]  # right edge shares x
        assert p0[2] == p1[2]  # bottom edge shares z
        assert p2[2] == p3[2]  # top edge shares z

    def test_rejects_non_positive_resolution(self):
        with pytest.raises(ValueError):
            build_patch_grid(resolution=0, size=4.0)


class TestPatchCount:
    def test_matches_resolution_squared(self):
        assert patch_count(16) == 256
        assert patch_count(1) == 1


class TestTessLevelFromDistance:
    def test_near_clamps_to_max_level(self):
        level = tess_level_from_distance(
            distance=0.0, near_distance=2.0, far_distance=20.0
        )
        assert level == pytest.approx(64.0)

    def test_far_clamps_to_min_level(self):
        level = tess_level_from_distance(
            distance=100.0, near_distance=2.0, far_distance=20.0
        )
        assert level == pytest.approx(1.0)

    def test_midpoint_is_between_bounds(self):
        level = tess_level_from_distance(
            distance=11.0, near_distance=2.0, far_distance=20.0
        )
        assert 1.0 < level < 64.0

    def test_monotonically_decreasing_with_distance(self):
        levels = [
            tess_level_from_distance(d, near_distance=2.0, far_distance=20.0)
            for d in (2.0, 6.0, 10.0, 14.0, 18.0, 22.0)
        ]
        assert all(a >= b for a, b in zip(levels, levels[1:]))

    def test_rejects_bad_distance_bounds(self):
        with pytest.raises(ValueError):
            tess_level_from_distance(5.0, near_distance=10.0, far_distance=5.0)
