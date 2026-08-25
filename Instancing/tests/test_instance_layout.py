"""Headless tests for the instancing demo's numpy-only layout maths."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from instance_layout import cube, golden_spiral, grid

RECORD_FLOATS = 8  # offset.xyz, scale, colour.rgba


# ----------------------------------------------------------------------
# cube
# ----------------------------------------------------------------------
class TestCube:
    def test_shape_is_36_verts_of_pos_and_normal(self):
        data = cube()
        assert data.shape == (36 * 6,)

    def test_dtype_is_float32(self):
        assert cube().dtype == np.float32

    def test_size_scales_extent(self):
        small = cube(1.0).reshape(-1, 6)
        big = cube(2.0).reshape(-1, 6)
        assert np.abs(big[:, :3]).max() == pytest.approx(
            2.0 * np.abs(small[:, :3]).max()
        )

    def test_normals_are_unit_length(self):
        data = cube().reshape(-1, 6)
        lengths = np.linalg.norm(data[:, 3:6], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# golden_spiral
# ----------------------------------------------------------------------
class TestGoldenSpiral:
    def test_shape_and_dtype(self):
        data = golden_spiral(100, radius=5.0)
        assert data.shape == (100, RECORD_FLOATS)
        assert data.dtype == np.float32

    def test_radii_are_monotonically_non_decreasing(self):
        """Sunflower-style spiral: later instances sit further from centre."""
        data = golden_spiral(200, radius=10.0)
        radii = np.linalg.norm(data[:, [0, 2]], axis=1)
        assert np.all(np.diff(radii) >= -1e-6)

    def test_max_radius_respects_bound(self):
        data = golden_spiral(500, radius=8.0)
        radii = np.linalg.norm(data[:, [0, 2]], axis=1)
        assert radii.max() <= 8.0 + 1e-5

    def test_colours_in_unit_range(self):
        data = golden_spiral(300, radius=5.0)
        colours = data[:, 4:8]
        assert colours.min() >= 0.0
        assert colours.max() <= 1.0

    def test_scale_is_positive(self):
        data = golden_spiral(50, radius=5.0)
        assert np.all(data[:, 3] > 0.0)

    def test_single_instance_at_origin(self):
        data = golden_spiral(1, radius=5.0)
        np.testing.assert_allclose(data[0, [0, 2]], [0.0, 0.0], atol=1e-6)


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
class TestGrid:
    def test_shape_and_dtype(self):
        data = grid(64)
        assert data.shape == (64, RECORD_FLOATS)
        assert data.dtype == np.float32

    def test_centred_on_origin(self):
        data = grid(225, spacing=1.5)
        centre = data[:, :3].mean(axis=0)
        np.testing.assert_allclose(centre, [0.0, 0.0, 0.0], atol=1e-5)

    def test_colours_in_unit_range(self):
        data = grid(81)
        colours = data[:, 4:8]
        assert colours.min() >= 0.0
        assert colours.max() <= 1.0

    def test_non_square_count_still_covers_n_instances(self):
        data = grid(17)
        assert data.shape[0] == 17

    def test_spacing_sets_extent(self):
        wide = grid(9, spacing=3.0)
        narrow = grid(9, spacing=1.0)
        assert np.ptp(wide[:, :3]) > np.ptp(narrow[:, :3])
