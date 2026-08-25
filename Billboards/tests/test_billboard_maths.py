"""Headless tests for the camera-facing basis maths in billboard_maths.py.

These pin down the one thing this demo is teaching: how to pull a
"face the camera" right/up frame out of a view matrix. Get the row/column
convention backwards and every billboard renders edge-on, so each test
asserts a real geometric property (orthonormality, which way the implied
quad normal points) rather than "it ran without crashing".

The view matrices under test are built with ``ncca.ngl.look_at``, which
follows the repo's row-vector convention: a point transforms as
``row_vector @ matrix`` (see RayPickingSelection/picking_maths.py and the
design spec's maths-conventions section). Column 0 of such a matrix is the
camera's world-space right axis, column 1 is its world-space up axis, and
column 2 is its world-space *backward* axis (pointing from the look-at
target back towards the eye) -- equivalently, rows 0..2 of view^T.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from billboard_maths import (
    back_to_front,
    cylindrical_basis,
    spherical_basis,
)
from ncca.ngl import Vec3, look_at


def _view(eye, look=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> np.ndarray:
    return look_at(Vec3(*eye), Vec3(*look), Vec3(*up)).to_numpy().astype(np.float64)


def _random_eyes(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random eye positions on a sphere, azimuth/elevation both varied,
    avoiding the poles (straight up/down) where look_at's own up-vector
    handling gets ambiguous for *any* implementation."""
    azimuth = rng.uniform(0.0, 2.0 * np.pi, n)
    elevation = rng.uniform(-1.2, 1.2, n)  # radians, keeps away from poles
    radius = rng.uniform(3.0, 12.0, n)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.sin(azimuth)
    return np.stack([x, y, z], axis=1)


# ----------------------------------------------------------------------
# spherical_basis
# ----------------------------------------------------------------------
class TestSphericalBasis:
    def test_orthonormal_for_random_views(self):
        rng = np.random.default_rng(0)
        for eye in _random_eyes(rng, 30):
            view = _view(eye)
            right, up = spherical_basis(view)
            assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-5)
            assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-5)
            assert np.dot(right, up) == pytest.approx(0.0, abs=1e-5)

    def test_camera_facing_for_random_views(self):
        """The billboard's implied normal (right x up) must point from the
        billboard toward the camera eye -- i.e. have a positive dot product
        with (eye - billboard_position) -- for the quad to actually face
        the viewer rather than away from them."""
        rng = np.random.default_rng(1)
        billboard_pos = np.array([0.0, 0.0, 0.0])
        for eye in _random_eyes(rng, 30):
            view = _view(eye)
            right, up = spherical_basis(view)
            normal = np.cross(right, up)
            to_camera = np.asarray(eye) - billboard_pos
            to_camera /= np.linalg.norm(to_camera)
            assert np.dot(normal, to_camera) == pytest.approx(1.0, abs=1e-4), (
                "billboard normal should point directly at the camera for a "
                "spherical (always-face-camera) billboard"
            )

    def test_matches_view_matrix_columns(self):
        """Pins the extraction convention itself: right/up must be exactly
        columns 0/1 of the view matrix (== rows 0/1 of view^T), per the
        design spec's row-vector convention."""
        view = _view((3.0, 2.0, 5.0))
        right, up = spherical_basis(view)
        np.testing.assert_allclose(right, view[:3, 0], atol=1e-5)
        np.testing.assert_allclose(up, view[:3, 1], atol=1e-5)


# ----------------------------------------------------------------------
# cylindrical_basis
# ----------------------------------------------------------------------
class TestCylindricalBasis:
    def test_up_is_exactly_world_y(self):
        rng = np.random.default_rng(2)
        for eye in _random_eyes(rng, 20):
            view = _view(eye)
            _, up = cylindrical_basis(view)
            np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-6)

    def test_right_is_orthonormal_to_up(self):
        rng = np.random.default_rng(3)
        for eye in _random_eyes(rng, 20):
            view = _view(eye)
            right, up = cylindrical_basis(view)
            assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-5)
            assert np.dot(right, up) == pytest.approx(0.0, abs=1e-5)

    def test_faces_camera_when_eye_is_level(self):
        """With the camera at the same height as the target, a cylindrical
        billboard (up locked, only yaw follows the camera) should still
        face it exactly, same as the spherical case."""
        rng = np.random.default_rng(4)
        billboard_pos = np.array([0.0, 0.0, 0.0])
        azimuths = rng.uniform(0.0, 2.0 * np.pi, 20)
        for az in azimuths:
            eye = (6.0 * np.cos(az), 0.0, 6.0 * np.sin(az))
            view = _view(eye)
            right, up = cylindrical_basis(view)
            normal = np.cross(right, up)
            to_camera = np.asarray(eye) - billboard_pos
            to_camera /= np.linalg.norm(to_camera)
            assert np.dot(normal, to_camera) == pytest.approx(1.0, abs=1e-4)

    def test_degenerate_straight_down_view_has_no_nan(self):
        """Looking straight down means the camera's backward axis is
        parallel to world +y, so cross(world_up, backward) -- the natural
        way to build a horizontal 'right' vector -- collapses to zero.
        cylindrical_basis must fall back to a fixed right axis rather than
        dividing by zero: this is a documented degenerate case (see the
        docstring/comment in billboard_maths.py), not a crash.
        """
        view = _view((0.0, 8.0, 0.0), look=(0.0, 0.0, 0.0), up=(0.0, 0.0, -1.0))
        right, up = cylindrical_basis(view)
        assert not np.isnan(right).any()
        assert not np.isnan(up).any()
        assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-6)


# ----------------------------------------------------------------------
# back_to_front
# ----------------------------------------------------------------------
class TestBackToFront:
    def test_orders_furthest_first(self):
        view = _view((0.0, 0.0, 10.0))
        positions = np.array(
            [
                [0.0, 0.0, 3.0],  # closest to the eye
                [0.0, 0.0, -3.0],  # furthest from the eye
                [0.0, 0.0, 0.0],  # middle
            ]
        )
        order = back_to_front(positions, view)
        assert order == [1, 2, 0]

    def test_stable_for_single_billboard(self):
        view = _view((0.0, 0.0, 10.0))
        order = back_to_front(np.array([[0.0, 0.0, 0.0]]), view)
        assert order == [0]
