"""Headless tests for GimbalLock's rotation maths.

These pin down the one thing the whole demo exists to show: driving three
Euler angles through a nested rotate_x/rotate_y/rotate_z stack loses a
degree of freedom at ry = +/-90 (gimbal lock), while the quaternion path
built from the same angles keeps producing the correct, non-degenerate
rotation throughout.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rotation_maths import (  # noqa: E402
    euler_to_mat,
    euler_to_quat,
    is_gimbal_locked,
    lost_dof_axis,
    quat_to_mat,
)


# ----------------------------------------------------------------------
# is_gimbal_locked
# ----------------------------------------------------------------------
class TestIsGimbalLocked:
    def test_locked_at_positive_90(self):
        assert is_gimbal_locked(90.0)

    def test_locked_at_negative_90(self):
        assert is_gimbal_locked(-90.0)

    def test_locked_at_270_is_equivalent_to_negative_90(self):
        assert is_gimbal_locked(270.0)

    def test_not_locked_at_zero(self):
        assert not is_gimbal_locked(0.0)

    def test_not_locked_at_45(self):
        assert not is_gimbal_locked(45.0)


# ----------------------------------------------------------------------
# lost_dof_axis
# ----------------------------------------------------------------------
class TestLostDofAxis:
    def test_none_when_not_locked(self):
        assert lost_dof_axis(10.0, 30.0, 20.0) is None

    def test_reports_axis_collapse_when_locked(self):
        axis = lost_dof_axis(10.0, 90.0, 20.0)
        assert axis is not None
        assert "x" in axis.lower()
        assert "z" in axis.lower()


# ----------------------------------------------------------------------
# euler_to_mat / euler_to_quat round trip (non-degenerate angles)
# ----------------------------------------------------------------------
class TestEulerRoundTrip:
    @pytest.mark.parametrize(
        "rx, ry, rz",
        [
            (0.0, 0.0, 0.0),
            (30.0, 20.0, 10.0),
            (-45.0, 60.0, 15.0),
            (10.0, -30.0, 200.0),
        ],
    )
    def test_quaternion_path_matches_euler_matrix(self, rx, ry, rz):
        """The Mat4-composed quaternion and the plain-numpy Euler matrix are
        two independent implementations of the same rotation -- for any
        non-degenerate pose they must agree."""
        m_euler = euler_to_mat(rx, ry, rz)
        m_quat = quat_to_mat(euler_to_quat(rx, ry, rz))
        np.testing.assert_allclose(m_euler, m_quat, atol=1e-4)


# ----------------------------------------------------------------------
# The gimbal-lock DOF collapse itself
# ----------------------------------------------------------------------
class TestGimbalLockCollapse:
    def test_rx_and_rz_collapse_to_one_dof_at_ry_90(self):
        """At ry=90 the rig has lost a degree of freedom: rotating rx and rz
        oppositely (rx - rz held constant) leaves the resulting orientation
        unchanged, because roll and yaw now spin about the same world axis."""
        base = euler_to_mat(10.0, 90.0, 20.0)
        shifted = euler_to_mat(10.0 + 5.0, 90.0, 20.0 + 5.0)
        np.testing.assert_allclose(base, shifted, atol=1e-5)

    def test_different_rx_minus_rz_gives_a_different_orientation(self):
        """Sanity check on the above: it is specifically (rx - rz) that is
        degenerate, not any pair of angles whatsoever."""
        base = euler_to_mat(10.0, 90.0, 20.0)
        different = euler_to_mat(10.0 + 5.0, 90.0, 20.0)
        assert not np.allclose(base, different, atol=1e-5)

    def test_quaternion_path_still_well_defined_at_lock(self):
        """The quaternion built from the same locked pose is a perfectly
        ordinary unit quaternion -- no singularity, no NaNs -- it just
        happens to represent the same (collapsed) orientation as the
        Euler matrix."""
        q = euler_to_quat(10.0, 90.0, 20.0)
        m = quat_to_mat(q)
        assert np.all(np.isfinite(m))
        np.testing.assert_allclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-4)


# ----------------------------------------------------------------------
# euler_to_mat basics
# ----------------------------------------------------------------------
class TestEulerToMat:
    def test_identity_at_zero(self):
        np.testing.assert_allclose(euler_to_mat(0.0, 0.0, 0.0), np.eye(4), atol=1e-6)

    def test_result_is_a_valid_rotation(self):
        m = euler_to_mat(33.0, -12.0, 77.0)
        r = m[:3, :3]
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-5)
