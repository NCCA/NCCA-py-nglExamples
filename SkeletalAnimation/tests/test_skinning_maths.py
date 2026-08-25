"""Headless tests for the LBS / DQS skinning maths.

These pin the numpy reference implementation that the GLSL vertex shader in
``SkeletalAnimation/shaders/SkinVertex.glsl`` mirrors line-for-line, so
run them whenever the shader changes -- if the CPU and GPU skinning ever
disagree, one of the two transcriptions has a bug.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from skinning_maths import (  # noqa: E402
    bind_pose,
    dlb_blend,
    dq_to_mat,
    dual_quat_from_mat,
    pose_matrices,
    quat_from_mat,
    skin_dqs,
    skin_lbs,
)

BONE_LENGTH = 0.5
N_BONES = 4


# ----------------------------------------------------------------------
# bind_pose / pose_matrices (forward kinematics)
# ----------------------------------------------------------------------
class TestBindPose:
    def test_bind_matrices_translate_along_chain(self):
        """Bind matrices place each bone's origin at i * bone_length up +y,
        with no rotation (row-vector convention: translation in row 3)."""
        bind, inv_bind = bind_pose(N_BONES, BONE_LENGTH)
        assert len(bind) == N_BONES
        assert len(inv_bind) == N_BONES
        for i, m in enumerate(bind):
            np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-6)
            np.testing.assert_allclose(m[3, :3], [0.0, i * BONE_LENGTH, 0.0], atol=1e-6)

    def test_inverse_bind_undoes_bind(self):
        bind, inv_bind = bind_pose(N_BONES, BONE_LENGTH)
        for m, inv in zip(bind, inv_bind):
            np.testing.assert_allclose(m @ inv, np.eye(4), atol=1e-5)


class TestPoseMatrices:
    def test_zero_angles_match_bind_pose(self):
        bind, _ = bind_pose(N_BONES, BONE_LENGTH)
        posed = pose_matrices([0.0] * N_BONES, BONE_LENGTH)
        for b, p in zip(bind, posed):
            np.testing.assert_allclose(b, p, atol=1e-6)

    def test_twist_does_not_move_bone_origins(self):
        """Bones rotate about their own +y axis (the chain axis), so twisting
        any bone changes orientation only -- the chain stays straight."""
        posed = pose_matrices([30.0, -60.0, 90.0, 180.0], BONE_LENGTH)
        for i, m in enumerate(posed):
            np.testing.assert_allclose(m[3, :3], [0.0, i * BONE_LENGTH, 0.0], atol=1e-5)

    def test_ninety_degree_end_bone_rotates_x_axis_into_z(self):
        """A single-joint sanity check: rotating bone 0 by 90 degrees about
        +y should carry the local +x axis onto -z (right-hand rule)."""
        posed = pose_matrices([90.0, 0.0, 0.0, 0.0], BONE_LENGTH)
        rotated_x_axis = posed[0][0, :3]
        np.testing.assert_allclose(rotated_x_axis, [0.0, 0.0, -1.0], atol=1e-5)


# ----------------------------------------------------------------------
# skin_lbs
# ----------------------------------------------------------------------
def _palette(pose, inv_bind):
    """v_skinned = v_bind @ inv_bind[j] @ pose[j] -- combine into one mat per bone."""
    return np.stack([inv_bind[j] @ pose[j] for j in range(len(pose))])


class TestSkinLBS:
    def test_identity_pose_equals_bind_mesh(self):
        bind, inv_bind = bind_pose(N_BONES, BONE_LENGTH)
        posed = pose_matrices([0.0] * N_BONES, BONE_LENGTH)
        palette = _palette(posed, inv_bind)

        verts = np.array([[0.1, 0.25, 0.0], [0.0, 0.75, 0.1]])
        normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        idx = np.array([[0, 1], [1, 2]], dtype=np.int32)
        wts = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        skinned_verts, skinned_normals = skin_lbs(verts, normals, idx, wts, palette)
        np.testing.assert_allclose(skinned_verts, verts, atol=1e-5)
        np.testing.assert_allclose(skinned_normals, normals, atol=1e-5)

    def test_single_weight_matches_rigid_transform(self):
        """weight (1, 0): the vertex must follow bone 0 exactly like a rigid
        body -- this is what an LBS palette degenerates to per-influence."""
        bind, inv_bind = bind_pose(N_BONES, BONE_LENGTH)
        posed = pose_matrices([45.0, 0.0, 0.0, 0.0], BONE_LENGTH)
        palette = _palette(posed, inv_bind)

        verts = np.array([[0.2, 0.1, 0.0]])
        normals = np.array([[0.0, 1.0, 0.0]])
        idx = np.array([[0, 1]], dtype=np.int32)
        wts = np.array([[1.0, 0.0]], dtype=np.float32)

        skinned_verts, _ = skin_lbs(verts, normals, idx, wts, palette)
        expected = np.append(verts[0], 1.0) @ palette[0]
        np.testing.assert_allclose(skinned_verts[0], expected[:3], atol=1e-5)


# ----------------------------------------------------------------------
# dual quaternion helpers
# ----------------------------------------------------------------------
class TestQuatFromMat:
    def test_identity(self):
        q = quat_from_mat(np.eye(4))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_roundtrip_via_pose_matrix(self):
        """Extracting a quaternion from a pose matrix and rebuilding a dual
        quaternion's rotation should reproduce the same rotation matrix."""
        posed = pose_matrices([37.0], BONE_LENGTH)
        q = quat_from_mat(posed[0])
        qd = dual_quat_from_mat(posed[0])[1]
        rebuilt = dq_to_mat(q, qd)
        np.testing.assert_allclose(rebuilt[:3, :3], posed[0][:3, :3], atol=1e-4)


class TestDualQuatFromMat:
    def test_translation_recovered(self):
        m = np.eye(4)
        m[3, :3] = [1.0, 2.0, 3.0]
        qr, qd = dual_quat_from_mat(m)
        rebuilt = dq_to_mat(qr, qd)
        np.testing.assert_allclose(rebuilt[3, :3], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(rebuilt[:3, :3], np.eye(3), atol=1e-6)


class TestDLBBlend:
    def test_hemisphere_alignment(self):
        """Two dual quaternions representing the *same* rotation but with
        opposite quaternion sign (q and -q encode identical rotations) must
        still blend to that rotation, not cancel to zero."""
        m = pose_matrices([60.0], BONE_LENGTH)[0]
        qr, qd = dual_quat_from_mat(m)
        qr_neg, qd_neg = -qr, -qd

        blended_qr, blended_qd = dlb_blend([(qr, qd), (qr_neg, qd_neg)], [0.5, 0.5])
        rebuilt = dq_to_mat(blended_qr, blended_qd)
        np.testing.assert_allclose(rebuilt[:3, :3], m[:3, :3], atol=1e-4)


# ----------------------------------------------------------------------
# skin_dqs
# ----------------------------------------------------------------------
class TestSkinDQS:
    def test_identity_pose_equals_bind_mesh(self):
        bind, inv_bind = bind_pose(N_BONES, BONE_LENGTH)
        posed = pose_matrices([0.0] * N_BONES, BONE_LENGTH)
        palette = _palette(posed, inv_bind)
        dq_palette = [dual_quat_from_mat(m) for m in palette]

        verts = np.array([[0.1, 0.25, 0.0], [0.0, 0.75, 0.1]])
        normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        idx = np.array([[0, 1], [1, 2]], dtype=np.int32)
        wts = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        skinned_verts, skinned_normals = skin_dqs(verts, normals, idx, wts, dq_palette)
        np.testing.assert_allclose(skinned_verts, verts, atol=1e-5)
        np.testing.assert_allclose(skinned_normals, normals, atol=1e-5)


# ----------------------------------------------------------------------
# The candy-wrapper case -- the whole point of the demo
# ----------------------------------------------------------------------
class TestCandyWrapper:
    """A 180 degree twist of the end bone, 50/50 weighted vertices sampled
    around a ring straddling the joint. LBS blends the two bone matrices'
    *linear* transforms, which for a 180 degree relative twist rotates each
    influence's contribution to nearly opposite points on the ring and they
    (partially) cancel in the linear average, collapsing the ring radius
    towards the chain axis. DQS blends the shortest-path rotation instead,
    so the ring keeps its radius all the way through the twist. This
    asymmetry (Kavan et al. 2007, "Skinning with Dual Quaternions") is the
    artefact the whole demo exists to show.
    """

    @staticmethod
    def _ring_of_vertices(radius, y, n=16):
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack(
            [radius * np.cos(angles), np.full(n, y), radius * np.sin(angles)], axis=1
        )

    def test_lbs_collapses_radius_dqs_preserves_it(self):
        n_bones = 2
        bone_length = 1.0
        bind, inv_bind = bind_pose(n_bones, bone_length)
        # 180 degree twist of the end (last) bone only
        posed = pose_matrices([0.0, 180.0], bone_length)
        palette = _palette(posed, inv_bind)
        dq_palette = [dual_quat_from_mat(m) for m in palette]

        radius = 0.3
        joint_y = bone_length  # the joint straddled by the twisting bone
        verts = self._ring_of_vertices(radius, joint_y)
        normals = np.tile([1.0, 0.0, 0.0], (len(verts), 1))
        idx = np.tile([0, 1], (len(verts), 1)).astype(np.int32)
        wts = np.tile([0.5, 0.5], (len(verts), 1)).astype(np.float32)

        lbs_verts, _ = skin_lbs(verts, normals, idx, wts, palette)
        dqs_verts, _ = skin_dqs(verts, normals, idx, wts, dq_palette)

        axis_point = np.array([0.0, joint_y, 0.0])
        lbs_radius = np.linalg.norm(lbs_verts - axis_point, axis=1).mean()
        dqs_radius = np.linalg.norm(dqs_verts - axis_point, axis=1).mean()

        # LBS: a 180 degree relative twist averages each vertex with its own
        # antipode, collapsing the ring to (near) zero radius.
        assert lbs_radius == pytest.approx(0.0, abs=1e-4)
        # DQS: the ring keeps its original radius throughout the twist.
        assert dqs_radius == pytest.approx(radius, abs=1e-4)
        # pin the asymmetry itself, not just the individual values
        assert dqs_radius - lbs_radius > radius * 0.9
