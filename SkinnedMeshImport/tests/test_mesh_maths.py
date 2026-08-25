"""Headless tests for mesh.py's skinning maths and impasse-bug workarounds.

These load the real guard model through impasse (no GL/Qt involved) and
pin down the two impasse struct bugs documented in mesh.py, plus the
matrix-convention maths that composes bone transforms from it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from ncca.ngl import Mat4, Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh import (
    SkinnedMesh,
    _bracketing_keys,
    _interpolate_rotation,
    _interpolate_vector,
    _Key,
    _read_rotation_keys,
    ai_matrix_to_mat4,
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "guard" / "boblampclean.md5mesh"
)


@pytest.fixture(scope="module")
def guard_mesh() -> SkinnedMesh:
    return SkinnedMesh(str(MODEL_PATH))


def test_ai_matrix_to_mat4_moves_translation_from_column_to_row():
    # assimp convention: translation in the last column (rows 0-2, col 3).
    assimp_style = np.eye(4, dtype=np.float32)
    assimp_style[0, 3] = 1.0
    assimp_style[1, 3] = 2.0
    assimp_style[2, 3] = 3.0

    result = ai_matrix_to_mat4(assimp_style)

    assert result.to_numpy()[3, :3] == pytest.approx([1.0, 2.0, 3.0])


def test_bind_pose_round_trip_is_identity(guard_mesh: SkinnedMesh):
    """offset, current-pose and the root's global inverse should cancel out
    at the bind pose -- this is exactly the invariant that catches the
    hierarchy-composition-order bug this demo hit during development."""

    def bind_global(node, parent):
        local = ai_matrix_to_mat4(node.transformation)
        combined = parent @ local
        result = {}
        if node.name in guard_mesh.bone_names:
            result[node.name] = combined
        for child in node.children:
            result.update(bind_global(child, combined))
        return result

    globals_at_bind = bind_global(guard_mesh._scene.root_node, Mat4())

    for name, index in guard_mesh.bone_names.items():
        offset = guard_mesh.bone_offsets[index]
        check = guard_mesh._global_inverse @ globals_at_bind[name] @ offset
        assert check.to_numpy() == pytest.approx(np.eye(4), abs=1e-3), name


def test_bone_weights_sum_to_one_per_vertex(guard_mesh: SkinnedMesh):
    sums = guard_mesh.bone_weights.sum(axis=1)
    assert sums == pytest.approx(np.ones_like(sums), abs=1e-4)


def test_rotation_keys_are_monotonic_for_every_channel(guard_mesh: SkinnedMesh):
    animation = guard_mesh._scene.animations[0]
    for channel in animation.channels:
        keys = _read_rotation_keys(channel)
        times = [key.time for key in keys]
        assert times == sorted(times), channel.node_name
        assert times[0] >= 0.0
        assert times[-1] <= animation.duration


def test_bracketing_keys_uses_the_last_segment_past_the_end():
    keys = [_Key(0.0, (0,)), _Key(1.0, (1,)), _Key(2.0, (2,))]
    k0, k1, factor = _bracketing_keys(5.0, keys)
    assert (k0.time, k1.time) == (1.0, 2.0)
    assert factor == pytest.approx((5.0 - 1.0) / (2.0 - 1.0))


def test_bracketing_keys_single_key_has_zero_factor():
    keys = [_Key(3.0, (9,))]
    k0, k1, factor = _bracketing_keys(1.0, keys)
    assert k0 is keys[0] and k1 is keys[0]
    assert factor == 0.0


def test_interpolate_vector_midpoint():
    keys = [_Key(0.0, (0.0, 0.0, 0.0)), _Key(2.0, (2.0, 4.0, -2.0))]
    result = _interpolate_vector(1.0, keys)
    assert (result.x, result.y, result.z) == pytest.approx((1.0, 2.0, -1.0))


def test_interpolate_rotation_midpoint_is_normalised():
    identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    half_turn_y = Quaternion(0.0, 0.0, 1.0, 0.0)
    keys = [
        _Key(0.0, (identity.s, identity.x, identity.y, identity.z)),
        _Key(1.0, (half_turn_y.s, half_turn_y.x, half_turn_y.y, half_turn_y.z)),
    ]
    result = _interpolate_rotation(0.5, keys)
    length = (result.s**2 + result.x**2 + result.y**2 + result.z**2) ** 0.5
    assert length == pytest.approx(1.0, abs=1e-5)
