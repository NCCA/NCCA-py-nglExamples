"""Load a rigged mesh via impasse and animate it with linear blend skinning.

Direct port of the ogldev/NGL "SkeletalAnimation" tutorial
(``AssetImportDemos/SkeletalAnimation`` in NGL9Demos) onto PyNGL, using
impasse (https://pypi.org/project/impasse/) instead of NGL's C++ assimp
bindings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cffi
import impasse
import numpy as np
from impasse import helper
from impasse.constants import ProcessingStep
from impasse.structs_base import ffi as _impasse_ffi
from ncca.ngl import Mat4, Quaternion, Vec3

MAX_BONES_PER_VERTEX = 4

# ---------------------------------------------------------------------------
# impasse 5.4.2 (the latest release on PyPI at time of writing) declares
# `struct aiBone` as mName, mArmature, mNode, mNumWeights, mWeights,
# mOffsetMatrix. The installed libassimp (6.0.5 here; see
# /opt/homebrew/include/assimp/mesh.h) actually orders it mName,
# mNumWeights, mArmature, mNode, mWeights, mOffsetMatrix -- mNumWeights
# and the armature/node pointer pair are swapped. Reading `Bone.weights` or
# `Bone.offset_matrix` through impasse's own accessors walks off that
# offset and segfaults. Positions, faces, UVs, animation channels and the
# node hierarchy are all unaffected -- only aiBone is misdeclared -- so the
# two fields below are read by re-casting the same struct pointer through a
# private cffi definition with the corrected layout, and impasse is used
# normally for everything else.
_bone_ffi = cffi.FFI()
_bone_ffi.cdef("""
    struct aiVertexWeightFixed { unsigned int vertex_id; float weight; };
    struct aiMatrix4x4Fixed {
        float a1, a2, a3, a4, b1, b2, b3, b4,
              c1, c2, c3, c4, d1, d2, d3, d4;
    };
    struct aiBoneFixed {
        struct { unsigned int length; char data[1024]; } name;
        unsigned int num_weights;
        void *armature;
        void *node;
        struct aiVertexWeightFixed *weights;
        struct aiMatrix4x4Fixed offset_matrix;
    };
""")


def _read_bone_weights_and_offset(bone) -> tuple[list[tuple[int, float]], np.ndarray]:
    """Read a Bone's vertex weights and bind-pose offset matrix.

    Returns
    -------
        weights : list of (vertex_id, weight) pairs
        offset : the 4x4 offset matrix, assimp convention (column vector,
            translation in the last column)
    """
    address = int(_impasse_ffi.cast("uintptr_t", bone.struct))
    fixed = _bone_ffi.cast("struct aiBoneFixed *", address)
    weights = [
        (fixed.weights[i].vertex_id, fixed.weights[i].weight)
        for i in range(fixed.num_weights)
    ]
    m = fixed.offset_matrix
    offset = np.array(
        [
            [m.a1, m.a2, m.a3, m.a4],
            [m.b1, m.b2, m.b3, m.b4],
            [m.c1, m.c2, m.c3, m.c4],
            [m.d1, m.d2, m.d3, m.d4],
        ],
        dtype=np.float32,
    )
    return weights, offset


# ---------------------------------------------------------------------------
# A second, distinct impasse struct bug: its `aiQuatKey` cdef omits the
# trailing `aiAnimInterpolation mInterpolation` field that real assimp (see
# /opt/homebrew/include/assimp/anim.h) has on both aiVectorKey and
# aiQuatKey. That only matters for aiQuatKey: aiVectorKey's 20 real bytes
# (double + vec3) round up to a 24-byte stride for 8-byte alignment either
# way, so the missing field is absorbed by padding and position/scaling
# keys read fine. aiQuatKey's 24 real bytes (double + vec4) are already
# 8-byte aligned, so the missing 4-byte enum field is NOT absorbed --
# impasse computes a 24-byte element stride where the real array uses 32,
# and `rotation_keys[i]` for i > 0 reads from the wrong offset. Position 0
# happens to read correctly regardless, which is why a quick smoke-test of
# a single key looks fine. rotation_keys is read directly here instead.
_key_ffi = cffi.FFI()
_key_ffi.cdef("""
    struct aiQuaternionFixed { float w, x, y, z; };
    struct aiQuatKeyFixed {
        double time;
        struct aiQuaternionFixed value;
        int interpolation;
    };
""")


class _Key:
    __slots__ = ("time", "value")

    def __init__(self, time: float, value: tuple[float, ...]) -> None:
        self.time = time
        self.value = value


def _read_rotation_keys(channel) -> list[_Key]:
    """Read a channel's rotation keys, working around the impasse bug above."""
    count = channel.struct.mNumRotationKeys
    address = int(_impasse_ffi.cast("uintptr_t", channel.struct.mRotationKeys))
    fixed = _key_ffi.cast("struct aiQuatKeyFixed *", address)
    return [
        _Key(
            fixed[i].time,
            (fixed[i].value.w, fixed[i].value.x, fixed[i].value.y, fixed[i].value.z),
        )
        for i in range(count)
    ]


def ai_matrix_to_mat4(matrix: np.ndarray) -> Mat4:
    """Convert an assimp matrix to PyNGL's row-vector ``Mat4``.

    assimp matrices are column-vector (translation in the last column);
    PyNGL's are row-vector (translation in row 3), so this transposes.
    """
    return Mat4.from_numpy(np.asarray(matrix, dtype=np.float32).T)


@dataclass
class SubMesh:
    """One draw call's worth of the combined mesh: an index range and its texture."""

    index_count: int
    index_offset: int
    texture_path: str | None


class SkinnedMesh:
    """A rigged mesh imported via impasse, ready for GPU linear blend skinning.

    All sub-meshes in the source scene are merged into single position /
    normal / UV / bone-id / bone-weight buffers and one index buffer, with
    each sub-mesh's vertex offset baked directly into its indices --
    :class:`SubMesh` then just records an (offset, count) range per texture.
    """

    def __init__(self, filename: str) -> None:
        self._scene = impasse.load(
            filename,
            processing=ProcessingStep.Triangulate | ProcessingStep.GenSmoothNormals,
        )
        self._directory = Path(filename).parent
        self.bone_names: dict[str, int] = {}
        self.bone_offsets: list[Mat4] = []
        self.submeshes: list[SubMesh] = []
        self._global_inverse = ai_matrix_to_mat4(
            self._scene.root_node.transformation
        ).inverse()

        self.positions: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.texcoords: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.bone_ids: np.ndarray = np.empty(
            (0, MAX_BONES_PER_VERTEX), dtype=np.float32
        )
        self.bone_weights: np.ndarray = np.empty(
            (0, MAX_BONES_PER_VERTEX), dtype=np.float32
        )
        self.indices: np.ndarray = np.empty((0,), dtype=np.uint32)

        if len(self._scene.animations) < 1:
            raise ValueError(f"{filename} has no animations")

        self._build_buffers()

    # ------------------------------------------------------------ loading

    def _diffuse_texture(self, material) -> str | None:
        for prop in material.properties:
            if prop.key == "$tex.file":
                return prop.data
        return None

    def _build_buffers(self) -> None:
        positions: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        texcoords: list[tuple[float, float]] = []
        bone_ids: list[list[float]] = []
        bone_weights: list[list[float]] = []
        indices: list[int] = []

        vertex_base = 0
        index_base = 0

        for ai_mesh in self._scene.meshes:
            vertex_count = len(ai_mesh.vertices)
            mesh_bone_ids = [[0.0] * MAX_BONES_PER_VERTEX for _ in range(vertex_count)]
            mesh_bone_weights = [
                [0.0] * MAX_BONES_PER_VERTEX for _ in range(vertex_count)
            ]

            for bone in ai_mesh.bones:
                name = bone.name
                weights, offset = _read_bone_weights_and_offset(bone)
                bone_index = self.bone_names.get(name)
                if bone_index is None:
                    bone_index = len(self.bone_names)
                    self.bone_names[name] = bone_index
                    self.bone_offsets.append(ai_matrix_to_mat4(offset))

                for vertex_id, weight in weights:
                    slots = mesh_bone_weights[vertex_id]
                    for slot in range(MAX_BONES_PER_VERTEX):
                        if slots[slot] == 0.0:
                            mesh_bone_ids[vertex_id][slot] = float(bone_index)
                            slots[slot] = weight
                            break

            uv_channel = ai_mesh.texture_coords[0] if ai_mesh.texture_coords else None
            for i in range(vertex_count):
                positions.append(tuple(ai_mesh.vertices[i]))
                normals.append(
                    tuple(ai_mesh.normals[i]) if ai_mesh.normals else (0.0, 0.0, 1.0)
                )
                if uv_channel is not None:
                    uv = uv_channel[i]
                    texcoords.append((float(uv[0]), float(uv[1])))
                else:
                    texcoords.append((0.0, 0.0))
                bone_ids.append(mesh_bone_ids[i])
                bone_weights.append(mesh_bone_weights[i])

            face_index_count = len(ai_mesh.faces) * 3
            for face in ai_mesh.faces:
                indices.extend(index + vertex_base for index in face)

            texture_name = self._diffuse_texture(ai_mesh.material)
            self.submeshes.append(
                SubMesh(
                    index_count=face_index_count,
                    index_offset=index_base,
                    texture_path=(
                        str(self._directory / texture_name) if texture_name else None
                    ),
                )
            )
            vertex_base += vertex_count
            index_base += face_index_count

        self.positions = np.array(positions, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.texcoords = np.array(texcoords, dtype=np.float32)
        self.bone_ids = np.array(bone_ids, dtype=np.float32)
        self.bone_weights = np.array(bone_weights, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

    # ---------------------------------------------------------- animation

    def bounding_box(self) -> tuple[list[float], list[float]]:
        """Return the (min, max) corners of the scene's world-space bounding box."""
        return helper.get_bounding_box(self._scene)

    def duration(self) -> float:
        """Duration of the (only) animation, in ticks."""
        return self._scene.animations[0].duration

    def ticks_per_second(self) -> float:
        """Playback rate of the animation, in ticks per second."""
        tps = self._scene.animations[0].ticks_per_second
        return tps if tps != 0 else 25.0

    def bone_transforms(self, time_seconds: float) -> list[Mat4]:
        """Return the current skinning matrix for every bone, indexed by ``bone_names``."""
        animation = self._scene.animations[0]
        time_in_ticks = time_seconds * self.ticks_per_second()
        animation_time = math.fmod(time_in_ticks, animation.duration)

        transforms = [Mat4() for _ in self.bone_names]
        self._walk_hierarchy(
            animation, animation_time, self._scene.root_node, Mat4(), transforms
        )
        return transforms

    def _find_channel(self, animation, node_name: str):
        for channel in animation.channels:
            if channel.node_name == node_name:
                return channel
        return None

    def _walk_hierarchy(self, animation, animation_time, node, parent_transform, out):
        channel = self._find_channel(animation, node.name)
        if channel is not None:
            local_transform = self._interpolated_transform(animation_time, channel)
        else:
            local_transform = ai_matrix_to_mat4(node.transformation)

        global_transform = parent_transform @ local_transform

        bone_index = self.bone_names.get(node.name)
        if bone_index is not None:
            out[bone_index] = (
                self._global_inverse @ global_transform @ self.bone_offsets[bone_index]
            )

        for child in node.children:
            self._walk_hierarchy(
                animation, animation_time, child, global_transform, out
            )

    def _interpolated_transform(self, time: float, channel) -> Mat4:
        scale = _interpolate_vector(time, channel.scaling_keys)
        rotation = _interpolate_rotation(time, _read_rotation_keys(channel))
        translation = _interpolate_vector(time, channel.position_keys)

        transform = Mat4.scale(scale.x, scale.y, scale.z) @ rotation.to_mat4()
        transform[3, 0] = translation.x
        transform[3, 1] = translation.y
        transform[3, 2] = translation.z
        return transform


def _bracketing_keys(time: float, keys):
    """Return the two keys either side of ``time``, clamping at the ends."""
    if len(keys) == 1:
        return keys[0], keys[0], 0.0
    for i in range(len(keys) - 1):
        if time < keys[i + 1].time:
            k0, k1 = keys[i], keys[i + 1]
            break
    else:
        # impasse's key sequences don't support negative indices.
        k0, k1 = keys[len(keys) - 2], keys[len(keys) - 1]
    delta = k1.time - k0.time
    factor = 0.0 if delta <= 0.0 else (time - k0.time) / delta
    return k0, k1, factor


def _interpolate_vector(time: float, keys) -> Vec3:
    """Linearly interpolate a position/scaling animation channel at ``time``."""
    k0, k1, factor = _bracketing_keys(time, keys)
    start = Vec3(*k0.value)
    end = Vec3(*k1.value)
    return start.lerp(end, factor)


def _interpolate_rotation(time: float, keys) -> Quaternion:
    """Spherically interpolate a rotation animation channel at ``time``."""
    k0, k1, factor = _bracketing_keys(time, keys)
    start = Quaternion(*k0.value)
    end = Quaternion(*k1.value)
    return start.slerp(end, factor).normalized()
