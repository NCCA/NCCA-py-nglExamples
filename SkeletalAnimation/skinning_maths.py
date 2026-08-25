"""Linear blend skinning and dual-quaternion skinning, numpy-only.

This module is the CPU reference for the SkeletalAnimation demo, and it is
deliberately free of any GL/Qt import so the maths can be pytest'd headless.
The GLSL vertex shader (``shaders/SkinVertex.glsl``) is a line-for-line
transcription of ``skin_lbs`` and ``skin_dqs`` below -- when you change one,
change the other, and re-run the tests as the cross-check.

Everything follows the repo's row-vector convention: a point is skinned as
``v @ bone_matrix``, with translation living in row 3. There is no bone
asset to load -- the skeleton is a straight chain of bones along +y, built
procedurally by ``bind_pose``/``pose_matrices``, and the mesh is a tube
wrapped around it by ``build_tube_mesh``.

The reason this demo exists at all is the "candy-wrapper" artefact: twist a
joint a long way and LBS's per-influence *linear* blend of world matrices
pinches the mesh towards the bone axis, whereas dual quaternion blending
(DLB, Kavan et al. 2007, "Skinning with Dual Quaternions") blends the
*rotations* instead and keeps the cross-section rigid. See
``tests/test_skinning_maths.py::TestCandyWrapper`` for the numeric proof.
"""

import numpy as np


# ----------------------------------------------------------------------
# Small affine-matrix builders (row-vector convention: v_world = v @ M,
# translation in row 3). These are plain numpy, not ncca.ngl.Mat4, so that
# composing two transforms with a bare ``A @ B`` means "apply A, then B" --
# regular matrix multiplication is associative, so no operator surgery is
# needed the way ncca.ngl's Mat4.__matmul__ needs it for its
# column-major-reading style.
# ----------------------------------------------------------------------
def _rotate_y(angle_deg: float) -> np.ndarray:
    """Rotation about +y (a bone's own long axis -- this is the twist axis)."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 2] = -s
    m[2, 0] = s
    m[2, 2] = c
    return m


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[3, 0] = x
    m[3, 1] = y
    m[3, 2] = z
    return m


# ----------------------------------------------------------------------
# skeleton: a straight chain of bones along +y
# ----------------------------------------------------------------------
def pose_matrices(angles_deg: list[float], bone_length: float) -> list[np.ndarray]:
    """Forward-kinematics world matrix per bone for a straight +y chain.

    Each bone twists about its own +y axis by ``angles_deg[i]`` (the axis
    the chain runs along), then is placed ``bone_length`` above its parent's
    tip -- bone 0 sits at the origin. Because the twist axis and the offset
    axis are the same, twisting never moves a bone's origin, only its
    orientation: exactly what you want to isolate the candy-wrapper effect
    from ordinary bending.
    """
    world = []
    parent = np.eye(4)
    for i, angle in enumerate(angles_deg):
        offset = 0.0 if i == 0 else bone_length
        local = _rotate_y(angle) @ _translate(0.0, offset, 0.0)
        w = local @ parent
        world.append(w)
        parent = w
    return world


def bind_pose(
    n_bones: int, bone_length: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rest-pose world matrices and their inverses, one pair per bone."""
    bind = pose_matrices([0.0] * n_bones, bone_length)
    inverse_bind = [np.linalg.inv(m) for m in bind]
    return bind, inverse_bind


# ----------------------------------------------------------------------
# procedural mesh: a tube shell wrapped around the chain
# ----------------------------------------------------------------------
def build_tube_mesh(
    n_bones: int,
    bone_length: float,
    radius: float = 0.15,
    segments: int = 16,
    ring_spacing: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cylinder shell around the bind-pose chain.

    Returns (verts (V,3), normals (V,3), triangle indices (T,3) int32,
    bone_origins (n_bones,3)) -- all in bind space, ready for
    ``compute_weights`` and ``skin_lbs``/``skin_dqs``.
    """
    total_length = (n_bones - 1) * bone_length
    n_rings = round(total_length / ring_spacing) + 1
    ring_ys = np.linspace(0.0, total_length, n_rings)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    verts = np.empty((n_rings * segments, 3))
    normals = np.empty((n_rings * segments, 3))
    for ring, y in enumerate(ring_ys):
        base = ring * segments
        verts[base : base + segments, 0] = radius * cos_a
        verts[base : base + segments, 1] = y
        verts[base : base + segments, 2] = radius * sin_a
        normals[base : base + segments, 0] = cos_a
        normals[base : base + segments, 1] = 0.0
        normals[base : base + segments, 2] = sin_a

    faces = []
    for ring in range(n_rings - 1):
        for seg in range(segments):
            seg_next = (seg + 1) % segments
            a = ring * segments + seg
            b = ring * segments + seg_next
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + seg_next
            faces.append((a, b, c))
            faces.append((b, d, c))
    indices = np.array(faces, dtype=np.int32)

    bone_origins = np.array([[0.0, i * bone_length, 0.0] for i in range(n_bones)])
    return verts, normals, indices, bone_origins


def compute_palette(
    pose: list[np.ndarray], inverse_bind: list[np.ndarray]
) -> np.ndarray:
    """Combine pose and inverse-bind matrices into the per-bone skinning
    matrices ``skin_lbs``/``skin_dqs`` expect: ``v_skinned = v_bind @
    palette[i]``. Shape (n_bones, 4, 4)."""
    return np.stack([inverse_bind[i] @ pose[i] for i in range(len(pose))])


def compute_dq_palette(palette: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-bone dual quaternions built from the same skinning matrices LBS
    uses, for ``skin_dqs``/the GPU DQS path."""
    return [dual_quat_from_mat(m) for m in palette]


def compute_weights(
    verts: np.ndarray, bone_origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-nearest-bone skin weights with linear (inverse-distance) falloff.

    Returns (indices (V,2) int32, weights (V,2) float32); weights always
    sum to 1 per vertex.
    """
    verts = np.asarray(verts, dtype=np.float64)
    bone_origins = np.asarray(bone_origins, dtype=np.float64)

    diffs = verts[:, None, :] - bone_origins[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (V, n_bones)
    nearest2 = np.argsort(dists, axis=1)[:, :2]
    d0 = np.take_along_axis(dists, nearest2[:, 0:1], axis=1)[:, 0]
    d1 = np.take_along_axis(dists, nearest2[:, 1:2], axis=1)[:, 0]
    total = d0 + d1
    on_top_of_bone = total == 0.0
    total_safe = np.where(on_top_of_bone, 1.0, total)
    # nearer bone gets the larger weight, hence swapped d1/d0
    w0 = np.where(on_top_of_bone, 1.0, d1 / total_safe)
    w1 = np.where(on_top_of_bone, 0.0, d0 / total_safe)

    idx = nearest2.astype(np.int32)
    wts = np.stack([w0, w1], axis=1).astype(np.float32)
    return idx, wts


# ----------------------------------------------------------------------
# linear blend skinning (LBS) -- the CPU reference for the vertex shader
# ----------------------------------------------------------------------
def skin_lbs(
    verts: np.ndarray,
    normals: np.ndarray,
    idx: np.ndarray,
    wts: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend per-influence bone matrices linearly, then apply to (v, n).

    ``palette`` is the array of per-bone skinning matrices
    (``inverse_bind[j] @ pose[j]``), shape (n_bones, 4, 4). This is exactly
    what the artefact test pins: for a vertex weighted 50/50 between two
    bones related by a 180 degree twist, the two transformed positions are
    each other's antipode about the joint axis and the *linear* average
    collapses to the axis itself.
    """
    verts = np.asarray(verts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    idx = np.asarray(idx)
    wts = np.asarray(wts, dtype=np.float64)
    palette = np.asarray(palette, dtype=np.float64)

    verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)

    skinned_verts = np.zeros_like(verts)
    skinned_normals = np.zeros_like(normals)
    for k in range(idx.shape[1]):
        bone_idx = idx[:, k]
        w = wts[:, k : k + 1]
        mats = palette[bone_idx]  # (V, 4, 4)
        transformed = np.einsum("vi,vij->vj", verts_h, mats)
        skinned_verts += w * transformed[:, :3]
        rotated_normals = np.einsum("vi,vij->vj", normals, mats[:, :3, :3])
        skinned_normals += w * rotated_normals

    lengths = np.linalg.norm(skinned_normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    skinned_normals = skinned_normals / lengths
    return skinned_verts, skinned_normals


# ----------------------------------------------------------------------
# quaternion / dual-quaternion helpers for DQS
#
# Quaternions are stored as numpy arrays [s, x, y, z] (scalar first),
# matching ncca.ngl.Quaternion's component order and its matrix<->quaternion
# conversion formulae, adapted here for a plain (4,4) numpy matrix instead
# of a Mat4 object so this module has no dependency beyond numpy.
# ----------------------------------------------------------------------
def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2."""
    s1, x1, y1, z1 = q1
    s2, x2, y2, z2 = q2
    return np.array(
        [
            s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2,
            s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2,
            s1 * y2 - x1 * z2 + y1 * s2 + z1 * x2,
            s1 * z2 + x1 * y2 - y1 * x2 + z1 * s2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_mat(m: np.ndarray) -> np.ndarray:
    """Extract a unit quaternion [s, x, y, z] from a rotation's upper 3x3.

    Same branch-on-trace algorithm as ``ncca.ngl.Quaternion.from_mat4``,
    just indexed straight off the 2D array instead of a flat row-major list.
    """
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]

    trace = 1.0 + m00 + m11 + m22
    if trace > 1e-8:
        scale = np.sqrt(trace) * 2.0
        x = (m12 - m21) / scale
        y = (m20 - m02) / scale
        z = (m01 - m10) / scale
        s = 0.25 * scale
    elif m00 > m11 and m00 > m22:
        scale = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        x = 0.25 * scale
        y = (m10 + m01) / scale
        z = (m02 + m20) / scale
        s = (m12 - m21) / scale
    elif m11 > m22:
        scale = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        x = (m10 + m01) / scale
        y = 0.25 * scale
        z = (m21 + m12) / scale
        s = (m20 - m02) / scale
    else:
        scale = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        x = (m20 + m02) / scale
        y = (m21 + m12) / scale
        z = 0.25 * scale
        s = (m01 - m10) / scale

    q = np.array([s, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Rotation quaternion -> 3x3, row-vector convention (mirrors
    ``ncca.ngl.Quaternion.to_mat4``'s upper-left block)."""
    s, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + s * z), 2.0 * (x * z - s * y)],
            [2.0 * (x * y - s * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + s * x)],
            [2.0 * (x * z + s * y), 2.0 * (y * z - s * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def dual_quat_from_mat(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 4x4 (row-vector, translation in row 3) -> dual quaternion (qr, qd)."""
    qr = quat_from_mat(m)
    t = m[3, :3]
    qt = np.array([0.0, t[0], t[1], t[2]])
    qd = 0.5 * quat_mult(qt, qr)
    return qr, qd


def dq_to_mat(qr: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Dual quaternion -> 4x4 rigid matrix (row-vector, translation in row 3)."""
    qr = qr / np.linalg.norm(qr)
    rot = quat_to_mat3(qr)
    t_quat = 2.0 * quat_mult(qd, quat_conjugate(qr))
    m = np.eye(4)
    m[:3, :3] = rot
    m[3, :3] = t_quat[1:4]
    return m


def dlb_blend(
    dqs: list[tuple[np.ndarray, np.ndarray]], wts: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Dual quaternion linear blending (Kavan et al. 2007).

    Two unit quaternions q and -q encode the identical rotation, so before
    summing, every dq is hemisphere-aligned against the first (dot < 0 ->
    negate both qr and qd) -- skip this and opposite-signed influences
    cancel instead of blending, which is the bug this step exists to avoid.
    """
    reference = dqs[0][0]
    qr_sum = np.zeros(4)
    qd_sum = np.zeros(4)
    for (qr, qd), w in zip(dqs, wts):
        if np.dot(qr, reference) < 0.0:
            qr, qd = -qr, -qd
        qr_sum += w * qr
        qd_sum += w * qd
    norm = np.linalg.norm(qr_sum)
    return qr_sum / norm, qd_sum / norm


def skin_dqs(
    verts: np.ndarray,
    normals: np.ndarray,
    idx: np.ndarray,
    wts: np.ndarray,
    palette: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-quaternion skinning -- the CPU reference for the vertex shader.

    ``palette`` is a list of per-bone (qr, qd) dual quaternions built from
    the same skinning matrices LBS uses (``dual_quat_from_mat`` of
    ``inverse_bind[j] @ pose[j]``). Kept as a small per-vertex Python loop
    (fine at the ~2k-vertex scale this demo stays under) rather than
    vectorised, since ``dlb_blend``'s hemisphere test is inherently
    per-vertex data-dependent branching.
    """
    verts = np.asarray(verts, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    idx = np.asarray(idx)
    wts = np.asarray(wts, dtype=np.float64)

    skinned_verts = np.empty_like(verts)
    skinned_normals = np.empty_like(normals)
    for v in range(len(verts)):
        influences = [palette[idx[v, k]] for k in range(idx.shape[1])]
        qr, qd = dlb_blend(influences, wts[v])
        m = dq_to_mat(qr, qd)
        v_h = np.append(verts[v], 1.0)
        skinned_verts[v] = (v_h @ m)[:3]
        skinned_normals[v] = normals[v] @ m[:3, :3]

    lengths = np.linalg.norm(skinned_normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    skinned_normals = skinned_normals / lengths
    return skinned_verts, skinned_normals
