"""Image-based lighting precompute -- numpy only, no GL/Qt imports.

Everything here runs once on the CPU at startup and is cached to ``.npy``
files next to this script (see ``load_or_compute`` at the bottom) so the
demo pays the cost once, not every launch. Splitting it out of ``main.py``
also means these are the only bits of the demo you can unit-test without a
GL context -- see ``tests/test_ibl_precompute.py``.

Two of the brief's four precompute stages are implemented:

1. **Irradiance map** -- a 16x16-per-face cubemap holding the
   cosine-weighted hemisphere integral of the source environment at each
   texel's normal direction. This is the diffuse IBL term: sampling it by
   surface normal and multiplying by albedo replaces the flat
   ``vec3(0.03) * albedo * ao`` ambient hack in ``PBR/SimplePBR``.
2. **BRDF LUT** -- Karis's split-sum approximation
   (B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course notes,
   https://blog.selfshadow.com/publications/s2013-shading-course/), ported
   from the reference GLSL in the LearnOpenGL IBL chapter
   (https://learnopengl.com/PBR/IBL/Specular-IBL). A 2D texture of (NdotV,
   roughness) -> (A, B) scale/bias pairs for the environment BRDF integral,
   computed once via GGX importance sampling over a Hammersley sequence.

Stage 2 of the brief (the roughness-mip-mapped prefiltered specular chain)
was **cut for time** -- this is the largest of the six tasks and the brief
explicitly sanctions shipping stages 1-2 (here: the irradiance map and the
LUT, since they were flagged as the most testable/highest-value pair) and
recording the cut. ``main.py`` approximates the missing prefiltered
environment by sampling the base cubemap directly along the reflection
vector (no roughness blur) -- see its module docstring for the exact
consequence on the render. There is deliberately no test for the prefilter
chain here, because there is no code for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent
IRRADIANCE_CACHE = CACHE_DIR / "irradiance_cache.npy"
BRDF_LUT_CACHE = CACHE_DIR / "brdf_lut_cache.npy"

IRRADIANCE_SIZE = 16
BRDF_LUT_SIZE = 256
BRDF_LUT_SAMPLES = 256

# Face order matches cubemap_gen.FACE_ORDER: +x, -x, +y, -y, +z, -z.
_FACE_ORDER = ("+x", "-x", "+y", "-y", "+z", "-z")


# ---------------------------------------------------------------------------
# Cubemap sampling (needed to convolve the *source* environment)
# ---------------------------------------------------------------------------
def _direction_to_face_uv(
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised inverse of ``cubemap_gen._face_directions``.

    Given an (..., 3) array of direction vectors, returns
    ``(face_index, u, v)`` where ``face_index`` indexes ``_FACE_ORDER`` and
    ``u, v`` are in [-1, 1], using exactly the per-face sign/axis
    conventions ``cubemap_gen.py`` bakes its analytic sky into (so a
    generated cubemap round-trips through this sampler without a seam).
    """
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    face_index = np.zeros(direction.shape[:-1], dtype=np.int64)
    u = np.zeros(direction.shape[:-1])
    v = np.zeros(direction.shape[:-1])

    is_x = (ax >= ay) & (ax >= az)
    is_y = (~is_x) & (ay >= az)
    is_z = (~is_x) & (~is_y)

    # +x / -x
    px = is_x & (x > 0)
    nx = is_x & ~px
    m = np.where(px, x, -x)
    face_index = np.where(px, 0, face_index)
    face_index = np.where(nx, 1, face_index)
    u = np.where(px, z / np.where(m == 0, 1, m), u)
    v = np.where(px, y / np.where(m == 0, 1, m), v)
    m_nx = -x
    u = np.where(nx, -z / np.where(m_nx == 0, 1, m_nx), u)
    v = np.where(nx, y / np.where(m_nx == 0, 1, m_nx), v)

    # +y / -y
    py = is_y & (y > 0)
    ny = is_y & ~py
    m_py = y
    face_index = np.where(py, 2, face_index)
    u = np.where(py, x / np.where(m_py == 0, 1, m_py), u)
    v = np.where(py, -z / np.where(m_py == 0, 1, m_py), v)
    m_ny = -y
    face_index = np.where(ny, 3, face_index)
    u = np.where(ny, x / np.where(m_ny == 0, 1, m_ny), u)
    v = np.where(ny, z / np.where(m_ny == 0, 1, m_ny), v)

    # +z / -z
    pz = is_z & (z > 0)
    nz = is_z & ~pz
    m_pz = z
    face_index = np.where(pz, 4, face_index)
    u = np.where(pz, -x / np.where(m_pz == 0, 1, m_pz), u)
    v = np.where(pz, y / np.where(m_pz == 0, 1, m_pz), v)
    m_nz = -z
    face_index = np.where(nz, 5, face_index)
    u = np.where(nz, x / np.where(m_nz == 0, 1, m_nz), u)
    v = np.where(nz, y / np.where(m_nz == 0, 1, m_nz), v)

    return face_index, np.clip(u, -1.0, 1.0), np.clip(v, -1.0, 1.0)


def sample_cubemap(faces: list[np.ndarray], direction: np.ndarray) -> np.ndarray:
    """Bilinear-sample a 6-face cubemap (list of (size, size, 3) arrays, in
    ``_FACE_ORDER``) at a batch of direction vectors, shape (..., 3) ->
    (..., 3)."""
    face_index, u, v = _direction_to_face_uv(direction)
    size = faces[0].shape[0]
    out = np.zeros(direction.shape[:-1] + (3,))

    col = (u + 1.0) * 0.5 * (size - 1)
    row = (v + 1.0) * 0.5 * (size - 1)
    col0 = np.clip(np.floor(col).astype(np.int64), 0, size - 2 if size > 1 else 0)
    row0 = np.clip(np.floor(row).astype(np.int64), 0, size - 2 if size > 1 else 0)
    col1 = np.minimum(col0 + 1, size - 1)
    row1 = np.minimum(row0 + 1, size - 1)
    fc = (col - col0)[..., None]
    fr = (row - row0)[..., None]

    for i, face in enumerate(faces):
        mask = face_index == i
        if not np.any(mask):
            continue
        c0, c1, r0, r1 = col0[mask], col1[mask], row0[mask], row1[mask]
        top = face[r0, c0] * (1 - fc[mask]) + face[r0, c1] * fc[mask]
        bottom = face[r1, c0] * (1 - fc[mask]) + face[r1, c1] * fc[mask]
        out[mask] = top * (1 - fr[mask]) + bottom * fr[mask]
    return out


# ---------------------------------------------------------------------------
# Stage 1: irradiance map (cosine-weighted hemisphere convolution)
# ---------------------------------------------------------------------------
def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, up) orthonormal to ``normal`` (single 3-vector)."""
    up_hint = (
        np.array([0.0, 1.0, 0.0])
        if abs(normal[1]) < 0.999
        else np.array([1.0, 0.0, 0.0])
    )
    right = np.cross(up_hint, normal)
    right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    return right, up


def _face_normals(size: int) -> list[np.ndarray]:
    """The world-space direction (== normal, since it's a unit cube) each
    output texel of a ``size``x``size`` cubemap face corresponds to."""
    from cubemap_gen import (
        FACE_ORDER,
        _face_directions,
    )  # local import: numpy-only helper

    normals = []
    for face in FACE_ORDER:
        dirs = _face_directions(face, size)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        normals.append(dirs)
    return normals


def generate_irradiance_map(
    faces: list[np.ndarray], out_size: int = IRRADIANCE_SIZE, sample_delta: float = 0.15
) -> list[np.ndarray]:
    """Convolve a source cubemap (list of 6 (size, size, 3) arrays, values
    in [0, 1], in ``_FACE_ORDER``) into an irradiance map of the requested
    per-face resolution.

    Follows the LearnOpenGL derivation
    (https://learnopengl.com/PBR/IBL/Diffuse-irradiance): for each output
    normal N, integrate incoming radiance over the hemisphere weighted by
    ``cos(theta) * sin(theta)`` (the ``sin`` is the solid-angle Jacobian),
    then scale by pi so a *constant* environment colour C convolves back to
    exactly C -- this is exactly what
    ``test_uniform_environment_irradiance_equals_its_colour`` checks.
    """
    thetas = np.arange(0.0, 0.5 * np.pi, sample_delta)
    phis = np.arange(0.0, 2.0 * np.pi, sample_delta)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    tangent_samples = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weights = (np.sin(theta_grid) * np.cos(theta_grid)).reshape(-1)
    n_samples = tangent_samples.shape[0]

    output_normals = _face_normals(out_size)
    result = []
    for normals in output_normals:
        flat_normals = normals.reshape(-1, 3)
        face_out = np.zeros((flat_normals.shape[0], 3))
        for i, normal in enumerate(flat_normals):
            right, up = _tangent_basis(normal)
            world_dirs = (
                tangent_samples[:, 0:1] * right
                + tangent_samples[:, 1:2] * up
                + tangent_samples[:, 2:3] * normal
            )
            radiance = sample_cubemap(faces, world_dirs)
            face_out[i] = np.pi * (weights[:, None] * radiance).sum(axis=0) / n_samples
        result.append(face_out.reshape(out_size, out_size, 3))
    return result


# ---------------------------------------------------------------------------
# Stage 3: BRDF split-sum LUT
# ---------------------------------------------------------------------------
def _hammersley(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Hammersley low-discrepancy sequence (Karis 2013 appendix), vectorised
    bit-reversal radical inverse -- the standard GGX importance-sampling
    driver used throughout real-time PBR (LearnOpenGL, UE4)."""
    i = np.arange(n_samples, dtype=np.uint32)
    bits = i.copy()
    bits = (bits << np.uint32(16)) | (bits >> np.uint32(16))
    bits = ((bits & np.uint32(0x55555555)) << np.uint32(1)) | (
        (bits & np.uint32(0xAAAAAAAA)) >> np.uint32(1)
    )
    bits = ((bits & np.uint32(0x33333333)) << np.uint32(2)) | (
        (bits & np.uint32(0xCCCCCCCC)) >> np.uint32(2)
    )
    bits = ((bits & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | (
        (bits & np.uint32(0xF0F0F0F0)) >> np.uint32(4)
    )
    bits = ((bits & np.uint32(0x00FF00FF)) << np.uint32(8)) | (
        (bits & np.uint32(0xFF00FF00)) >> np.uint32(8)
    )
    radical_inverse = bits.astype(np.float64) * 2.3283064365386963e-10  # 1 / 2**32
    return i.astype(np.float64) / n_samples, radical_inverse


def compute_brdf_lut(
    size: int = BRDF_LUT_SIZE, sample_count: int = BRDF_LUT_SAMPLES
) -> np.ndarray:
    """Karis split-sum BRDF LUT: (size, size, 2) array of (A, B) indexed by
    ``[NdotV_index, roughness_index]``, both axes running 0..1.

    Ported from the reference GLSL ``IntegrateBRDF`` in the LearnOpenGL
    specular-IBL chapter (https://learnopengl.com/PBR/IBL/Specular-IBL),
    which is itself Karis's derivation
    (B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013,
    https://blog.selfshadow.com/publications/s2013-shading-course/).
    Sample loop is over ``sample_count`` Hammersley points, each iteration
    vectorised across the full (NdotV, roughness) grid at once.
    """
    eps = 1e-3
    ndotv = np.linspace(eps, 1.0, size)
    roughness = np.linspace(eps, 1.0, size)
    ndotv_grid, rough_grid = np.meshgrid(ndotv, roughness, indexing="ij")

    v_x = np.sqrt(1.0 - ndotv_grid**2)
    v_z = ndotv_grid

    a2 = rough_grid**4  # a = roughness^2, a2 = a^2 = roughness^4
    k = (rough_grid**2) / 2.0  # IBL variant of Schlick-GGX k

    xi0, xi1 = _hammersley(sample_count)

    a_acc = np.zeros_like(ndotv_grid)
    b_acc = np.zeros_like(ndotv_grid)

    for s in range(sample_count):
        phi = 2.0 * np.pi * xi0[s]
        cos_theta = np.sqrt((1.0 - xi1[s]) / (1.0 + (a2 - 1.0) * xi1[s]))
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, None))

        h_x = sin_theta * np.cos(phi)
        h_z = cos_theta

        v_dot_h = v_x * h_x + v_z * h_z
        v_dot_h = np.clip(v_dot_h, 0.0, None)

        l_z = 2.0 * v_dot_h * h_z - v_z  # L = 2*(V.H)*H - V, we only need L.z

        n_dot_l = np.clip(l_z, 0.0, None)
        n_dot_h = np.clip(h_z, 0.0, None)
        mask = n_dot_l > 0.0

        g1v = ndotv_grid / (ndotv_grid * (1.0 - k) + k)
        g1l = n_dot_l / (n_dot_l * (1.0 - k) + k)
        g = g1v * g1l

        denom = n_dot_h * ndotv_grid
        g_vis = np.where(
            denom > 1e-6, (g * v_dot_h) / np.where(denom > 1e-6, denom, 1.0), 0.0
        )
        fc = (1.0 - v_dot_h) ** 5

        a_acc += np.where(mask, (1.0 - fc) * g_vis, 0.0)
        b_acc += np.where(mask, fc * g_vis, 0.0)

    a_acc /= sample_count
    b_acc /= sample_count
    return np.stack([a_acc, b_acc], axis=-1)


# ---------------------------------------------------------------------------
# .npy caching
# ---------------------------------------------------------------------------
def load_or_compute_irradiance(faces: list[np.ndarray]) -> np.ndarray:
    """Stack of (6, out_size, out_size, 3) -- computed once, cached to
    ``irradiance_cache.npy`` next to this script thereafter."""
    if IRRADIANCE_CACHE.exists():
        return np.load(IRRADIANCE_CACHE)
    stacked = np.stack(generate_irradiance_map(faces))
    np.save(IRRADIANCE_CACHE, stacked)
    return stacked


def load_or_compute_brdf_lut() -> np.ndarray:
    """(BRDF_LUT_SIZE, BRDF_LUT_SIZE, 2) -- computed once, cached to
    ``brdf_lut_cache.npy`` next to this script thereafter."""
    if BRDF_LUT_CACHE.exists():
        return np.load(BRDF_LUT_CACHE)
    lut = compute_brdf_lut()
    np.save(BRDF_LUT_CACHE, lut)
    return lut
