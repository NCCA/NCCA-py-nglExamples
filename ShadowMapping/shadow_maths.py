"""Pure-maths helpers for the ShadowMapping demo: the light's view/projection.

Numpy-only (no GL/Qt) so the light-space transform can be unit tested
headless. This mirrors ``ncca.ngl.look_at``/``ncca.ngl.ortho`` (row-vector
convention: points transform as ``row_vector @ matrix``, translation lives
in row 3), but uses plain numpy arrays so the module has no dependency on
the GL-context-aware parts of the library.

Composition order note: because these are *raw* numpy matrices (not
``ncca.ngl.Mat4``, whose custom ``__matmul__`` reverses the operand order
to make ``project @ view @ model`` read like column-major maths), the
standard numpy rule applies here: ``row_vector @ A @ B`` applies ``A``
first, then ``B``. ``light_space_matrix`` therefore returns ``view @ proj``
so a world-space point is transformed by the light's view matrix first and
then projected -- the demo's ``main.py`` builds the equivalent matrix with
``ncca.ngl.Mat4`` objects as ``light_project @ light_view`` (reversed,
because of that library's operand-order convention) to get the same
result.
"""

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of v (or v unchanged if it is ~zero)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Row-vector view matrix (mirrors ``ncca.ngl.look_at``).

    Args:
        eye: Camera/light position, shape (3,).
        target: Point to look at, shape (3,).
        up: World up vector, shape (3,).

    Returns:
        4x4 float64 view matrix; translation lives in row 3.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    n = target - eye
    u = up
    v = np.cross(n, u)
    u = np.cross(v, n)
    n = _normalise(n)
    v = _normalise(v)
    u = _normalise(u)

    m = np.identity(4, dtype=np.float64)
    m[0, 0], m[1, 0], m[2, 0] = v
    m[0, 1], m[1, 1], m[2, 1] = u
    m[0, 2], m[1, 2], m[2, 2] = -n
    m[3, 0] = -np.dot(eye, v)
    m[3, 1] = -np.dot(eye, u)
    m[3, 2] = np.dot(eye, n)
    return m


def ortho(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Row-vector orthographic projection, OpenGL [-1, 1] NDC depth range.

    Mirrors ``ncca.ngl.ortho`` with the default ``PerspMode.OpenGL``.
    """
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[3, 0] = -(right + left) / (right - left)
    m[3, 1] = -(top + bottom) / (top - bottom)
    m[3, 2] = -(far + near) / (far - near)
    return m


def light_space_matrix(
    light_pos: np.ndarray,
    target: np.ndarray,
    ortho_extents: float,
    near: float = 0.1,
    far: float = 20.0,
    up: np.ndarray = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Combined light view-projection matrix used for shadow mapping.

    Models a directional light as an orthographic "camera" sitting at
    ``light_pos`` and looking at ``target``. ``ortho_extents`` is the
    half-width/half-height of its orthographic frustum in world units.

    Returns:
        4x4 float64 matrix such that, following the row-vector convention,
        ``point_h @ light_space_matrix`` (with ``point_h`` a homogeneous
        world-space point) yields clip space directly.
    """
    view = look_at(np.asarray(light_pos, dtype=np.float64), target, up)
    proj = ortho(
        -ortho_extents, ortho_extents, -ortho_extents, ortho_extents, near, far
    )
    return view @ proj


def project_to_shadow_uv(
    point: np.ndarray, light_space: np.ndarray
) -> tuple[float, float, float]:
    """Project a world-space point through ``light_space`` into shadow-map space.

    Returns ``(u, v, depth)`` where u, v are texture coordinates in [0, 1]
    (only meaningful if the point is inside the light frustum) and depth is
    the OpenGL NDC depth remapped to [0, 1], matching what a fragment
    shader compares against ``texture(shadowMap, uv).r``.
    """
    p = np.asarray(point, dtype=np.float64)
    p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    clip = p_h @ light_space
    ndc = clip[:3] / clip[3]
    u, v = ndc[0] * 0.5 + 0.5, ndc[1] * 0.5 + 0.5
    depth = ndc[2] * 0.5 + 0.5
    return float(u), float(v), float(depth)
