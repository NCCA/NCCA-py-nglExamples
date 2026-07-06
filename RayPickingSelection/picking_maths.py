"""
Pure-maths picking helpers: ray casting and screen-space distance tests.

This module is deliberately numpy-only (no GL, no Qt) so the picking maths
can be unit tested headless. Two families of helpers live here:

Ray casting (object selection)
    ``ray_from_screen`` unprojects a mouse position into a ray, then
    ``intersect_sphere`` (broad phase) and ``intersect_triangles``
    (vectorised Moller-Trumbore narrow phase) find what it hits. Unlike
    colour-ID picking there is no extra render pass and no GPU readback --
    picking is pure CPU maths, and it returns the hit *distance* for free.

Screen-space tests (gizmo handles)
    ``point_segment_distance`` and ``point_polyline_distance`` measure the
    pixel distance from the mouse to a projected handle, which is how real
    DCCs (Maya, Blender) hit-test gizmos: the click tolerance is a clean
    radius in pixels rather than a block of pixels read back from an ID
    buffer.

All matrices follow the PyNGL row-vector convention: points transform as
``row_vector @ matrix`` on the numpy side.
"""

import numpy as np

# t values closer than this are behind (or on) the ray origin and ignored
_T_EPSILON = 1e-6
# determinant magnitude below this means the ray is parallel to the triangle
_DET_EPSILON = 1e-8


# ----------------------------------------------------------------------
# ray construction
# ----------------------------------------------------------------------
def ray_from_screen(
    x: float, y: float, width: int, height: int, mvp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a pixel position into a ray in the space mvp transforms from.

    x, y are pixel coordinates with Qt's top-left origin. mvp is the full
    4x4 (projection @ view @ model) matrix as numpy; the ray is returned in
    that model space, so folding the scene's global transform into mvp gives
    a scene-space ray directly.

    Returns (origin, direction) with direction normalised.
    """
    ndc_x = 2.0 * x / width - 1.0
    ndc_y = 1.0 - 2.0 * y / height  # flip: NDC y is up, pixel y is down
    inverse = np.linalg.inv(mvp.astype(np.float64))
    # OpenGL NDC z runs -1 (near) to +1 (far)
    near = np.array([ndc_x, ndc_y, -1.0, 1.0]) @ inverse
    far = np.array([ndc_x, ndc_y, 1.0, 1.0]) @ inverse
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    direction = far - near
    direction /= np.linalg.norm(direction)
    return near.astype(np.float32), direction.astype(np.float32)


def transform_ray(
    origin: np.ndarray, direction: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a ray by a 4x4 matrix (row-vector convention).

    The direction transforms with w=0 and is *not* re-normalised: keeping its
    length means a parameter t on the transformed ray measures the same
    distance as t on the original ray, so hit distances compare directly
    across objects.
    """
    o = np.array([origin[0], origin[1], origin[2], 1.0]) @ matrix
    d = np.array([direction[0], direction[1], direction[2], 0.0]) @ matrix
    return o[:3], d[:3]


# ----------------------------------------------------------------------
# intersection tests
# ----------------------------------------------------------------------
def intersect_sphere(
    origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float
) -> bool:
    """True if the ray hits the sphere (broad-phase test; direction need not
    be normalised)."""
    oc = np.asarray(origin, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    a = float(d @ d)
    if a < _DET_EPSILON:
        return False
    b = 2.0 * float(oc @ d)
    c = float(oc @ oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False
    # at least one root must be in front of the origin
    root = np.sqrt(discriminant)
    return (-b + root) / (2.0 * a) > _T_EPSILON


def intersect_triangles(
    origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray
) -> float | None:
    """Nearest hit parameter t of a ray against a (N, 3, 3) triangle array.

    Vectorised Moller-Trumbore over every triangle at once; returns the
    smallest positive t, or None if nothing is hit. Triangles are tested
    double-sided (no backface culling) so picking works from any angle.
    """
    tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(d, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > _DET_EPSILON
    # avoid divide-by-zero warnings on degenerate / parallel triangles
    safe_det = np.where(valid, det, 1.0)
    inv_det = 1.0 / safe_det

    tvec = o - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = (qvec @ d) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det

    hits = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > _T_EPSILON)
    if not hits.any():
        return None
    return float(t[hits].min())


def bounding_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of a simple AABB-centred bounding sphere."""
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    centre = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    radius = float(np.linalg.norm(verts - centre, axis=1).max())
    return centre.astype(np.float32), radius


# ----------------------------------------------------------------------
# screen-space distances (gizmo handle hit tests)
# ----------------------------------------------------------------------
def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from 2D point p to the segment a-b."""
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < _DET_EPSILON:
        return float(np.linalg.norm(p - a))
    t = np.clip(float((p - a) @ ab) / length_sq, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def point_polyline_distance(p: np.ndarray, points: np.ndarray, closed: bool) -> float:
    """Distance from 2D point p to a polyline given as an (N, 2) array."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    best = np.inf
    count = len(pts) if closed else len(pts) - 1
    for i in range(count):
        best = min(best, point_segment_distance(p, pts[i], pts[(i + 1) % len(pts)]))
    return best
