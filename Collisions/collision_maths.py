"""Pure-maths collision-test helpers, ported from NGL9Demos/Collisions
(RaySphere, RayTriangle, SpherePlane, SphereSphere, BoundingBox).

Deliberately numpy-only (no GL/Qt/wgpu) so the collision maths is
unit-testable headless, mirroring RayPickingSelection/picking_maths.py's
pattern. These functions work directly on world-space points/vectors --
no matrix transforms are needed for analytic collision tests.
"""

from __future__ import annotations

import numpy as np

_DET_EPSILON = 1e-5


def ray_sphere_intersect(
    ray_start: np.ndarray,
    ray_dir: np.ndarray,
    sphere_pos: np.ndarray,
    radius: float,
) -> bool:
    """True if the ray (direction need not be normalised) hits the sphere
    ahead of its origin.

    Ported from NGL9Demos/Collisions/RaySphere's raySphere(): quadratic
    discriminant test. A tangent hit (discriminant == 0) counts as a miss,
    matching the C++'s `discrim <= 0.0 -> false`. Deliberately extended
    past the C++ with a check that the far root sits at t >= 0 -- the
    original tests the discriminant alone, which treats the ray as an
    infinite *line* and reports a sphere sitting entirely behind the
    origin as a hit; a genuine ray must not.
    """
    d = np.asarray(ray_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    p = np.asarray(ray_start, dtype=np.float64) - np.asarray(
        sphere_pos, dtype=np.float64
    )
    a = float(d @ d)
    b = 2.0 * float(d @ p)
    c = float(p @ p) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return False
    t_far = (-b + discriminant**0.5) / (2.0 * a)
    return t_far > 0.0


def ray_triangle_intersect(
    ray_start: np.ndarray,
    ray_end: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """Moller-Trumbore ray/triangle intersection, ported from
    NGL9Demos/Collisions/RayTriangle's rayTriangleIntersect().

    ray_start/ray_end define a finite ray segment (direction = end - start,
    not normalised, matching the C++). Returns (hit, hit_point); hit_point
    is None when hit is False.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    origin = np.asarray(ray_start, dtype=np.float64)
    direction = np.asarray(ray_end, dtype=np.float64) - origin

    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(edge1 @ pvec)
    if -_DET_EPSILON < det < _DET_EPSILON:
        return False, None
    inv_det = 1.0 / det

    tvec = origin - v0
    u = float(tvec @ pvec) * inv_det
    if u < -0.001 or u > 1.001:
        return False, None

    qvec = np.cross(tvec, edge1)
    v = float(direction @ qvec) * inv_det
    if v < -0.001 or u + v > 1.001:
        return False, None

    t = float(edge2 @ qvec) * inv_det
    if t <= 0.0:
        return False, None

    hit_point = origin + t * direction
    return True, hit_point.astype(np.float32)


def sphere_plane_collide(
    sphere_pos: np.ndarray,
    radius: float,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
    plane_width: float,
    plane_depth: float,
) -> bool:
    """True if the sphere has sunk a full diameter through the plane,
    within the plane's rectangular extent (width along x, depth along z,
    centred on plane_center). Ported from NGL9Demos/Collisions/SpherePlane's
    spherePlaneCollide(), generalised to a plane_center not at the world
    origin (the C++ only handles a plane through the origin -- see the
    plan's "deliberate deviations" note).

    The threshold matches the C++ literally: `D = normal.(pos - center) +
    radius; hit when D <= 0`, i.e. normal.(pos - center) <= -radius. That
    means merely touching the plane (surface distance == 0) is NOT a hit --
    the sphere's centre has to cross all the way to a full diameter past
    it. The C++'s own inline comment right above that check talks about a
    "BBox extent / 2" threshold that doesn't match the code at all
    (looks like a copy-paste from the BoundingBox demo), so this may well
    be an unintentional bug in the original -- but the plan calls for
    reproducing NGL9Demos's behaviour exactly, so it's kept as-is."""
    pos = np.asarray(sphere_pos, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    center = np.asarray(plane_center, dtype=np.float64)

    offset = float(normal @ (pos - center)) + radius
    if offset > 0.0:
        return False

    half_w = plane_width / 2.0
    half_d = plane_depth / 2.0
    return (
        center[0] - half_w < pos[0] < center[0] + half_w
        and center[2] - half_d < pos[2] < center[2] + half_d
    )


def sphere_sphere_collide(
    pos1: np.ndarray, radius1: float, pos2: np.ndarray, radius2: float
) -> bool:
    """True if two spheres overlap or touch. Ported from
    NGL9Demos/Collisions/SphereSphere's sphereSphereCollision(): squared-
    distance test against the sum of radii."""
    p1 = np.asarray(pos1, dtype=np.float64)
    p2 = np.asarray(pos2, dtype=np.float64)
    rel = p1 - p2
    dist_sq = float(rel @ rel)
    min_dist = radius1 + radius2
    return dist_sq <= min_dist * min_dist


_BBOX_FACE_NORMALS = (
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
)


def sphere_bbox_reflect(
    position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    half_extent: float,
) -> tuple[bool, np.ndarray]:
    """True/new-direction if a sphere moving inside a cube centred on the
    origin (half_extent along every axis) has crossed one of the cube's 6
    axis-aligned walls. Ported from NGL9Demos/Collisions/BoundingBox's
    BBoxCollision(): for each face normal, if position . normal + radius
    >= half_extent, reflect direction across that normal (in place, so a
    corner can reflect off two -- or three -- walls in one call, matching
    the C++'s unconditional loop over all 6 faces every frame)."""
    pos = np.asarray(position, dtype=np.float64)
    new_dir = np.asarray(direction, dtype=np.float64).copy()
    hit = False
    for normal in _BBOX_FACE_NORMALS:
        d = float(normal @ pos) + radius
        if d >= half_extent:
            new_dir = new_dir - 2.0 * float(new_dir @ normal) * normal
            hit = True
    return hit, new_dir
