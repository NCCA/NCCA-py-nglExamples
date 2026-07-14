"""Numpy mirror of the signed-distance-field (SDF) functions used by the ray
marcher. No GL / Qt / wgpu imports here on purpose: this module exists so
the maths behind the shader can be unit tested on the CPU, and so a reader
can check a function here against its GLSL/WGSL twin without a GPU.

Every function has the same name in three places:

    sdf_maths.py             -- numpy, tested (this file)
    shaders/RayMarchFragment.glsl -- OpenGL, in the `RayMarching` demo
    RayMarch.wgsl                 -- WebGPU, same demo

They are written as line-for-line transcriptions of each other. If you
change the scene composition in one, change it in all three or the two
renderers will visibly disagree.
"""

import numpy as np

# Scene layout constants, shared with both shaders so the CPU-side tests
# describe the same world the GPU draws.
PLANE_HEIGHT = 0.0
SPHERE_CENTRE = np.array([-0.9, 0.9, 0.0])
SPHERE_RADIUS = 0.9
BOX_CENTRE = np.array([0.9, 0.6, 0.0])
BOX_HALF_EXTENTS = np.array([0.6, 0.6, 0.6])
TORUS_CENTRE = np.array([0.0, 0.4, -1.4])
TORUS_MAJOR = 0.7
TORUS_MINOR = 0.22
MOVING_SPHERE_RADIUS = 0.5
MOVING_SPHERE_ORBIT_RADIUS = 1.6
MOVING_SPHERE_HEIGHT = 1.1


def sd_sphere(p: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """Distance from `p` to a sphere of `radius` centred at `centre`.

    Negative inside, zero on the surface, positive outside -- the SDF sign
    convention every function in this file follows.
    """
    return float(np.linalg.norm(p - centre) - radius)


def sd_box(p: np.ndarray, half_extents: np.ndarray) -> float:
    """Distance from `p` to an axis-aligned box centred at the origin.

    Standard Inigo Quilez box SDF: clamp the per-axis penetration to zero
    to get the "outside" component, and take the least-negative axis for
    the "inside" component.
    """
    q = np.abs(p) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], max(q[1], q[2])), 0.0)
    return float(outside + inside)


def sd_torus(p: np.ndarray, major_radius: float, minor_radius: float) -> float:
    """Distance from `p` to a torus lying in the XZ plane, centred at the
    origin: `major_radius` is the ring radius, `minor_radius` the tube
    radius.
    """
    q = np.array(
        [np.hypot(p[0], p[2]) - major_radius, p[1]],
        dtype=np.float64,
    )
    return float(np.linalg.norm(q) - minor_radius)


def sd_plane(p: np.ndarray, normal: np.ndarray, height: float) -> float:
    """Distance from `p` to an infinite plane with the given unit `normal`,
    offset `height` along that normal from the origin.
    """
    return float(np.dot(p, normal) - height)


def smooth_min(a: float, b: float, k: float) -> float:
    """Polynomial smooth minimum (Inigo Quilez's smin): blends two distance
    fields together so their union has a rounded fillet instead of a hard
    seam. `k` is the blend radius -- 0 recovers the ordinary `min`, and the
    result always stays at or below `min(a, b)`.
    """
    if k <= 0.0:
        return min(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def _moving_sphere_centre(time: float) -> np.ndarray:
    """The one animated primitive: a sphere orbiting above the rest of the
    scene, melting through the plane/sphere/box/torus blend as it passes
    overhead (see scene()).
    """
    angle = time
    x = MOVING_SPHERE_ORBIT_RADIUS * np.cos(angle)
    z = MOVING_SPHERE_ORBIT_RADIUS * np.sin(angle)
    return np.array([x, MOVING_SPHERE_HEIGHT, z])


def scene(p: np.ndarray, time: float, k: float) -> float:
    """The full scene distance field at point `p`: a ground plane, then a
    sphere/box/torus group smooth-blended together with blend radius `k`,
    and one moving sphere smooth-blended into that group so it looks like
    it melts through the other shapes as it orbits.
    """
    d_plane = sd_plane(p, np.array([0.0, 1.0, 0.0]), PLANE_HEIGHT)
    d_sphere = sd_sphere(p, SPHERE_CENTRE, SPHERE_RADIUS)
    d_box = sd_box(p - BOX_CENTRE, BOX_HALF_EXTENTS)
    d_torus = sd_torus(p - TORUS_CENTRE, TORUS_MAJOR, TORUS_MINOR)
    d_moving = sd_sphere(p, _moving_sphere_centre(time), MOVING_SPHERE_RADIUS)

    d = smooth_min(d_sphere, d_box, k)
    d = smooth_min(d, d_torus, k)
    d = smooth_min(d, d_moving, k)
    return min(d_plane, d)


def estimate_normal(sdf, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Surface normal at `p` via the central-difference gradient of an SDF
    (any callable p -> float, not just scene() -- the tests exercise it
    against a bare sphere so the numerics can be checked in isolation).
    """
    dx = np.array([eps, 0.0, 0.0])
    dy = np.array([0.0, eps, 0.0])
    dz = np.array([0.0, 0.0, eps])
    n = np.array(
        [
            sdf(p + dx) - sdf(p - dx),
            sdf(p + dy) - sdf(p - dy),
            sdf(p + dz) - sdf(p - dz),
        ]
    )
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return n / norm
