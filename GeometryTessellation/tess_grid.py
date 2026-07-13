"""Numpy-only maths for the tessellated-plane demo (no GL/Qt imports).

Two pure-maths pieces live here so they can be pytest-covered headlessly:

    build_patch_grid(...)  -- lays out an N x N grid of 4-vertex GL_PATCHES
                               (quad control points, non-indexed) in the xz
                               plane, ready to upload as a flat vertex buffer.
    tess_level_from_distance(...) -- the same distance -> tessellation-level
                               curve used by the TCS, so the LOD policy is
                               testable without a GL context.

The GLSL fbm() used to displace the surface is deliberately *not*
reimplemented here: it only ever runs on the GPU (inside the tessellation
evaluation shader) and duplicating noise functions across numpy/GLSL is a
maintenance trap, not a teaching point.
"""

from __future__ import annotations

import numpy as np


def build_patch_grid(resolution: int, size: float) -> np.ndarray:
    """Build a flat (N*N*4, 3) float32 array of quad-patch control points.

    Args:
        resolution: number of patches per side (e.g. 16 -> 16x16 patches).
        size: total width/depth of the grid, centred on the origin.

    Returns:
        A (resolution*resolution*4, 3) float32 array. Every consecutive
        group of 4 rows is one patch's corners in the order
        (x0,z0) (x1,z0) (x1,z1) (x0,z1) -- i.e. counter-clockwise looking
        down the -y axis -- which is what GL_PATCHES with GL_PATCH_VERTICES=4
        expects to feed to a quads-domain tessellation evaluation shader.
    """
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    half = size * 0.5
    step = size / resolution
    verts = np.empty((resolution * resolution * 4, 3), dtype=np.float32)

    idx = 0
    for j in range(resolution):
        z0 = -half + j * step
        z1 = z0 + step
        for i in range(resolution):
            x0 = -half + i * step
            x1 = x0 + step
            verts[idx + 0] = (x0, 0.0, z0)
            verts[idx + 1] = (x1, 0.0, z0)
            verts[idx + 2] = (x1, 0.0, z1)
            verts[idx + 3] = (x0, 0.0, z1)
            idx += 4

    return verts


def patch_count(resolution: int) -> int:
    """Number of GL_PATCHES (each 4 control-point vertices) in the grid."""
    return resolution * resolution


def tess_level_from_distance(
    distance: float,
    near_distance: float,
    far_distance: float,
    min_level: float = 1.0,
    max_level: float = 64.0,
) -> float:
    """Map camera distance to a tessellation level, clamped to [min, max].

    Mirrors the TCS's distance-based LOD: close geometry gets max_level,
    far geometry decays linearly (in distance) down to min_level. This is
    the same policy expressed once so it can be pytest-checked without a
    GL context; the shader re-implements it per-patch-corner in GLSL.
    """
    if far_distance <= near_distance:
        raise ValueError("far_distance must be > near_distance")

    t = (distance - near_distance) / (far_distance - near_distance)
    t = min(max(t, 0.0), 1.0)
    level = max_level + t * (min_level - max_level)
    return min(max(level, min_level), max_level)
