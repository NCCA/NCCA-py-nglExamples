"""Cube geometry and per-instance layout maths for the Instancing demo.

Deliberately numpy-only (no GL, no Qt, no wgpu) so the layout maths is unit
testable headless, and so the OpenGL and WebGPU versions of the demo place
their cubes identically. See ``Blending/blend_scene.py`` for the pattern this
follows.

Each instance record is 8 float32s: offset.xyz, a uniform scale, and
colour.rgba. That is deliberately *not* a mat4 -- a per-instance transform
would need four attribute locations (and four glVertexAttribDivisor calls)
for what a cube field only ever uses translation + uniform scale for. See the
README for the full argument.
"""

import colorsys

import numpy as np

RECORD_FLOATS = 8  # offset.x, offset.y, offset.z, scale, r, g, b, a

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees


def cube(size: float = 1.0) -> np.ndarray:
    """A unit cube (centred on the origin) as 36 flat-shaded triangle verts.

    Returns a flat float32 array of interleaved position (x,y,z) and normal
    (nx,ny,nz) -- stride 6 floats, no UVs, matching the fixed attribute
    locations 0/1 the instancing shaders read from. Two triangles per face,
    each face its own normal (no shared/averaged vertices), so lighting
    reads as a cube rather than a smoothed blob.
    """
    h = size * 0.5
    # (normal, 4 corners of the face in winding order for a +normal-facing
    # front, listed so the two triangles cover the face as 0,1,2 / 0,2,3)
    faces = [
        ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),  # +z
        ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),  # -z
        ((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),  # +x
        ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),  # -x
        ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),  # +y
        ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),  # -y
    ]
    order = (0, 1, 2, 0, 2, 3)
    verts = []
    for normal, corners in faces:
        for i in order:
            verts.append((*corners[i], *normal))
    return np.array(verts, dtype=np.float32).reshape(-1)


def _colour_wheel(n: int) -> np.ndarray:
    """n colours evenly spread around the hue wheel, RGBA in [0, 1]."""
    hues = np.arange(n, dtype=np.float64) / max(n, 1)
    rgb = np.array([colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in hues])
    return np.concatenate([rgb, np.ones((n, 1))], axis=1)


def golden_spiral(n: int, radius: float = 6.0, scale: float = 0.35) -> np.ndarray:
    """Sunflower-seed layout: n instances spiralling out to ``radius``.

    Points are placed by the golden-angle phyllotaxis construction (each
    point advances by the golden angle and sqrt(i/n) out from the centre),
    which is the standard way to spread points evenly over a disc without
    any two points ever running along a common radial line.

    Returns an (n, 8) float32 array: offset.xyz, scale, colour.rgba.
    """
    n = max(int(n), 1)
    i = np.arange(n, dtype=np.float64)
    r = radius * np.sqrt(i / n)
    theta = i * GOLDEN_ANGLE
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = np.zeros(n)

    data = np.empty((n, RECORD_FLOATS), dtype=np.float32)
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z
    data[:, 3] = scale
    data[:, 4:8] = _colour_wheel(n)
    return data


def grid(n: int, spacing: float = 1.5, scale: float = 0.35) -> np.ndarray:
    """n instances on a square-ish grid in the xz plane, centred on origin.

    The grid is sized to ceil(sqrt(n)) per side and any excess cells beyond
    n are simply not emitted, so non-square n (e.g. 17) still returns
    exactly n instances.

    Returns an (n, 8) float32 array: offset.xyz, scale, colour.rgba.
    """
    n = max(int(n), 1)
    side = int(np.ceil(np.sqrt(n)))
    ix, iz = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
    ix = ix.ravel()[:n]
    iz = iz.ravel()[:n]

    offset = (side - 1) * 0.5
    x = (ix - offset) * spacing
    z = (iz - offset) * spacing
    y = np.zeros(n)

    data = np.empty((n, RECORD_FLOATS), dtype=np.float32)
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z
    data[:, 3] = scale
    data[:, 4:8] = _colour_wheel(n)
    return data
