"""Procedural cubemap sky generation -- numpy only, no GL/Qt/wgpu imports.

Copied verbatim from ``SkyBoxEnvMap/cubemap_gen.py`` (demo folders stay
standalone, no cross-folder imports) -- see that demo for the skybox/env-map
rendering this environment was originally built for. IBL reuses it here as
the source environment its irradiance map and BRDF split-sum are
precomputed from.

Generates six RGBA8 faces (a horizon-banded gradient sky with a ground plane
and a sun disc baked into the +z face) so the demo needs no external image
assets. Faces are returned in **OpenGL cubemap face order**:
``+x, -x, +y, -y, +z, -z`` -- both the GL and WebGPU demos upload them in
this order (GL as ``GL_TEXTURE_CUBE_MAP_POSITIVE_X + i``, WebGPU as array
layers of a ``dimension="cube"`` texture view).

Per-face UV -> direction mapping. Each face is parameterised by
``u, v in [-1, 1]`` (inclusive, so edge pixels land exactly on the cube
edges -- important for the horizon-continuity test below):

    +x: (1,  v,  u)      -x: (-1, v, -u)
    +y: (u,  1, -v)      -y: ( u, -1, v)
    +z: (-u, v,  1)      -z: ( u,  v, -1)

This mapping is chosen so that the shared cube edge x=1, z=1 (the last
column of +x and the first column of +z) resolves to the *same* direction
vector from both faces -- the classic cubemap face-order/orientation bug
would break that equality, which is exactly what the horizon-continuity
test checks.
"""

from __future__ import annotations

import numpy as np

FACE_SIZE = 256
FACE_ORDER: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")

_SKY_ZENITH = np.array([0.25, 0.45, 0.85])
_SKY_HORIZON = np.array([0.75, 0.82, 0.95])
_GROUND_COLOUR = np.array([0.35, 0.28, 0.22])

_SUN_DIR = np.array([0.0, 0.35, 1.0])
_SUN_DIR = _SUN_DIR / np.linalg.norm(_SUN_DIR)
_SUN_COLOUR = np.array([1.0, 0.97, 0.85])
_SUN_COS_THRESHOLD = 0.985  # ~ half-angle of ~10 degrees


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _face_directions(face: str, size: int) -> np.ndarray:
    """Return an (size, size, 3) array of unnormalised direction vectors for
    the given face, indexed [row, col]."""
    coord = np.linspace(-1.0, 1.0, size)
    u, v = np.meshgrid(coord, coord)  # u varies along columns, v along rows

    if face == "+x":
        dirs = np.stack([np.ones_like(u), v, u], axis=-1)
    elif face == "-x":
        dirs = np.stack([-np.ones_like(u), v, -u], axis=-1)
    elif face == "+y":
        dirs = np.stack([u, np.ones_like(u), -v], axis=-1)
    elif face == "-y":
        dirs = np.stack([u, -np.ones_like(u), v], axis=-1)
    elif face == "+z":
        dirs = np.stack([-u, v, np.ones_like(u)], axis=-1)
    elif face == "-z":
        dirs = np.stack([u, v, -np.ones_like(u)], axis=-1)
    else:
        raise ValueError(f"unknown face {face!r}")
    return dirs


def _sky_colour(direction: np.ndarray) -> np.ndarray:
    """Horizon-banded gradient colour for a (..., 3) array of *normalised*
    direction vectors, blending ground / horizon / zenith by elevation."""
    elevation = direction[..., 1]
    above = elevation >= 0.0

    t_up = _smoothstep(0.0, 0.6, elevation)[..., None]
    colour_up = _SKY_HORIZON * (1.0 - t_up) + _SKY_ZENITH * t_up

    t_down = _smoothstep(0.0, 0.4, -elevation)[..., None]
    colour_down = _SKY_HORIZON * (1.0 - t_down) + _GROUND_COLOUR * t_down

    return np.where(above[..., None], colour_up, colour_down)


def _apply_sun(colour: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Blend a soft-edged sun disc into ``colour`` where ``direction`` is
    close to the fixed sun direction."""
    cos_angle = direction @ _SUN_DIR
    glow = _smoothstep(_SUN_COS_THRESHOLD, _SUN_COS_THRESHOLD + 0.01, cos_angle)
    return colour * (1.0 - glow[..., None]) + _SUN_COLOUR * glow[..., None]


def generate_cubemap_faces(size: int = FACE_SIZE) -> list[np.ndarray]:
    """Generate the six cubemap faces as a list of ``(size, size, 4)``
    ``uint8`` RGBA arrays, in :data:`FACE_ORDER`."""
    faces = []
    for face in FACE_ORDER:
        directions = _face_directions(face, size)
        normalised = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        colour = _sky_colour(normalised)
        if face == "+z":
            colour = _apply_sun(colour, normalised)
        rgb = np.clip(colour, 0.0, 1.0) * 255.0
        rgba = np.empty((size, size, 4), dtype=np.uint8)
        rgba[..., :3] = rgb.astype(np.uint8)
        rgba[..., 3] = 255
        faces.append(rgba)
    return faces
