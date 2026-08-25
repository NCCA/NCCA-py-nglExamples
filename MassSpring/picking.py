"""Colour-ID picking and the maths for dragging a mass.

Nothing here touches Qt or OpenGL so it can be tested headlessly. The
unproject follows RayPickingSelection/picking_maths.py -- copied rather than
imported, because the demos in this repo are meant to stand alone.
"""

import numpy as np

# The ID pass clears to black, so black has to mean "nothing here". Shifting
# every index up by one keeps index 0 pickable.
_ID_OFFSET = 1

# A ray whose direction is this close to perpendicular to the plane normal is
# parallel to the plane for our purposes, and never meets it.
_PARALLEL_EPSILON = 1e-8


def encode_id(index: int) -> tuple[int, int, int]:
    """Turn a mass index into a unique RGB colour for the ID pass."""
    value = index + _ID_OFFSET
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)


def decode_id(pixel) -> int | None:
    """Turn a pixel from the ID pass back into a mass index, or None for the
    background."""
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    value = r | (g << 8) | (b << 16)
    if value < _ID_OFFSET:
        return None
    return value - _ID_OFFSET


def ray_from_screen(
    x: float, y: float, width: int, height: int, mvp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a pixel position into a ray in the space `mvp` transforms from.

    x, y are pixel coordinates with Qt's top-left origin. Returns
    (origin, direction) with direction normalised.
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
    return near, direction


def intersect_plane(
    origin: np.ndarray, direction: np.ndarray, point: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """Where a ray meets a plane, or None if it runs parallel to it."""
    denominator = float(np.dot(direction, normal))
    if abs(denominator) < _PARALLEL_EPSILON:
        return None
    t = float(np.dot(point - origin, normal)) / denominator
    return origin + direction * t


def transform_point(p: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Transform a point by a 4x4, row-vector convention, dividing through by w."""
    v = np.array([p[0], p[1], p[2], 1.0]) @ mat
    return v[:3] / v[3]
