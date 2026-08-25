"""
Shared scene description and reference maths for the weighted blended
order-independent transparency (OIT) demos.

This module is deliberately numpy-only (no GL, no Qt, no wgpu) so the OIT
maths can be unit tested headless: ``weight`` and ``composite_pixel`` are a
CPU reference implementation of exactly what the accumulation and composite
shaders compute, which lets a test assert the key property of the technique
-- the result does not depend on fragment order.

Reference: McGuire & Bavoil, "Weighted Blended Order-Independent
Transparency", JCGT 2013. http://jcgt.org/published/0002/02/09/

All matrices follow the PyNGL row-vector convention: points transform as
``row_vector @ matrix`` on the numpy side.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panel:
    """A single transparent quad in the scene."""

    name: str
    colour: tuple[float, float, float]
    position: tuple[float, float, float]
    rotate_y: float  # degrees


# The red/blue pair intersect in an X through the teapot: no per-object
# sort can draw them correctly, which is the motivating case for OIT.
# The green/orange panels are ordinary front/back panels so mode 2
# (sorted) still visibly fixes *something* relative to mode 1 (unsorted).
PANELS: tuple[Panel, ...] = (
    Panel("red", (1.00, 0.15, 0.15), (0.0, 1.0, 0.0), 45.0),
    Panel("blue", (0.20, 0.40, 1.00), (0.0, 1.0, 0.0), -45.0),
    Panel("green", (0.15, 0.80, 0.20), (-0.8, 1.0, -2.2), 0.0),
    Panel("orange", (1.00, 0.55, 0.10), (0.8, 1.0, 2.2), 0.0),
)

PANEL_SIZE = 2.4
DEFAULT_ALPHA = 0.45


# ----------------------------------------------------------------------
# weighted blended OIT reference implementation
# ----------------------------------------------------------------------
def weight(view_z: float, alpha: float) -> float:
    """The depth weight from McGuire & Bavoil (a tuned variant of eq. 9).

    view_z is the view-space z of the fragment (negative in front of the
    camera in GL conventions; the magnitude is what matters). Fragments
    close to the camera get large weights, distant ones small weights, so
    near surfaces dominate the weighted average exactly as they would have
    dominated a correctly sorted OVER composite.
    """
    z = abs(view_z)
    w = 10.0 / (1e-5 + (z / 5.0) ** 2 + (z / 200.0) ** 6)
    return alpha * float(np.clip(w, 1e-2, 3e3))


def composite_pixel(opaque_rgb, fragments) -> np.ndarray:
    """CPU reference of the accumulate + composite passes for one pixel.

    opaque_rgb is the colour already in the opaque buffer, fragments is an
    iterable of (rgb, alpha, view_z) for every transparent fragment covering
    the pixel, *in any order*.

    Accumulation (what the two render targets store):
        accum  += vec4(rgb * a, a) * weight(z, a)     (blend ONE, ONE)
        reveal *= (1 - a)                             (blend ZERO, 1-SRC)

    Composite:
        transparent = accum.rgb / max(accum.a, eps)
        final = opaque * reveal + transparent * (1 - reveal)

    Both accumulators are commutative (a sum and a product), which is the
    whole trick: the result cannot depend on draw order.
    """
    accum = np.zeros(4, dtype=np.float64)
    reveal = 1.0
    for rgb, alpha, view_z in fragments:
        w = weight(view_z, alpha)
        accum[:3] += np.asarray(rgb, dtype=np.float64) * alpha * w
        accum[3] += alpha * w
        reveal *= 1.0 - alpha
    opaque = np.asarray(opaque_rgb, dtype=np.float64)
    if accum[3] < 1e-5:
        return opaque
    transparent = accum[:3] / max(accum[3], 1e-5)
    return opaque * reveal + transparent * (1.0 - reveal)


def over_composite(opaque_rgb, fragments_back_to_front) -> np.ndarray:
    """Ground-truth OVER composite of correctly sorted fragments, used by
    the tests as the reference OIT should approximate."""
    colour = np.asarray(opaque_rgb, dtype=np.float64)
    for rgb, alpha, _ in fragments_back_to_front:
        colour = np.asarray(rgb, dtype=np.float64) * alpha + colour * (1.0 - alpha)
    return colour


# ----------------------------------------------------------------------
# per-object sorting (mode 2 of the demo)
# ----------------------------------------------------------------------
def view_space_z(model_view: np.ndarray, point=(0.0, 0.0, 0.0)) -> float:
    """Depth of a local-space point in view space (row-vector convention)."""
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return float((p @ model_view.astype(np.float64))[2])


def back_to_front(model_views: list[np.ndarray]) -> list[int]:
    """Indices sorted furthest first (most negative view-space z first)."""
    return sorted(range(len(model_views)), key=lambda i: view_space_z(model_views[i]))
