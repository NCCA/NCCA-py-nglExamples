"""
Shared scene description and sorting maths for the blending demos.

This module is deliberately numpy-only (no GL, no Qt, no wgpu) so the
depth-sorting maths can be unit tested headless, and so the OpenGL and
WebGPU versions of the demo render exactly the same scene.

All matrices follow the PyNGL row-vector convention: points transform as
``row_vector @ matrix`` on the numpy side (see RayPickingSelection for the
same pattern).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panel:
    """A single transparent quad in the scene."""

    name: str
    colour: tuple[float, float, float]
    position: tuple[float, float, float]


# Five overlapping panels straddling the opaque teapot. They are staggered
# in x so the overlap chain is visible, and spread in z so draw order
# matters from the default camera.
PANELS: tuple[Panel, ...] = (
    Panel("red", (1.00, 0.15, 0.15), (-1.2, 1.0, -1.6)),
    Panel("orange", (1.00, 0.55, 0.10), (-0.6, 1.0, -0.8)),
    Panel("green", (0.15, 0.80, 0.20), (0.0, 1.0, 0.0)),
    Panel("blue", (0.20, 0.40, 1.00), (0.6, 1.0, 0.8)),
    Panel("purple", (0.70, 0.20, 0.90), (1.2, 1.0, 1.6)),
)

PANEL_SIZE = 1.8
DEFAULT_ALPHA = 0.45


def view_space_z(model_view: np.ndarray, point=(0.0, 0.0, 0.0)) -> float:
    """Depth of a local-space point in view space.

    model_view is a 4x4 numpy matrix in the row-vector convention. In view
    space the camera looks down -z, so a *more negative* result is *further*
    from the camera.
    """
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return float((p @ model_view.astype(np.float64))[2])


def back_to_front(model_views: list[np.ndarray]) -> list[int]:
    """Indices of the given model-view matrices sorted furthest first.

    This is the order transparent geometry must be drawn in for the standard
    OVER operator (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) to composite correctly.
    Sorting ascending by view-space z puts the most negative (furthest)
    objects first.
    """
    return sorted(range(len(model_views)), key=lambda i: view_space_z(model_views[i]))
