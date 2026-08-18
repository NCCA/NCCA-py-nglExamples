"""Screen-to-world unprojection, ported from NGL9Demos/ViewToWorldTransform.

Pure numpy, no GL/Qt/wgpu imports, so this is unit-testable headless and
shared unchanged between the OpenGL and WebGPU entry points of this demo.
All matrices follow the PyNGL row-vector convention: points transform as
``row_vector @ matrix``.
"""

import numpy as np


def unproject_point(
    x: float,
    y: float,
    width: int,
    height: int,
    view_projection: np.ndarray,
    ndc_z: float = 1.0,
) -> np.ndarray:
    """Unproject a screen pixel at a fixed NDC depth into world space.

    x, y are pixel coordinates with Qt's top-left origin. view_projection is
    ``(projection @ view).to_numpy()`` (no model term, so the result is a
    world-space point directly). ndc_z selects the depth plane in OpenGL NDC
    (-1 near .. +1 far); the demo defaults to the far plane (ndc_z=1.0),
    matching NGL9Demos' ``ngl::unProject(Vec3(x, y, 1.0f), ...)`` call.
    """
    ndc_x = 2.0 * x / width - 1.0
    ndc_y = 1.0 - 2.0 * y / height
    inverse = np.linalg.inv(view_projection.astype(np.float64))
    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0]) @ inverse
    return (clip[:3] / clip[3]).astype(np.float32)
