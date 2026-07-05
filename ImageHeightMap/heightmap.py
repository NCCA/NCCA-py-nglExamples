"""Builds a coloured terrain mesh from an image, sampling the red channel for
height and using RGB directly as vertex colour. Ported from
NGL9Demos/ImageHeightMap, but uses a plain triangle-list index buffer (one
glDrawElements call) instead of GL_PRIMITIVE_RESTART, and downsamples large
source images to keep vertex counts interactive-friendly.

Reports back to the caller: 200x200 max grid resolution regardless of source
image size, since the C++ original samples 1 vertex per source pixel (up to
1M+ vertices for MountainBig.bmp) which isn't necessary for a demo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def build_heightmap_mesh(
    image_path: str,
    width: float = 40.0,
    depth: float = 40.0,
    max_height: float = 4.0,
    max_resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    if img.width > max_resolution or img.height > max_resolution:
        img = img.resize((max_resolution, max_resolution))
    pixels = np.asarray(img, dtype=np.float32) / 255.0  # (h, w, 3)

    h, w = pixels.shape[0], pixels.shape[1]
    verts: list[float] = []
    for z in range(h):
        z_pos = -depth / 2 + depth * (z / (h - 1))
        for x in range(w):
            x_pos = -width / 2 + width * (x / (w - 1))
            r, g, b = pixels[z, x]
            y_pos = r * max_height
            verts.extend([x_pos, y_pos, z_pos, r, g, b])

    indices: list[int] = []
    for z in range(h - 1):
        for x in range(w - 1):
            top_left = z * w + x
            top_right = z * w + x + 1
            bottom_left = (z + 1) * w + x
            bottom_right = (z + 1) * w + x + 1
            indices.extend([top_left, bottom_left, top_right])
            indices.extend([top_right, bottom_left, bottom_right])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)
