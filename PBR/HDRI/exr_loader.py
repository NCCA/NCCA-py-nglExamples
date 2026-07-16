"""Read an equirectangular OpenEXR HDRI into a numpy float32 RGB array.

This is the one part of the HDRI demo with no GL or Qt, so it is the one part
that gets a headless unit test. Everything downstream is on the GPU.
"""

from __future__ import annotations

from pathlib import Path

import Imath
import numpy as np
import OpenEXR


def load_equirect_hdr(path: str | Path) -> np.ndarray:
    """Load an equirectangular ``.exr`` as an ``(H, W, 3)`` float32 RGB array.

    Parameters
    ----------
        path : str | Path
            the ``.exr`` file to read

    Returns
    -------
        np.ndarray
            shape ``(H, W, 3)``, dtype ``float32``, RGB order, HDR range preserved
    """
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channels(["R", "G", "B"], pixel_type)
    channels = [np.frombuffer(c, dtype=np.float32).reshape(height, width) for c in raw]
    return np.ascontiguousarray(np.stack(channels, axis=-1))
