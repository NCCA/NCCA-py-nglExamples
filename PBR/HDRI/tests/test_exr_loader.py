import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from exr_loader import load_equirect_hdr  # noqa: E402

HDRI = (
    Path(__file__).resolve().parents[1] / "images" / "historic_cloister_passage_1k.exr"
)


def test_loads_rgb_float32_2to1_equirect():
    img = load_equirect_hdr(HDRI)
    assert img.dtype == np.float32
    assert img.ndim == 3 and img.shape[2] == 3
    h, w = img.shape[:2]
    assert w == 2 * h  # equirectangular maps are 2:1


def test_preserves_hdr_range():
    img = load_equirect_hdr(HDRI)
    # A real HDRI has bright spots well above the LDR 1.0 ceiling.
    assert img.max() > 1.5
    assert img.min() >= 0.0
    assert np.isfinite(img).all()
