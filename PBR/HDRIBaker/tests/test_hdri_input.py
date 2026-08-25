import numpy as np
import pytest
from hdri_input import decode_rgbe, load_equirect_hdr


def _make_rle_hdr() -> bytes:
    """An 8x1 Radiance .hdr, new-style RLE, every pixel RGBE=(128,128,128,128).

    scale = 2**(128-136) = 1/256, so every channel decodes to 128/256 = 0.5.
    """
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n"
    # new-RLE scanline header: 2, 2, width_hi, width_lo
    scan = bytes([2, 2, 0, 8])
    # four channels R,G,B,E: run code 128+8=136 means "run of 8 of the next byte"
    for _ in range(4):
        scan += bytes([136, 128])
    return header + scan


def test_decode_rgbe_rle_scanline():
    img = decode_rgbe(_make_rle_hdr())
    assert img.shape == (1, 8, 3)
    assert img.dtype == np.float32
    np.testing.assert_allclose(img, 0.5, atol=1e-6)


def test_decode_rgbe_zero_exponent_is_black():
    # width 4 (<8) forces old flat format: 4 bytes RGBE per pixel, E=0 -> black
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n"
    body = bytes([200, 200, 200, 0]) * 4
    img = decode_rgbe(header + body)
    assert img.shape == (1, 4, 3)
    np.testing.assert_allclose(img, 0.0)


def test_load_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "panorama.png"
    bad.write_bytes(b"not hdr")
    with pytest.raises(ValueError, match="unsupported"):
        load_equirect_hdr(bad)


def test_load_hdr_dispatch(tmp_path):
    p = tmp_path / "flat.hdr"
    p.write_bytes(_make_rle_hdr())
    img = load_equirect_hdr(p)
    assert img.shape == (1, 8, 3)
    np.testing.assert_allclose(img, 0.5, atol=1e-6)
