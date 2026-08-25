"""Read an equirectangular HDRI (.exr or .hdr) into a numpy float32 RGB array.

This is the headless, GPU-free front of the baker, so it is the part that
gets real unit tests. The .exr path reuses the OpenEXR reader the HDRI demo
already relies on; the .hdr path is a small built-in Radiance RGBE decoder so
we do not pull in another image library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_equirect_hdr(path: str | Path) -> np.ndarray:
    """Load an equirectangular ``.exr`` or ``.hdr`` as ``(H, W, 3)`` float32 RGB.

    HDR range is preserved; the suffix picks the decoder.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".exr":
        return _load_exr(path)
    if suffix == ".hdr":
        return decode_rgbe(path.read_bytes())
    raise ValueError(f"unsupported HDRI format {suffix!r} (use .exr or .hdr)")


def _load_exr(path: Path) -> np.ndarray:
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(str(path))
    window = exr.header()["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channels(["R", "G", "B"], pixel_type)
    channels = [np.frombuffer(c, dtype=np.float32).reshape(height, width) for c in raw]
    return np.ascontiguousarray(np.stack(channels, axis=-1))


def _rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 4)`` uint8 RGBE array to ``(..., 3)`` float32 RGB."""
    rgb = rgbe[..., :3].astype(np.float32)
    exponent = rgbe[..., 3].astype(np.int32)
    # value = mantissa * 2**(E - 128 - 8); E == 0 is the encoded black.
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0)
    return (rgb * scale[..., None].astype(np.float32)).astype(np.float32)


def decode_rgbe(data: bytes) -> np.ndarray:
    """Decode Radiance ``.hdr`` (RGBE) bytes to an ``(H, W, 3)`` float32 array."""
    pos = data.find(b"\n\n")
    if pos < 0 or not data.startswith(b"#?"):
        raise ValueError("not a Radiance .hdr file")
    body = data[pos + 2 :]

    # Resolution line, e.g. "-Y 512 +X 1024"; we only handle the common
    # top-down, left-to-right orientation used by every panorama we bake.
    eol = body.find(b"\n")
    res = body[:eol].split()
    body = body[eol + 1 :]
    if res[0] != b"-Y" or res[2] != b"+X":
        raise ValueError(f"unsupported HDR orientation {res!r}")
    height, width = int(res[1]), int(res[3])

    rgbe = np.zeros((height, width, 4), dtype=np.uint8)
    offset = 0
    for y in range(height):
        offset = _decode_scanline(body, offset, rgbe[y], width)
    return _rgbe_to_float(rgbe)


def _decode_scanline(body: bytes, offset: int, row: np.ndarray, width: int) -> int:
    """Fill one ``(width, 4)`` scanline from ``body`` at ``offset``; return the
    new offset. Handles both new-style adaptive RLE and old flat RGBE."""
    new_rle = (
        8 <= width < 0x8000
        and body[offset] == 2
        and body[offset + 1] == 2
        and (body[offset + 2] << 8 | body[offset + 3]) == width
    )
    if not new_rle:
        flat = np.frombuffer(body, dtype=np.uint8, count=width * 4, offset=offset)
        row[:] = flat.reshape(width, 4)
        return offset + width * 4

    offset += 4
    for channel in range(4):  # R, G, B, E stored as four separate RLE streams
        x = 0
        while x < width:
            count = body[offset]
            offset += 1
            if count > 128:  # a run: (count - 128) copies of the next byte
                run = count - 128
                row[x : x + run, channel] = body[offset]
                offset += 1
                x += run
            else:  # a literal span of `count` bytes
                row[x : x + count, channel] = np.frombuffer(
                    body, dtype=np.uint8, count=count, offset=offset
                )
                offset += count
                x += count
    return offset
