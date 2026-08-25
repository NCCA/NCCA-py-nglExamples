"""DXT1 (S3TC) block compression and the ngl::cmptx file format.

A from-scratch encoder/decoder pair -- no Python S3TC/squish library is
available in this environment, and the C++ original's `squish` dependency
isn't portable here. This trades libsquish's cluster-fit quality for a
simple principal-axis endpoint choice: still spec-correct DXT1 data (any
GPU decodes it normally), with more visible block artefacts on hard edges
than a production encoder -- which is fine, even useful, for a demo whose
whole point is to make block compression visible.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from OpenGL.GL.EXT.texture_compression_s3tc import GL_COMPRESSED_RGBA_S3TC_DXT1_EXT

_MAGIC = b"ngl::cmptx"
_DXT1 = 0


def _pack_rgb565(rgb: np.ndarray) -> int:
    r, g, b = (int(c) for c in rgb)
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _unpack_rgb565(value: int) -> np.ndarray:
    r = (value >> 11) & 0x1F
    g = (value >> 5) & 0x3F
    b = value & 0x1F
    return np.array([r << 3, g << 2, b << 3], dtype=np.float32)


def _compress_block(block: np.ndarray) -> bytes:
    """block: (16, 3) uint8 RGB texels of one 4x4 block, row-major."""
    pixels = block.astype(np.float32)
    mean = pixels.mean(axis=0)
    centred = pixels - mean
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centred @ axis
    c0 = np.clip(pixels[np.argmax(proj)], 0, 255).astype(np.uint8)
    c1 = np.clip(pixels[np.argmin(proj)], 0, 255).astype(np.uint8)

    r0 = _pack_rgb565(c0)
    r1 = _pack_rgb565(c1)
    if r0 < r1:
        r0, r1 = r1, r0

    e0 = _unpack_rgb565(r0)
    e1 = _unpack_rgb565(r1)
    palette = np.stack([e0, e1, (2 * e0 + e1) / 3.0, (e0 + 2 * e1) / 3.0])
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    indices = dists.argmin(axis=1).astype(np.uint32)

    index_bits = 0
    for i, idx in enumerate(indices):
        index_bits |= int(idx) << (2 * i)

    return struct.pack("<HHI", r0, r1, index_bits)


def compress_dxt1(rgb: np.ndarray) -> bytes:
    """rgb: (height, width, 3) uint8. height and width must be multiples of 4."""
    height, width, _ = rgb.shape
    if height % 4 or width % 4:
        raise ValueError("DXT1 compression requires dimensions that are multiples of 4")
    blocks = bytearray()
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            block = rgb[by : by + 4, bx : bx + 4].reshape(16, 3)
            blocks += _compress_block(block)
    return bytes(blocks)


def write_cmptx(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    data = compress_dxt1(rgb)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<ii", width, height))
        f.write(struct.pack("<I", int(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT)))
        f.write(struct.pack("<i", _DXT1))
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def read_cmptx(path: Path) -> tuple[int, int, int, bytes]:
    """Returns (width, height, internal_format, data)."""
    with open(path, "rb") as f:
        magic = f.read(10)
        if magic != _MAGIC:
            raise ValueError(f"{path} is not an ngl::cmptx file")
        width, height = struct.unpack("<ii", f.read(8))
        (internal_format,) = struct.unpack("<I", f.read(4))
        f.read(
            4
        )  # compression enum -- unused on read, DXT1 is the only variant this demo writes
        (size,) = struct.unpack("<I", f.read(4))
        data = f.read(size)
    return width, height, internal_format, data


def make_test_pattern(size: int = 256) -> np.ndarray:
    """A synthetic RGB test image with sharp edges and gradients -- deliberately
    a mix even DXT1's coarse 4-colour-per-block palette will visibly struggle
    with in places, so the compression artefacts this demo exists to show are
    actually visible. Self-generated so this folder has no dependency on any
    binary asset from another demo folder.
    """
    y, x = np.mgrid[0:size, 0:size]
    checker = (((x // 16) + (y // 16)) % 2) * 255
    gradient_r = (x * 255 // size).astype(np.uint8)
    gradient_g = (y * 255 // size).astype(np.uint8)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(checker > 0, gradient_r, 255 - gradient_r)
    rgb[..., 1] = gradient_g
    rgb[..., 2] = checker.astype(np.uint8)
    return rgb
