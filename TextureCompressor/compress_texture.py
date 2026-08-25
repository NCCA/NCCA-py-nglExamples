#!/usr/bin/env -S uv run --script
"""Compress an image to the ngl::cmptx DXT1 format used by main.py's viewer.

A from-scratch reinterpretation of the NGL9Demos Compressor CLI tool: same
job (image in, compressed .cmptx file out) and the same output file format,
without linking libsquish (unavailable from Python here) -- see
dxt_texture.py for the encoder.

Usage:
    ./compress_texture.py input.png [-o output.cmptx]
"""

import argparse
from pathlib import Path

import numpy as np
from dxt_texture import write_cmptx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    rgb = np.array(img)
    h, w, _ = rgb.shape
    h4, w4 = h - (h % 4), w - (w % 4)
    rgb = rgb[:h4, :w4]

    output = args.output or args.input.with_suffix(".cmptx")
    write_cmptx(output, rgb)
    print(f"wrote {output} ({w4}x{h4}, DXT1, {output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
