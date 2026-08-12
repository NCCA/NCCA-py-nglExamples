#!/usr/bin/env -S uv run --script
"""Launch the PyNGL maths node editor."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MathNodeEditor.node_editor import main

if __name__ == "__main__":
    raise SystemExit(main())
