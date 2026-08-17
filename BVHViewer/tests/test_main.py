"""Checks for the files needed by the BVH viewer application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402


def test_default_clip_is_shipped_with_the_viewer() -> None:
    assert bvh_viewer.DEFAULT_BVH.parent == Path(__file__).parent.parent / "bvh"
    assert bvh_viewer.DEFAULT_BVH.is_file()
