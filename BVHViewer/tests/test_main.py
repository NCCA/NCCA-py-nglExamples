"""Checks for the files needed by the BVH viewer application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402


def test_hud_uses_the_font_shipped_with_the_demos():
    expected = Path(__file__).parents[2] / "font" / "Arial.ttf"
    assert bvh_viewer.HUD_FONT == expected
    assert bvh_viewer.HUD_FONT.is_file()
