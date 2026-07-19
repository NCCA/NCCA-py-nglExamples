"""Headless unit tests for the depth-sorting maths in blend_scene.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from blend_scene import PANELS, back_to_front, view_space_z


def translation(x: float, y: float, z: float) -> np.ndarray:
    """A 4x4 translation in the PyNGL row-vector convention (row 3)."""
    m = np.identity(4, dtype=np.float64)
    m[3, :3] = (x, y, z)
    return m


def rotate_y_180() -> np.ndarray:
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = -1.0
    m[2, 2] = -1.0
    return m


def test_view_space_z_reads_translation_row():
    assert view_space_z(translation(1.0, 2.0, -3.5)) == -3.5


def test_view_space_z_of_offset_point():
    # local point (0,0,1) pushed back 2: view z = -2 + 1
    assert view_space_z(translation(0.0, 0.0, -2.0), point=(0.0, 0.0, 1.0)) == -1.0


def test_back_to_front_sorts_furthest_first():
    # camera looks down -z: most negative z is furthest and must come first
    mvs = [translation(0, 0, 0.5), translation(0, 0, -2.0), translation(0, 0, 1.5)]
    assert back_to_front(mvs) == [1, 0, 2]


def test_back_to_front_reverses_when_camera_flips():
    mvs = [translation(0, 0, z) for z in (-2.0, -1.0, 1.0)]
    flipped = [m @ rotate_y_180() for m in mvs]
    assert back_to_front(flipped) == list(reversed(back_to_front(mvs)))


def test_scene_panels_sort_by_z_from_default_camera():
    # identity view: panel draw order must be ascending panel z
    mvs = [translation(*p.position) for p in PANELS]
    order = back_to_front(mvs)
    zs = [PANELS[i].position[2] for i in order]
    assert zs == sorted(zs)
