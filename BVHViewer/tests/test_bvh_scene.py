"""Tests for BVH scene playback state that do not need an OpenGL context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bvh import Bvh  # noqa: E402
from bvh_scene import BvhScene  # noqa: E402

_CLIP = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
}
MOTION
Frames: 3
Frame Time: 0.04
0.0 0.0 0.0 0.0 0.0 0.0
1.0 0.0 0.0 0.0 0.0 0.0
2.0 0.0 0.0 0.0 0.0 0.0
"""


def test_set_character_replaces_the_previous_clip() -> None:
    scene = BvhScene()
    scene.add_character(Bvh.from_text(_CLIP))
    replacement = Bvh.from_text(_CLIP)

    scene.set_character(replacement)

    assert scene.characters == [replacement]


def test_seek_moves_the_character_to_the_requested_frame() -> None:
    scene = BvhScene()
    scene.set_character(Bvh.from_text(_CLIP))

    scene.seek(2)

    assert scene.current_frame_number() == 2


def test_frame_count_matches_the_loaded_character() -> None:
    scene = BvhScene()
    scene.set_character(Bvh.from_text(_CLIP))

    assert scene.frame_count() == 3
