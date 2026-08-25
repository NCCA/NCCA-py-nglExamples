"""Tests for BVH scene playback state that do not need an OpenGL context."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import bvh_scene
from bvh import Bvh
from bvh_scene import BvhScene
from ncca.ngl import Mat4

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


def test_advance_in_range_loops_at_the_selected_end_frame() -> None:
    scene = BvhScene()
    scene.set_character(Bvh.from_text(_CLIP))
    scene.seek(2)

    scene.advance_in_range(1, 2)

    assert scene.current_frame_number() == 1


def test_joint_position_traces_store_each_frame_as_float32() -> None:
    character = Bvh.from_text(_CLIP)

    traces = bvh_scene.joint_position_traces(character)

    assert traces.dtype == np.float32
    np.testing.assert_allclose(
        traces,
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]],
    )


def test_joint_trace_colours_are_unique() -> None:
    colours = bvh_scene.joint_trace_colours(20)

    assert colours.shape == (20, 4)
    assert len(np.unique(colours, axis=0)) == 20


def test_trace_mode_draws_ground_joint_position_lines_and_character(
    monkeypatch,
) -> None:
    scene = BvhScene()
    drawn: list[str] = []
    monkeypatch.setattr(
        scene, "_draw_trace_lines", lambda: drawn.append("traces"), raising=False
    )
    monkeypatch.setattr(
        scene, "_draw_characters", lambda: drawn.append("character"), raising=False
    )
    monkeypatch.setattr(scene, "_draw_ground", lambda: drawn.append("ground"))

    scene.draw(Mat4(), Mat4(), Mat4(), trace=True)

    assert drawn == ["ground", "traces", "character"]
