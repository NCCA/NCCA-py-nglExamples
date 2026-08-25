"""Tests for the WebGPU BVH scene data built without a GPU device."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bvh import Bvh
from bvh_scene_webgpu import (
    build_ground_line_vertices,
    build_skeleton_instances,
    build_trace_line_vertices,
)

_CLIP = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Knee
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.0 5.0 0.0
    }
  }
}
MOTION
Frames: 1
Frame Time: 0.04
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""


def test_skeleton_instances_pack_joints_before_bones() -> None:
    character = Bvh.from_text(_CLIP)

    instances, joint_count, bone_count = build_skeleton_instances([character])

    assert joint_count == 3
    assert bone_count == 2
    assert instances.dtype.names == ("model", "normal_matrix", "colour")
    assert instances.dtype.itemsize == 144
    np.testing.assert_allclose(instances[0]["model"], np.eye(4, dtype=np.float32))
    assert instances[3]["model"][1, 1] == 1.0
    assert instances[4]["model"][1, 1] == 0.5


def test_ground_is_built_as_a_grey_line_list() -> None:
    vertices = build_ground_line_vertices(size=2.0, divisions=2)

    assert vertices.dtype.names == ("position", "colour")
    assert len(vertices) == 12
    np.testing.assert_allclose(vertices[0]["position"], [-1.0, 0.0, -1.0])
    np.testing.assert_allclose(vertices[1]["position"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(vertices["colour"], [[0.6, 0.6, 0.6, 1.0]] * 12)


def test_trace_strips_are_expanded_into_independent_segments() -> None:
    positions = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]],
        dtype=np.float32,
    )
    colours = np.asarray([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32)

    vertices = build_trace_line_vertices(
        positions,
        ranges=[(0, 3), (3, 2)],
        colours=colours,
    )

    assert len(vertices) == 6
    np.testing.assert_allclose(
        vertices["position"],
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]],
    )
    np.testing.assert_allclose(vertices[0:4]["colour"], [[1, 0, 0, 1]] * 4)
    np.testing.assert_allclose(vertices[4:6]["colour"], [[0, 1, 0, 1]] * 2)
