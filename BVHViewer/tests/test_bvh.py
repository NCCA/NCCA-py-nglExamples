"""Tests for the .bvh parser and frame-playback logic.

No Qt, no OpenGL: `Bvh` only depends on `ncca.ngl`'s pure-numpy maths
(Vec3, Mat4, Transform), so these run headless and check the parsing and
per-frame skeleton maths rather than anything on screen.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from bvh import Bvh, BvhParseError, rotation_from_y  # noqa: E402
from ncca.ngl import Mat3, Vec3  # noqa: E402

DEMO_DIR = Path(__file__).parent.parent
TEST_BVH = DEMO_DIR / "bvh" / "test.bvh"

_MINIMAL = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
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
Frames: 2
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.0 2.0 3.0 10.0 20.0 30.0 40.0 50.0 60.0
"""

# A root with a downward-pointing leg (End Site 50 units below the root in
# the rest pose, in the *scaled* units local_matrix works in -- OFFSET/MOTION
# values here are all still raw file units, x10) and a large, arbitrary
# starting XZ translation -- exercises the grounding/recentring correction,
# which _MINIMAL's straight-up chain and zero starting translation can't.
# x0/z0/x1/z1 are raw (pre-scale) values -- pass 10x the desired scaled units.
_GROUNDING = """\
HIERARCHY
ROOT Hips
{{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Leg
  {{
    OFFSET 0.0 -300.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {{
      OFFSET 0.0 -200.0 0.0
    }}
  }}
}}
MOTION
Frames: 2
Frame Time: 0.033333
{x0} 2000.0 {z0} 0.0 0.0 0.0   0.0 0.0 0.0
{x1} 2100.0 {z1} 0.0 0.0 0.0   0.0 0.0 0.0
"""


def test_parses_root_and_child_joint_names():
    bvh = Bvh.from_text(_MINIMAL)
    assert bvh.root.name == "Hips"
    assert [child.name for child in bvh.root.children] == ["Spine"]


def test_end_site_is_a_child_with_no_channels():
    bvh = Bvh.from_text(_MINIMAL)
    spine = bvh.root.children[0]
    assert [child.name for child in spine.children] == ["End Site"]
    assert spine.children[0].channels == []


def test_offsets_are_scaled_from_bvh_units_to_scene_units():
    bvh = Bvh.from_text(_MINIMAL)
    spine = bvh.root.children[0]
    # OFFSET 0.0 10.0 0.0 -> scaled by 1/10
    assert spine.offset.y == pytest.approx(1.0)


def test_channels_are_recorded_in_declared_order():
    bvh = Bvh.from_text(_MINIMAL)
    assert bvh.root.channels == [
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Xrotation",
        "Yrotation",
    ]
    assert bvh.root.children[0].channels == ["Zrotation", "Xrotation", "Yrotation"]


def test_frame_count_and_frame_time_are_parsed():
    bvh = Bvh.from_text(_MINIMAL)
    assert bvh.num_frames == 2
    assert bvh.frame_time == pytest.approx(0.033333)


def test_motion_columns_are_split_to_the_matching_joint_in_declared_order():
    bvh = Bvh.from_text(_MINIMAL)
    spine = bvh.root.children[0]
    # second frame's last 3 values (40, 50, 60) are Spine's Z/X/Yrotation
    np.testing.assert_allclose(spine.motion[1], [40.0, 50.0, 60.0])
    # first 6 values of that frame are Hips' 6 channels
    np.testing.assert_allclose(bvh.root.motion[1], [1.0, 2.0, 3.0, 10.0, 20.0, 30.0])


def test_too_few_motion_lines_raises():
    truncated = _MINIMAL.replace("Frames: 2", "Frames: 3")
    with pytest.raises(BvhParseError):
        Bvh.from_text(truncated)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        Bvh(DEMO_DIR / "bvh" / "does_not_exist.bvh")


def test_loads_the_repo_hand_written_fixture():
    bvh = Bvh(TEST_BVH)
    assert bvh.root.name == "Hips"
    assert bvh.num_frames == 135
    # nine end effectors: left/right hand, foot, and head
    names = _all_names(bvh.root)
    assert names.count("End Site") == 5


def _all_names(joint) -> list[str]:
    names = [joint.name]
    for child in joint.children:
        names.extend(_all_names(child))
    return names


def test_replay_resets_to_frame_zero():
    bvh = Bvh.from_text(_MINIMAL)
    bvh.step_forward()
    bvh.replay()
    assert bvh.current_frame == 0


def test_step_forward_stops_at_the_last_frame():
    bvh = Bvh.from_text(_MINIMAL)
    bvh.step_forward()
    bvh.step_forward()
    bvh.step_forward()
    assert bvh.current_frame == bvh.num_frames - 1


def test_step_backward_stops_at_frame_zero():
    bvh = Bvh.from_text(_MINIMAL)
    bvh.step_backward()
    assert bvh.current_frame == 0


def test_advance_loops_back_to_zero_after_the_last_frame():
    bvh = Bvh.from_text(_MINIMAL)
    bvh.step_forward()  # now at the last frame (num_frames=2)
    bvh.advance()
    assert bvh.current_frame == 0


@pytest.mark.parametrize(
    ("requested_frame", "expected_frame"),
    [(-10, 0), (0, 0), (1, 1), (200, 1)],
)
def test_seek_clamps_the_requested_frame_to_the_clip_range(
    requested_frame: int, expected_frame: int
) -> None:
    bvh = Bvh.from_text(_MINIMAL)

    bvh.seek(requested_frame)

    assert bvh.current_frame == expected_frame


# --------------------------------------------------------- rotation_from_y


def _apply_rotation(m, v: Vec3) -> Vec3:
    return v @ Mat3.from_mat4(m)


def test_rotation_from_y_is_identity_for_straight_up():
    m = rotation_from_y(Vec3(0.0, 1.0, 0.0))
    up = _apply_rotation(m, Vec3(0.0, 1.0, 0.0))
    np.testing.assert_allclose([up.x, up.y, up.z], [0.0, 1.0, 0.0], atol=1e-5)


def test_rotation_from_y_flips_for_straight_down():
    m = rotation_from_y(Vec3(0.0, -1.0, 0.0))
    up = _apply_rotation(m, Vec3(0.0, 1.0, 0.0))
    np.testing.assert_allclose([up.x, up.y, up.z], [0.0, -1.0, 0.0], atol=1e-5)


def test_rotation_from_y_points_the_y_axis_along_an_arbitrary_direction():
    direction = Vec3(1.0, 1.0, 0.0).normalized()
    m = rotation_from_y(direction)
    up = _apply_rotation(m, Vec3(0.0, 1.0, 0.0))
    np.testing.assert_allclose(
        [up.x, up.y, up.z], [direction.x, direction.y, direction.z], atol=1e-5
    )


# ------------------------------------------------------------ local_matrix


def test_local_matrix_places_root_at_its_offset_plus_frame_translation():
    bvh = Bvh.from_text(_MINIMAL)
    bvh.step_forward()  # frame 1: Hips translation channels are (1, 2, 3)
    m = bvh.local_matrix(bvh.root)
    # root offset is (0,0,0), so the world position is just the scaled translation
    np.testing.assert_allclose(list(m[3])[:3], [0.1, 0.2, 0.3], atol=1e-5)


def test_local_matrix_for_a_leaf_with_no_channels_is_a_pure_offset():
    bvh = Bvh.from_text(_MINIMAL)
    end_site = bvh.root.children[0].children[0]
    m = bvh.local_matrix(end_site)
    np.testing.assert_allclose(list(m[3])[:3], [0.0, 0.5, 0.0], atol=1e-5)


def test_local_matrix_matches_intrinsic_zxy_composition():
    # Declared order Z,X,Y means physically rotate about the original Z
    # first, then the new X, then the newest Y (intrinsic composition) --
    # the standard BVH semantic, cross-checked against an independent BVH
    # library and against the maths directly. As a column-vector rotation
    # matrix this is R = Rz @ Rx @ Ry; applying R to (1,0,0) with angles
    # z=30, x=20, y=10 gives this fixed reference value.
    bvh = Bvh.from_text(_MINIMAL)
    bvh.root.motion = np.array([[0.0, 0.0, 0.0, 30.0, 20.0, 10.0]])
    v = Vec3(1.0, 0.0, 0.0)
    result = _apply_rotation(bvh.local_matrix(bvh.root), v)
    np.testing.assert_allclose(
        [result.x, result.y, result.z],
        [0.82317294, 0.54383814, -0.16317591],
        atol=1e-5,
    )


def test_local_matrix_honours_the_declared_rotation_channel_order():
    # Spine's channels are declared Z,X,Y -- rotating 90 about Z then X gives
    # a different result than rotating 90 about X then Z, so this pins down
    # that the parser's declared order is actually respected at pose time.
    zxy = Bvh.from_text(_MINIMAL)
    zxy.root.children[0].channels = ["Zrotation", "Xrotation", "Yrotation"]
    zxy.root.children[0].motion = np.array([[90.0, 90.0, 0.0], [90.0, 90.0, 0.0]])

    xzy = Bvh.from_text(_MINIMAL)
    xzy.root.children[0].channels = ["Xrotation", "Zrotation", "Yrotation"]
    xzy.root.children[0].motion = np.array([[90.0, 90.0, 0.0], [90.0, 90.0, 0.0]])

    v = Vec3(1.0, 0.0, 0.0)
    a = _apply_rotation(zxy.local_matrix(zxy.root.children[0]), v)
    b = _apply_rotation(xzy.local_matrix(xzy.root.children[0]), v)
    assert not np.allclose([a.x, a.y, a.z], [b.x, b.y, b.z], atol=1e-5)


# ------------------------------------------------------- grounding/recentring


def _hips_and_leg_end_world_y(bvh: Bvh) -> tuple[float, float]:
    hips = bvh.local_matrix(bvh.root)
    leg = bvh.root.children[0]
    leg_world = hips @ bvh.local_matrix(leg)
    end_site = leg.children[0]
    end_world = leg_world @ bvh.local_matrix(end_site)
    return float(hips[3][1]), float(end_world[3][1])


def test_lowest_rest_pose_point_is_grounded_to_y_zero():
    # rest pose: End Site sits 30+20=50 units below the root, regardless of
    # the root's own (arbitrary) starting height of 200/210.
    bvh = Bvh.from_text(_GROUNDING.format(x0=0, z0=0, x1=0, z1=0))
    _, end_y_frame0 = _hips_and_leg_end_world_y(bvh)
    assert end_y_frame0 == pytest.approx(0.0, abs=1e-4)

    bvh.current_frame = 1
    _, end_y_frame1 = _hips_and_leg_end_world_y(bvh)
    # the correction is a *constant* shift, not a re-ground every frame --
    # frame 1's root is 10 units higher than frame 0's, so its (unrotated)
    # leg end should be 10 units higher too, not re-grounded to 0.
    assert end_y_frame1 == pytest.approx(10.0, abs=1e-4)


def test_root_xz_is_recentred_to_the_origin_at_frame_zero():
    bvh = Bvh.from_text(_GROUNDING.format(x0=5000.0, z0=-3000.0, x1=5200.0, z1=-2800.0))
    m0 = bvh.local_matrix(bvh.root)
    assert float(m0[3][0]) == pytest.approx(0.0, abs=1e-4)
    assert float(m0[3][2]) == pytest.approx(0.0, abs=1e-4)

    # the *relative* motion between frames is preserved -- only the starting
    # point moves, not the shape of the path
    bvh.current_frame = 1
    m1 = bvh.local_matrix(bvh.root)
    assert float(m1[3][0]) == pytest.approx(20.0, abs=1e-4)
    assert float(m1[3][2]) == pytest.approx(20.0, abs=1e-4)
