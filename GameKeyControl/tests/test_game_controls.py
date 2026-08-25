import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_controls import (
    MOTION_TABLE,
    ROTATION_UPDATE,
    X_EXTENTS,
    Y_EXTENTS,
    GameControls,
    KeyRecorder,
    move_ship,
    ship_transform,
)
from ncca.ngl import Vec3


def test_motion_table_has_32_entries():
    assert len(MOTION_TABLE) == 32


def test_motion_table_matches_source_values():
    assert MOTION_TABLE[0] == (0.0, 0.0, 0.0)
    assert MOTION_TABLE[GameControls.UP] == (0.0, 1.0, 0.0)
    assert MOTION_TABLE[GameControls.UP | GameControls.LEFT] == (-0.707, 0.707, 0.0)
    # "nonsense" combo: opposite keys cancel
    assert MOTION_TABLE[GameControls.UP | GameControls.DOWN] == (0.0, 0.0, 0.0)
    # rotate-held half of the table mirrors the first half with rotation=1
    assert MOTION_TABLE[GameControls.ROTATE | GameControls.UP] == (0.0, 1.0, 1.0)


def test_move_ship_applies_offset():
    pos = Vec3(0.0, 0.0, 0.0)
    new_pos, new_rotation = move_ship(pos, 0.0, GameControls.UP)
    assert new_pos.x == 0.0
    assert new_pos.y == 1.0
    assert new_rotation == 0.0


def test_move_ship_clamps_to_extents():
    pos = Vec3(X_EXTENTS - 0.5, Y_EXTENTS - 0.5, 0.0)
    new_pos, _ = move_ship(pos, 0.0, GameControls.UP | GameControls.RIGHT)
    assert new_pos.x == X_EXTENTS
    assert new_pos.y == Y_EXTENTS


def test_move_ship_accumulates_rotation_while_held():
    pos = Vec3(0.0, 0.0, 0.0)
    _, rotation = move_ship(pos, 10.0, GameControls.ROTATE)
    assert rotation == 10.0 + ROTATION_UPDATE


def test_ship_transform_places_translation_in_row_3():
    tx = ship_transform(Vec3(1.0, 2.0, 3.0), 0.0)
    assert tx[3, 0] == 1.0
    assert tx[3, 1] == 2.0
    assert tx[3, 2] == 3.0


def test_key_recorder_round_trips_through_a_file(tmp_path):
    recorder = KeyRecorder()
    recorder.set_start_position(Vec3(1.5, -2.5, 0.0))
    recorder.add_frame(int(GameControls.UP))
    recorder.add_frame(int(GameControls.UP | GameControls.LEFT))
    recorder.add_frame(0)

    out_file = tmp_path / "recording.kp"
    recorder.save(out_file)

    reloaded = KeyRecorder()
    reloaded.load(out_file)
    assert reloaded.size() == 3
    assert reloaded[0] == int(GameControls.UP)
    assert reloaded[1] == int(GameControls.UP | GameControls.LEFT)
    assert reloaded[2] == 0
    start = reloaded.get_start_position()
    assert (start.x, start.y, start.z) == (1.5, -2.5, 0.0)
