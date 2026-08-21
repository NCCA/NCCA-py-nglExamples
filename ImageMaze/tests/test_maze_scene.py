import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from maze_scene import (
    ActorState,
    Direction,
    Maze,
    actor_forward,
    actor_world_position,
    move_actor,
)


def test_maze_loads_grayscale_png_as_rgb(tmp_path):
    path = tmp_path / "maze.png"
    Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(path)

    maze = Maze.from_file(path)

    assert maze.pixels.shape == (1, 2, 3)
    assert maze.is_open(0, 0)
    assert not maze.is_open(1, 0)


def test_wall_cells_use_source_image_to_world_mapping():
    pixels = np.full((2, 3, 3), 255, dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    pixels[1, 2] = (0, 128, 255)
    maze = Maze.from_pixels(pixels)

    walls = maze.wall_cells()

    assert [(wall.x, wall.z) for wall in walls] == [(1.5, -1.0), (-0.5, 0.0)]
    assert walls[0].colour == (1.0, 0.0, 0.0, 1.0)
    assert walls[1].colour == pytest.approx((0.0, 128.0 / 255.0, 1.0, 1.0))


def test_only_pure_white_pixels_are_open():
    pixels = np.full((2, 2, 3), 255, dtype=np.uint8)
    pixels[0, 0] = (254, 255, 255)
    maze = Maze.from_pixels(pixels)

    assert maze.is_open(1, 1)
    assert not maze.is_open(0, 0)


def test_actor_moves_to_an_open_pixel_and_turns():
    pixels = np.full((7, 7, 3), 255, dtype=np.uint8)
    maze = Maze.from_pixels(pixels)
    actor = ActorState(x=3, z=3)

    moved = move_actor(maze, actor, Direction.EAST)

    assert moved == ActorState(x=4, z=3, rotation=270.0)


def test_actor_does_not_move_or_turn_into_a_wall():
    pixels = np.full((7, 7, 3), 255, dtype=np.uint8)
    pixels[4, 4] = (0, 0, 0)
    maze = Maze.from_pixels(pixels)
    actor = ActorState(x=3, z=3, rotation=90.0)

    moved = move_actor(maze, actor, Direction.EAST)

    assert moved == actor


def test_actor_does_not_leave_the_image():
    maze = Maze.from_pixels(np.full((3, 3, 3), 255, dtype=np.uint8))
    actor = ActorState(x=0, z=2)

    moved = move_actor(maze, actor, Direction.WEST)

    assert moved == actor


def test_actor_world_position_matches_source_transform():
    maze = Maze.from_pixels(np.full((20, 20, 3), 255, dtype=np.uint8))

    position = actor_world_position(maze, ActorState(x=2, z=2))

    assert position == (8.0, 0.5, 8.0)


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0.0, (0.0, 0.0, 1.0)),
        (180.0, (0.0, 0.0, -1.0)),
        (270.0, (-1.0, 0.0, 0.0)),
        (90.0, (1.0, 0.0, 0.0)),
    ],
)
def test_actor_forward_follows_the_direction_of_travel(rotation, expected):
    assert actor_forward(ActorState(2, 2, rotation)) == pytest.approx(expected)
