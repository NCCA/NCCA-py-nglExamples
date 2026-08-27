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
    top_view,
)
from ncca.ngl import perspective


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

    assert [(wall.x, wall.z) for wall in walls] == [(1.5, 1.0), (-0.5, 0.0)]
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
    pixels[3, 4] = (0, 0, 0)
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


def test_walls_and_actor_share_one_image_to_world_mapping():
    """A cell the actor stands on must not also be drawn as a wall cube."""
    pixels = np.full((5, 5, 3), 255, dtype=np.uint8)
    pixels[1, 3] = (0, 0, 0)
    maze = Maze.from_pixels(pixels)

    wall = maze.wall_cells()[0]
    actor_x, _, actor_z = actor_world_position(maze, ActorState(x=3, z=1))

    assert (wall.x, wall.z) == (actor_x, actor_z)


def test_actor_start_cell_in_the_shipped_maze_is_open():
    maze = Maze.from_file(Path(__file__).parent.parent / "maps" / "small.png")
    actor = ActorState(2, 2)

    assert maze.is_open(actor.x, actor.z)

    x, _, z = actor_world_position(maze, actor)
    assert not any(wall.x == x and wall.z == z for wall in maze.wall_cells())


def test_actor_can_move_in_every_direction_from_the_default_start():
    maze = Maze.from_file(Path(__file__).parent.parent / "maps" / "small.png")
    actor = ActorState(2, 2)

    for direction in Direction:
        assert move_actor(maze, actor, direction) != actor, direction.name


def test_arrow_directions_move_the_actor_the_same_way_on_screen():
    """Up moves the troll up the overhead view, left moves it left, and so on."""
    maze = Maze.from_pixels(np.full((9, 9, 3), 255, dtype=np.uint8))
    actor = ActorState(x=4, z=4)
    view_project = (
        top_view().to_numpy() @ perspective(45.0, 4 / 3, 0.5, 50.0).to_numpy()
    )

    def screen(state):
        x, y, z = actor_world_position(maze, state)
        clip = np.array([x, y, z, 1.0]) @ view_project
        return clip[:2] / clip[3]

    start = screen(actor)
    for direction, expected in (
        (Direction.NORTH, (0.0, 1.0)),
        (Direction.SOUTH, (0.0, -1.0)),
        (Direction.EAST, (1.0, 0.0)),
        (Direction.WEST, (-1.0, 0.0)),
    ):
        moved = screen(move_actor(maze, actor, direction))
        assert tuple(np.sign(np.round(moved - start, 6))) == expected, direction.name
