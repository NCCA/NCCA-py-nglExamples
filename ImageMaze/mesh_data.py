"""Small vertex conversions used by the WebGPU ImageMaze renderer."""

from collections.abc import Iterable

import numpy as np
from maze_scene import WallCell


def build_coloured_mesh(
    source: np.ndarray,
    colour: tuple[float, float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    vertices = np.asarray(source, dtype=np.float32).reshape(-1, 8)
    positions = vertices[:, :3] + np.asarray(translation, dtype=np.float32)
    colours = np.tile(np.asarray(colour, dtype=np.float32), (len(vertices), 1))
    return np.ascontiguousarray(np.column_stack((positions, colours)), dtype=np.float32)


def build_wall_mesh(cube_data: np.ndarray, walls: Iterable[WallCell]) -> np.ndarray:
    meshes = [
        build_coloured_mesh(
            cube_data,
            wall.colour,
            translation=(wall.x, 0.0, wall.z),
        )
        for wall in walls
    ]
    if not meshes:
        return np.empty((0, 7), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(meshes), dtype=np.float32)


def ground_mesh(size: float, y: float) -> np.ndarray:
    half_size = size * 0.5
    positions = np.array(
        [
            (-half_size, y, half_size),
            (half_size, y, half_size),
            (half_size, y, -half_size),
            (-half_size, y, half_size),
            (half_size, y, -half_size),
            (-half_size, y, -half_size),
        ],
        dtype=np.float32,
    )
    colours = np.tile(np.array((0.3, 0.3, 0.3, 1.0), dtype=np.float32), (6, 1))
    return np.ascontiguousarray(np.column_stack((positions, colours)), dtype=np.float32)


def build_wireframe_wall_mesh(walls: Iterable[WallCell]) -> np.ndarray:
    corners = np.array(
        [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ],
        dtype=np.float32,
    )
    edges = (
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    )
    meshes = []
    for wall in walls:
        positions = corners[list(edges)] + np.array(
            (wall.x, 0.0, wall.z), dtype=np.float32
        )
        colours = np.tile(np.asarray(wall.colour, dtype=np.float32), (len(edges), 1))
        meshes.append(np.column_stack((positions, colours)))
    if not meshes:
        return np.empty((0, 7), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(meshes), dtype=np.float32)
