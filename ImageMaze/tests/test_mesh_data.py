import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from maze_scene import WallCell
from mesh_data import (
    build_coloured_mesh,
    build_wall_mesh,
    build_wireframe_wall_mesh,
    ground_mesh,
)


def test_build_coloured_mesh_translates_positions_and_adds_colour():
    source = np.array([1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.25, 0.75], dtype=np.float32)

    result = build_coloured_mesh(
        source, (0.25, 0.5, 0.75, 1.0), translation=(4.0, 5.0, 6.0)
    )

    np.testing.assert_allclose(
        result,
        np.array([[5.0, 7.0, 9.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32),
    )


def test_build_wall_mesh_combines_one_cube_per_wall():
    cube = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )
    walls = (
        WallCell(2.0, 3.0, (1.0, 0.0, 0.0, 1.0)),
        WallCell(-1.0, 4.0, (0.0, 0.0, 1.0, 1.0)),
    )

    result = build_wall_mesh(cube, walls)

    assert result.shape == (4, 7)
    np.testing.assert_allclose(result[0], (2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(result[2], (-1.0, 0.0, 4.0, 0.0, 0.0, 1.0, 1.0))


def test_ground_mesh_is_two_triangles_at_the_source_height():
    result = ground_mesh(40.0, -0.55)

    assert result.shape == (6, 7)
    np.testing.assert_allclose(result[:, 1], -0.55)
    np.testing.assert_allclose(result[:, 3:], np.tile((0.3, 0.3, 0.3, 1.0), (6, 1)))


def test_wireframe_wall_mesh_has_twelve_cube_edges_per_wall():
    wall = WallCell(2.0, 3.0, (1.0, 0.0, 0.0, 1.0))

    result = build_wireframe_wall_mesh((wall,))

    assert result.shape == (24, 7)
    assert set(result[:, 0]) == {1.5, 2.5}
    assert set(result[:, 1]) == {-0.5, 0.5}
    assert set(result[:, 2]) == {2.5, 3.5}
    np.testing.assert_allclose(result[:, 3:], np.tile((1.0, 0.0, 0.0, 1.0), (24, 1)))
