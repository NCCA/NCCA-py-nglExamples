import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from blade_field import (
    BASE_COLOUR,
    RESTART_INDEX,
    animate_blades,
    create_blade_field,
    expand_line_list,
)


def test_seeded_blade_field_is_repeatable():
    first = create_blade_field(rows=2, cols=3, seed=7)
    second = create_blade_field(rows=2, cols=3, seed=7)

    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.ranges, second.ranges)


def test_blade_roots_cover_the_requested_grid():
    field = create_blade_field(rows=2, cols=2, row_size=4.0, col_size=6.0, seed=1)

    roots = field.vertices[field.ranges[:, 0], :3]

    assert np.allclose(
        roots,
        [[-2.0, 0.0, -3.0], [0.0, 0.0, -3.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert np.allclose(field.vertices[field.ranges[:, 0], 3:], BASE_COLOUR)


def test_restart_index_terminates_every_blade():
    field = create_blade_field(rows=2, cols=3, seed=4)

    assert np.count_nonzero(field.indices == RESTART_INDEX) == 6
    assert field.indices[-1] == RESTART_INDEX


def test_line_list_does_not_join_neighbouring_blades():
    field = create_blade_field(rows=1, cols=2, seed=3)

    lines = expand_line_list(field.vertices, field.ranges)
    first_start, first_count = field.ranges[0]
    second_start, _ = field.ranges[1]

    assert len(lines) == 2 * ((field.ranges[:, 1] - 1).sum())
    assert not any(
        np.array_equal(lines[i, :3], field.vertices[first_start + first_count - 1, :3])
        and np.array_equal(lines[i + 1, :3], field.vertices[second_start, :3])
        for i in range(0, len(lines), 2)
    )


def test_animation_moves_stems_but_keeps_roots_fixed():
    field = create_blade_field(rows=1, cols=1, seed=9)
    original = field.vertices.copy()

    moved = animate_blades(field.vertices, field.ranges, phase=1.5)

    assert np.array_equal(moved[0], original[0])
    assert np.any(moved[1:, (0, 2)] != original[1:, (0, 2)])
    assert np.array_equal(moved[:, (1, 3, 4, 5)], original[:, (1, 3, 4, 5)])
