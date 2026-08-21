import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from morph_mesh import (  # noqa: E402
    adjust_weight,
    advance_punch,
    load_morph_mesh,
)


def write_pose(path: Path, offset: float = 0.0, face: str = "1//1 2//1 3//1") -> None:
    path.write_text(
        "\n".join(
            [
                f"v {0.0 + offset} 0.0 0.0",
                f"v {1.0 + offset} 0.0 0.0",
                f"v {0.0 + offset} 1.0 0.0",
                "vn 0.0 0.0 1.0",
                f"f {face}",
            ]
        )
        + "\n"
    )


def test_morph_mesh_packs_base_and_two_pose_deltas(tmp_path):
    paths = [tmp_path / f"pose{i}.obj" for i in range(3)]
    for path, offset in zip(paths, (0.0, 2.0, -1.0), strict=True):
        write_pose(path, offset)

    data = load_morph_mesh(*paths)

    assert data.shape == (3, 18)
    assert np.allclose(data[:, 0:3], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(data[:, 6:9], [2.0, 0.0, 0.0])
    assert np.allclose(data[:, 12:15], [-1.0, 0.0, 0.0])
    assert np.allclose(data[:, 3:6], [0.0, 0.0, 1.0])
    assert np.allclose(data[:, 9:12], 0.0)
    assert np.allclose(data[:, 15:18], 0.0)


def test_morph_mesh_rejects_different_vertex_count(tmp_path):
    paths = [tmp_path / f"pose{i}.obj" for i in range(3)]
    write_pose(paths[0])
    write_pose(paths[1])
    write_pose(paths[2])
    paths[2].write_text(paths[2].read_text().replace("vn ", "v 2 2 2\nvn ", 1))

    with pytest.raises(ValueError, match="vertex and normal counts"):
        load_morph_mesh(*paths)


def test_morph_mesh_accepts_different_normal_face_indices(tmp_path):
    paths = [tmp_path / f"pose{i}.obj" for i in range(3)]
    for path in paths:
        write_pose(path)
        path.write_text(
            path.read_text().replace("vn 0.0 0.0 1.0", "vn 0.0 0.0 1.0\nvn 0.0 1.0 0.0")
        )
    paths[1].write_text(
        paths[1].read_text().replace("f 1//1 2//1 3//1", "f 1//2 2//2 3//2")
    )

    data = load_morph_mesh(*paths)

    assert data.shape == (3, 18)
    assert np.allclose(data[:, 9:12], 0.0)


@pytest.mark.parametrize(
    ("weight", "delta", "expected"),
    [(0.5, 0.1, 0.6), (0.95, 0.1, 1.0), (0.05, -0.1, 0.0)],
)
def test_adjust_weight_clamps_to_unit_range(weight, delta, expected):
    assert adjust_weight(weight, delta) == pytest.approx(expected)


def test_punch_turns_round_at_one():
    weight, direction, active = advance_punch(0.9, 1)

    assert weight == 1.0
    assert direction == -1
    assert active


def test_punch_stops_back_at_zero():
    weight, direction, active = advance_punch(0.1, -1)

    assert weight == 0.0
    assert direction == 1
    assert not active
