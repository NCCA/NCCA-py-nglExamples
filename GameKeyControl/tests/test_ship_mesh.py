import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ship_mesh import load_ship_vertex_data

_SHIP_PATH = Path(__file__).parent.parent / "models" / "SpaceShip.obj"


def test_load_ship_vertex_data_returns_interleaved_float32():
    data, vertex_count = load_ship_vertex_data(_SHIP_PATH)
    assert data.dtype.name == "float32"
    assert vertex_count > 0
    assert data.shape == (vertex_count * 8,)


def test_load_ship_vertex_data_is_a_multiple_of_a_triangle():
    _, vertex_count = load_ship_vertex_data(_SHIP_PATH)
    assert vertex_count % 3 == 0
