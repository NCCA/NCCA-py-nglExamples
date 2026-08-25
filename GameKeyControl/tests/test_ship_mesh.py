import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Obj

_SHIP_PATH = Path(__file__).parent.parent / "models" / "SpaceShip.obj"


def test_obj_returns_interleaved_float32_ship_data():
    data = Obj.from_file(str(_SHIP_PATH)).triangle_vertex_data()
    vertex_count = data.size // 8
    assert data.dtype.name == "float32"
    assert vertex_count > 0
    assert data.shape == (vertex_count * 8,)


def test_obj_ship_data_is_a_multiple_of_a_triangle():
    vertex_count = Obj.from_file(str(_SHIP_PATH)).triangle_vertex_data().size // 8
    assert vertex_count % 3 == 0
