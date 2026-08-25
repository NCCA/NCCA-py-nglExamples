import sys
from pathlib import Path

from ncca.ngl import Obj

sys.path.insert(0, str(Path(__file__).parent.parent))

from colour_obj import ColourObj


def test_colour_obj_composes_parser_only_obj_data():
    mesh = ColourObj.from_file(
        str(Path(__file__).parent.parent / "models" / "face_mesh_neutral.obj")
    )

    assert not issubclass(ColourObj, Obj)
    assert isinstance(mesh.data, Obj)
    assert len(mesh.data.colour) == len(mesh.data.vertex)
