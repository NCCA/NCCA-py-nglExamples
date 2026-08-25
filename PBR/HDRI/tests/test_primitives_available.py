"""Regression guard for the primitive setup in main.py.

The OpenGL demo needs a "cube" (skybox / bake geometry) and a "teapot" (the
material grid). Both are *mesh* defaults loaded by
``Primitives.load_default_primitives()`` -- ``Prims.CUBE`` is deliberately NOT a
parametric ``Primitives.create()`` type, so calling ``create`` with it raises
``ValueError`` and, in a live run, aborts ``initializeGL`` before any primitive
registers (every later ``draw`` then logs "Failed to draw primitive"). These
tests pin both facts so the setup can't silently regress. They are headless:
the ``ValueError`` fires before any GL work, and ``PrimData.primitive`` reads
bundled mesh data without a context.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Prims
from ncca.ngl.opengl import Primitives
from ncca.ngl.prim_data import PrimData


def test_cube_is_not_a_parametric_create_type():
    # This is why main.py must NOT call Primitives.create(Prims.CUBE, ...).
    with pytest.raises(ValueError):
        Primitives.create(Prims.CUBE, "cube", 2.0)


@pytest.mark.parametrize("name", ["cube", "teapot"])
def test_default_mesh_primitive_data_present(name):
    data = PrimData.primitive(name)
    # 8 floats per vertex (pos, normal, uv); non-empty and a whole number of verts.
    assert data.size > 0
    assert data.size % 8 == 0
