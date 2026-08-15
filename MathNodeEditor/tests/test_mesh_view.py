"""Tests for Mesh Viewer state and per-context geometry caches."""

import sys
from importlib import import_module
from pathlib import Path

import pytest
from ncca.ngl import Vec3, Vec4

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mesh_inputs():
    """Build one valid triangle without optional vertex attributes."""
    math_graph = import_module("math_graph")
    return math_graph.MeshViewerInputs(
        math_graph.VertexArray(
            (
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.0, 1.0, 0.0),
            )
        ),
        math_graph.FaceArray((((0, None, None), (1, None, None), (2, None, None)),)),
        None,
        None,
        None,
    )


def test_display_settings_do_not_invalidate_the_geometry_cache() -> None:
    mesh_view = import_module("mesh_view")
    state = mesh_view.MeshRenderState()
    state.set_mesh(_mesh_inputs())
    geometry_version = state.version

    state.set_shading_mode(mesh_view.SHADING_DIFFUSE)
    state.set_wireframe(True)

    assert state.version == geometry_version


def test_colour_changes_do_not_invalidate_the_geometry_cache() -> None:
    math_graph = import_module("math_graph")
    mesh_view = import_module("mesh_view")
    state = mesh_view.MeshRenderState()
    inputs = _mesh_inputs()
    state.set_mesh(inputs)
    geometry_version = state.version
    recoloured = math_graph.MeshViewerInputs(
        inputs.vertices,
        inputs.faces,
        inputs.uvs,
        inputs.normals,
        Vec4(0.2, 0.4, 0.6, 1.0),
    )

    state.set_mesh(recoloured)

    assert state.version == geometry_version
    assert state.colour == pytest.approx((0.2, 0.4, 0.6, 1.0))


def test_failed_vao_build_remains_eligible_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_view = import_module("mesh_view")
    state = mesh_view.MeshRenderState()
    state.set_mesh(_mesh_inputs())

    class Renderer(mesh_view.MeshRenderMixin):
        pass

    renderer = Renderer()
    renderer._init_mesh_render(state)

    def fail_to_build(*_args: object) -> None:
        raise RuntimeError("simulated VAO failure")

    monkeypatch.setattr(mesh_view, "obj_from_arrays", fail_to_build)

    with pytest.raises(RuntimeError, match="simulated VAO failure"):
        renderer._rebuild_vao_if_needed()

    assert renderer._built_version == -1
    assert renderer._obj is None
