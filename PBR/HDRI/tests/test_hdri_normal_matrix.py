"""Regression tests: the PBR normal matrix must be world-space.

Both HDRI demos light in world space (V = camPos - WorldPos, N/R index the
IBL cube maps directly), so the normal matrix is the inverse transpose of
the model alone. Deriving it from model_view drags the normals into view
space: rotate the camera and dot(N, V) collapses to grazing over parts of
the mesh (Fresnel whites out the albedo) while reflections flip upside
down. Same bug and fix as PBR/HDRIBaker/tests/test_hdri_demo.py.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("wgpu")

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

from ncca.ngl import (  # noqa: E402
    FirstPersonCamera,
    Mat3,
    Mat4,
    PerspMode,
    Transform,
    Vec3,
)


def _load(name: str, filename: str):
    # Several demos are all called main.py, so load under a unique module
    # name rather than a bare `import main` (which pytest would cache
    # across demo folders).
    spec = importlib.util.spec_from_file_location(name, DEMO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dragged_camera() -> FirstPersonCamera:
    # The constructor leaves yaw/pitch at their defaults (identity view
    # rotation), so drag the camera as a user would to get a view matrix
    # with a real rotation part.
    camera = FirstPersonCamera(
        Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
    )
    camera.process_mouse_movement(400.0, 150.0)
    return camera


class _FakeQueue:
    def write_buffer(self, *args, **kwargs) -> None:
        pass


class _FakeDevice:
    queue = _FakeQueue()


def test_webgpu_grid_normal_matrix_ignores_the_camera():
    mod = _load("hdri_webgpu_demo", "HDRIWebGPU.py")
    model = Mat4()
    model[3, 0] = 2.0  # off-origin, like a grid teapot
    scene = SimpleNamespace(
        device=_FakeDevice(),
        camera=_dragged_camera(),
        grid_objects=[(model, 1.0, 0.25)],
        grid_transform_uniforms=np.zeros(1, dtype=mod._TRANSFORM_DTYPE),
        grid_transform_buffer=None,
    )
    mod.HDRIScene._write_grid_transform_uniforms(scene)
    expected = model.copy().inverse().transposed().to_numpy()
    np.testing.assert_allclose(
        scene.grid_transform_uniforms[0]["normalMatrix"], expected, atol=1e-6
    )


def test_opengl_normal_matrix_ignores_the_camera(monkeypatch):
    mod = _load("hdri_gl_demo", "main.py")
    recorded = {}
    monkeypatch.setattr(
        mod.ShaderLib, "set_uniform", lambda name, *v: recorded.__setitem__(name, v)
    )
    transform = Transform()
    transform.set_rotation(30.0, 20.0, 10.0)
    window = SimpleNamespace(transform=transform, camera=_dragged_camera())
    mod.MainWindow.load_matrices_to_shader(window)
    expected = Mat3.from_mat4(transform.matrix()).inverse().transposed()
    np.testing.assert_allclose(
        recorded["normalMatrix"][0].to_numpy(), expected.to_numpy(), atol=1e-5
    )
