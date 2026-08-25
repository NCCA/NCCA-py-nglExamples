"""Regression tests: the PBR normal matrix must be world-space.

PBRFragment.glsl / PBRTexture.wgsl light in world space (V = camPos -
WorldPos), so the normal matrix is the inverse transpose of the model
alone -- deriving it from model_view drags the normals into view space and
the shading rotates with the camera. Same bug and fix as
PBR/HDRIBaker/tests/test_hdri_demo.py.
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
    spec = importlib.util.spec_from_file_location(name, DEMO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dragged_camera(mode: PerspMode) -> FirstPersonCamera:
    camera = FirstPersonCamera(Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, mode)
    camera.process_mouse_movement(400.0, 150.0)
    return camera


class _FakeQueue:
    def write_buffer(self, *args, **kwargs) -> None:
        pass


class _FakeDevice:
    queue = _FakeQueue()


def test_webgpu_normal_matrix_ignores_the_camera():
    mod = _load("pbrtexture_webgpu_demo", "PBRTextureWebGPU.py")
    model = Mat4()
    model[3, 0] = 2.0
    scene = SimpleNamespace(
        device=_FakeDevice(),
        camera=_dragged_camera(PerspMode.WebGPU),
        scene_objects=[("sphere", model, (0.0, 0.0, 0.0, 0.0), None)],
        transform_uniforms=np.zeros(1, dtype=mod._TRANSFORM_DTYPE),
        transform_buffer=None,
    )
    mod.PBRTextureScene._write_transform_uniforms(scene)
    expected = model.copy().inverse().transposed().to_numpy()
    np.testing.assert_allclose(
        scene.transform_uniforms[0]["normalMatrix"], expected, atol=1e-6
    )


def test_opengl_normal_matrix_ignores_the_camera(monkeypatch):
    mod = _load("pbrtexture_gl_demo", "main.py")
    recorded = {}
    monkeypatch.setattr(
        mod.ShaderLib, "set_uniform", lambda name, *v: recorded.__setitem__(name, v)
    )
    monkeypatch.setattr(mod.ShaderLib, "use", lambda name: None)
    transform = Transform()
    transform.set_rotation(30.0, 20.0, 10.0)
    window = SimpleNamespace(
        transform=transform, camera=_dragged_camera(PerspMode.OpenGL)
    )
    mod.MainWindow.load_matrices_to_shader(window)
    expected = Mat3.from_mat4(transform.matrix()).inverse().transposed()
    np.testing.assert_allclose(
        recorded["normalMatrix"][0].to_numpy(), expected.to_numpy(), atol=1e-5
    )
