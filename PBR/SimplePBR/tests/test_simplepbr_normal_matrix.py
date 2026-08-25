"""Regression test: the PBR normal matrix must be world-space.

PBRFragment.glsl lights in world space (V = camPos - WorldPos), so the
normal matrix is the inverse transpose of the model alone -- here that
model includes the arcball mouse rotation (mouse_global_tx), but never the
look_at view. Same bug and fix as PBR/HDRIBaker/tests/test_hdri_demo.py.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

from ncca.ngl import Mat3, Mat4, Transform, Vec3, look_at, perspective


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "simplepbr_gl_demo", DEMO_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_matrix_comes_from_the_model_not_the_view(monkeypatch):
    mod = _load_main()
    recorded = {}
    monkeypatch.setattr(
        mod.ShaderLib, "set_uniform", lambda name, *v: recorded.__setitem__(name, v)
    )
    transform = Transform()
    transform.set_rotation(30.0, 20.0, 10.0)
    mouse_rot = Mat4().rotate_y(40.0)
    window = SimpleNamespace(
        mouse_global_tx=mouse_rot,
        transform=transform,
        # The demo's own view: eye above the origin, so its rotation part is
        # NOT the identity and a model_view-derived normal matrix differs.
        view=look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0)),
        project=perspective(45.0, 720.0 / 576.0, 0.05, 350.0),
    )
    mod.MainWindow.load_matrices_to_shader(window)
    M = mouse_rot @ transform.matrix()
    expected = Mat3.from_mat4(M).inverse().transposed()
    np.testing.assert_allclose(
        recorded["normalMatrix"][0].to_numpy(), expected.to_numpy(), atol=1e-5
    )
