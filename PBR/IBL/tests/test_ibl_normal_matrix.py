"""Regression test: the sphere grid's normal matrix must be world-space.

IBLFragment.glsl lights in world space (V = camPos - WorldPos, N indexes
the irradiance cube directly), so the normal matrix is the inverse
transpose of each sphere's model (which includes the arcball rotation) --
never model_view. Same bug and fix as
PBR/HDRIBaker/tests/test_hdri_demo.py.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

from ncca.ngl import Mat3, Mat4, Vec3, look_at, perspective


def _load_main():
    spec = importlib.util.spec_from_file_location("ibl_gl_demo", DEMO_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_normal_matrices_come_from_each_model(monkeypatch):
    mod = _load_main()
    recorded: list[tuple[str, tuple]] = []
    monkeypatch.setattr(
        mod.ShaderLib, "set_uniform", lambda name, *v: recorded.append((name, v))
    )
    monkeypatch.setattr(mod.Primitives, "draw", lambda name: None)
    window = SimpleNamespace(
        # A view with a real rotation part, like the demo's own orbiting eye.
        view=look_at(Vec3(4, 3, -5), Vec3(0, 0, 0), Vec3(0, 1, 0)),
        project=perspective(45.0, 1.2, 0.05, 350.0),
    )
    mod.MainWindow._draw_sphere_grid(window, Mat4().rotate_y(40.0))
    models = [v[0] for name, v in recorded if name == "M"]
    normals = [v[0] for name, v in recorded if name == "normalMatrix"]
    assert len(models) == len(normals) == 49
    for model, normal in zip(models, normals):
        expected = Mat3.from_mat4(model).inverse().transposed()
        np.testing.assert_allclose(normal.to_numpy(), expected.to_numpy(), atol=1e-5)
