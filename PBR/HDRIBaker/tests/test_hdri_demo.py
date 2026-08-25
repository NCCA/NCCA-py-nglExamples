"""Regression tests for hdri_demo's uniform packing.

PBR.wgsl works entirely in world space (WorldPos = M * v, V = camPos -
WorldPos, and N/R index the IBL cube maps directly), so the normal matrix
sent to it must come from the model matrix alone. Building it from
model_view instead puts the normals in view space: they rotate with the
camera, dot(N, V) collapses to grazing over parts of the mesh (the Fresnel
term whites out the albedo) and the reflection vector flips. These tests
call the real packing method with a rotated camera to catch a revert.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("wgpu")

from hdri_demo import _TRANSFORM_DTYPE, HDRIScene
from ncca.ngl import FirstPersonCamera, Mat4, PerspMode, Vec3


class _FakeQueue:
    def write_buffer(self, *args, **kwargs) -> None:
        pass


class _FakeDevice:
    queue = _FakeQueue()


def _pack_uniforms(camera: FirstPersonCamera, model: Mat4) -> np.ndarray:
    """Run the real _write_teapot_transform_uniforms against a stand-in
    scene (no GPU, no Qt) and return the packed uniform record."""
    scene = SimpleNamespace(
        device=_FakeDevice(),
        camera=camera,
        teapot_model=model,
        metallic=1.0,
        roughness=0.25,
        ao=1.0,
        albedo=(1.0, 1.0, 0.0),
        teapot_transform_buffer=None,
        teapot_transform_uniforms=np.zeros((), dtype=_TRANSFORM_DTYPE),
    )
    HDRIScene._write_teapot_transform_uniforms(scene)
    return scene.teapot_transform_uniforms


def _rotated_camera() -> FirstPersonCamera:
    # The constructor leaves yaw/pitch at their defaults (identity view
    # rotation), so drag the camera as a user would to get a view matrix
    # with a real rotation part -- exactly the case where a view-space
    # normal matrix goes wrong.
    camera = FirstPersonCamera(
        Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
    )
    camera.process_mouse_movement(400.0, 150.0)
    return camera


def test_normal_matrix_ignores_the_camera():
    uniforms = _pack_uniforms(_rotated_camera(), Mat4())
    # Identity model => world-space normal matrix is the identity, however
    # the camera is oriented.
    np.testing.assert_allclose(uniforms["normalMatrix"], np.eye(4), atol=1e-6)


def test_normal_matrix_is_inverse_transpose_of_model():
    model = Mat4().rotate_y(30.0) @ Mat4().rotate_x(20.0)
    uniforms = _pack_uniforms(_rotated_camera(), model)
    expected = model.copy().inverse().transposed().to_numpy()
    np.testing.assert_allclose(uniforms["normalMatrix"], expected, atol=1e-6)
