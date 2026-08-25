"""Tests for the WebGPU viewport that do not create a GPU device."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import main_webgpu as bvh_webgpu
from ncca.ngl import PerspMode, perspective
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def viewport(
    application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> bvh_webgpu.BvhWebGPUViewport:
    monkeypatch.setattr(
        bvh_webgpu.BvhWebGPUViewport,
        "_initialise_webgpu",
        lambda self: None,
    )
    return bvh_webgpu.BvhWebGPUViewport()


def test_webgpu_viewport_uses_the_webgpu_depth_range(
    viewport: bvh_webgpu.BvhWebGPUViewport,
) -> None:
    viewport.resizeWebGPU(800, 400)

    expected = perspective(45.0, 2.0, 0.05, 1500.0, PerspMode.WebGPU)
    assert viewport.camera.persp_mode is PerspMode.WebGPU
    np.testing.assert_allclose(
        viewport.camera.projection.to_numpy(), expected.to_numpy()
    )


def test_webgpu_four_view_rectangles_use_top_left_coordinates(
    viewport: bvh_webgpu.BvhWebGPUViewport,
) -> None:
    viewport.resizeWebGPU(800, 600)

    assert viewport._four_view_rectangles() == [
        (0, 0, 399, 299),
        (401, 0, 399, 299),
        (0, 301, 399, 299),
        (401, 301, 399, 299),
    ]


def test_webgpu_shader_is_shipped_with_the_viewer() -> None:
    assert bvh_webgpu.SHADER_PATH.is_file()


def test_four_view_keeps_the_single_view_background_colour() -> None:
    assert bvh_webgpu.viewport_clear_colour(False) == (0.18, 0.19, 0.20, 1.0)
    assert bvh_webgpu.viewport_clear_colour(True) == (0.18, 0.19, 0.20, 1.0)
