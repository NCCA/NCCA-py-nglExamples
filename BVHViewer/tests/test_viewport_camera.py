"""Tests for the BVH viewport camera controls."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402
from ncca.ngl import FirstPersonCamera, Mat4  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_viewport_uses_first_person_camera_matrices_for_drawing(
    application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewport = bvh_viewer.BvhViewport()
    drawn_scene: tuple[Mat4, Mat4, Mat4, bool] | None = None

    monkeypatch.setattr(viewport, "makeCurrent", lambda: None)
    monkeypatch.setattr(bvh_viewer.gl, "glViewport", lambda *args: None)
    monkeypatch.setattr(bvh_viewer.gl, "glClear", lambda *args: None)

    def record_draw(
        view: Mat4, project: Mat4, model: Mat4, trace: bool = False
    ) -> None:
        nonlocal drawn_scene
        drawn_scene = view, project, model, trace

    monkeypatch.setattr(viewport.scene, "draw", record_draw)

    viewport.resizeGL(800, 400)
    viewport.paintGL()

    assert isinstance(viewport.camera, FirstPersonCamera)
    assert drawn_scene is not None
    view, project, model, trace = drawn_scene
    assert np.allclose(view.to_numpy(), viewport.camera.view.to_numpy())
    assert np.allclose(project.to_numpy(), viewport.camera.projection.to_numpy())
    assert np.allclose(model.to_numpy(), Mat4().to_numpy())
    assert viewport.camera.aspect == pytest.approx(2.0)
    assert viewport.camera.near == pytest.approx(0.05)
    assert viewport.camera.far == pytest.approx(1500.0)
    assert trace is False


def test_left_mouse_drag_changes_the_camera_direction(
    application: QApplication,
) -> None:
    viewport = bvh_viewer.BvhViewport()
    initial_front = viewport.camera.front.to_numpy().copy()
    press_position = QPointF(10.0, 10.0)
    move_position = QPointF(50.0, 30.0)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        press_position,
        press_position,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        move_position,
        move_position,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    viewport.mousePressEvent(press)
    viewport.mouseMoveEvent(move)

    assert not np.allclose(viewport.camera.front.to_numpy(), initial_front)


def test_forward_key_moves_camera_until_released(application: QApplication) -> None:
    viewport = bvh_viewer.BvhViewport()
    initial_eye = viewport.camera.eye.to_numpy().copy()
    press = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_W,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_W,
        Qt.KeyboardModifier.NoModifier,
    )

    viewport.keyPressEvent(press)
    viewport.advance_camera(0.5)
    moved_eye = viewport.camera.eye.to_numpy().copy()
    viewport.keyReleaseEvent(release)
    viewport.advance_camera(0.5)

    assert moved_eye[2] < initial_eye[2]
    assert np.allclose(viewport.camera.eye.to_numpy(), moved_eye)


def test_mouse_wheel_changes_camera_field_of_view(
    application: QApplication,
) -> None:
    viewport = bvh_viewer.BvhViewport()
    initial_zoom = viewport.camera.zoom
    wheel = QWheelEvent(
        QPointF(20.0, 20.0),
        QPointF(20.0, 20.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    viewport.wheelEvent(wheel)

    assert viewport.camera.zoom < initial_zoom
