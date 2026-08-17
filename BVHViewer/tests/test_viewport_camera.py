"""Tests for the BVH viewport camera controls."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402
from ncca.ngl import FirstPersonCamera, Mat4, Vec3, look_at  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QWheelEvent,
)
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


def test_four_view_mode_draws_top_perspective_front_and_side(
    application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewport = bvh_viewer.BvhViewport()
    gl_viewports: list[tuple[int, int, int, int]] = []
    clear_colours: list[tuple[float, float, float, float]] = []
    draws: list[tuple[Mat4, Mat4]] = []

    monkeypatch.setattr(viewport, "makeCurrent", lambda: None)
    monkeypatch.setattr(viewport, "_draw_view_labels", lambda: None, raising=False)
    monkeypatch.setattr(
        bvh_viewer.gl,
        "glViewport",
        lambda x, y, width, height: gl_viewports.append((x, y, width, height)),
    )
    monkeypatch.setattr(bvh_viewer.gl, "glScissor", lambda *args: None)
    monkeypatch.setattr(bvh_viewer.gl, "glEnable", lambda *args: None)
    monkeypatch.setattr(bvh_viewer.gl, "glDisable", lambda *args: None)
    monkeypatch.setattr(
        bvh_viewer.gl,
        "glClearColor",
        lambda r, g, b, a: clear_colours.append((r, g, b, a)),
    )
    monkeypatch.setattr(bvh_viewer.gl, "glClear", lambda *args: None)
    monkeypatch.setattr(
        viewport.scene,
        "draw",
        lambda view, project, model, trace=False: draws.append((view, project)),
    )

    viewport.set_four_view(True)
    viewport.resizeGL(800, 600)
    viewport.paintGL()

    assert gl_viewports == [
        (0, 301, 399, 299),
        (401, 301, 399, 299),
        (0, 0, 399, 299),
        (401, 0, 399, 299),
    ]
    assert clear_colours == [
        (0.08, 0.08, 0.08, 1.0),
        (0.18, 0.19, 0.20, 1.0),
    ]
    assert len(draws) == 4
    assert np.allclose(
        draws[0][0].to_numpy(),
        look_at(Vec3(0, 500, 0), Vec3(0, 0, 0), Vec3(0, 0, -1)).to_numpy(),
    )
    assert np.allclose(draws[1][0].to_numpy(), viewport.camera.view.to_numpy())
    assert np.allclose(
        draws[2][0].to_numpy(),
        look_at(Vec3(0, 20, 500), Vec3(0, 20, 0), Vec3(0, 1, 0)).to_numpy(),
    )
    assert np.allclose(
        draws[3][0].to_numpy(),
        look_at(Vec3(500, 20, 0), Vec3(0, 20, 0), Vec3(0, 1, 0)).to_numpy(),
    )
    assert np.allclose(draws[0][1].to_numpy(), draws[2][1].to_numpy())
    assert np.allclose(draws[0][1].to_numpy(), draws[3][1].to_numpy())
    assert not np.allclose(draws[0][1].to_numpy(), draws[1][1].to_numpy())
    assert viewport.camera.aspect == pytest.approx(399 / 299)


def test_view_menu_toggles_four_view_mode(application: QApplication) -> None:
    window = bvh_viewer.MainWindow(Path(__file__).parent.parent / "bvh" / "test.bvh")

    assert window.four_view_action.isCheckable()
    assert window.four_view_action.shortcut() == QKeySequence(Qt.Key.Key_4)
    assert window.viewport.four_view is False

    window.four_view_action.trigger()

    assert window.four_view_action.isChecked()
    assert window.viewport.four_view is True


def test_four_view_labels_identify_each_camera(
    application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewport = bvh_viewer.BvhViewport()
    labels: list[tuple[int, int, str]] = []

    monkeypatch.setattr(bvh_viewer.gl, "glViewport", lambda *args: None)
    monkeypatch.setattr(
        bvh_viewer.Text,
        "render_text",
        lambda font, x, y, label, colour: labels.append((x, y, label)),
    )
    viewport.set_four_view(True)
    viewport.resizeGL(800, 600)

    viewport._draw_view_labels()

    assert labels == [
        (10, 20, "TOP"),
        (411, 20, "PERSPECTIVE"),
        (10, 321, "FRONT"),
        (411, 321, "SIDE"),
    ]


def test_orthographic_drag_does_not_rotate_the_perspective_camera(
    application: QApplication,
) -> None:
    viewport = bvh_viewer.BvhViewport()
    viewport.set_four_view(True)
    viewport.resizeGL(800, 600)
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

    assert np.allclose(viewport.camera.front.to_numpy(), initial_front)


def test_wheel_zooms_the_orthographic_views_under_the_pointer(
    application: QApplication,
) -> None:
    viewport = bvh_viewer.BvhViewport()
    viewport.set_four_view(True)
    viewport.resizeGL(800, 600)
    initial_size = viewport.orthographic_half_height
    initial_perspective_zoom = viewport.camera.zoom
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

    assert viewport.orthographic_half_height < initial_size
    assert viewport.camera.zoom == initial_perspective_zoom
