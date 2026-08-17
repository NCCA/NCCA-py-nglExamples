#!/usr/bin/env -S uv run --script
"""A PyNGL BVH viewer with a desktop animation timeline."""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import OpenGL.GL as gl
from bvh import Bvh, BvhParseError
from bvh_scene import BvhScene
from ncca.ngl import FirstPersonCamera, Mat4, Vec3, logger, look_at, ortho
from ncca.ngl.opengl import Text
from PySide6.QtCore import QElapsedTimer, QEvent, QObject, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QCursor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from timeline import FrameRangeSlider, TimelineWidget

__all__ = ["FrameRangeSlider", "TimelineWidget"]

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_BVH = DEMO_DIR / "bvh" / "Male1_B10_WalkTurnLeft45.bvh"

_APP_STYLE = """
QMainWindow, QWidget {
    background: #34373b;
    color: #dedede;
}
QMenuBar, QMenu, QStatusBar {
    background: #292c30;
    color: #dedede;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #4c5964;
}
#timelinePanel {
    background: #292c30;
    border-top: 1px solid #17191b;
}
QPushButton {
    background: #41454a;
    border: 1px solid #1e2023;
    border-radius: 2px;
}
QPushButton:hover { background: #51565c; }
QPushButton:pressed, QPushButton:checked { background: #697985; }
QSpinBox, QDoubleSpinBox {
    background: #202225;
    border: 1px solid #53585e;
    padding: 2px 5px;
    selection-background-color: #d48a35;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #17191b;
    border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #cf7b2a; }
QSlider::handle:horizontal {
    width: 11px;
    margin: -5px 0;
    background: #e6a14e;
    border: 1px solid #111;
    border-radius: 2px;
}
"""


TOP_VIEW = 0
PERSPECTIVE_VIEW = 1
FRONT_VIEW = 2
SIDE_VIEW = 3
_VIEW_LABELS = ("TOP", "PERSPECTIVE", "FRONT", "SIDE")


@dataclass
class OrthoView:
    """Zoom and pan state for one orthographic pane, independent of the others."""

    eye: Vec3
    target: Vec3
    up: Vec3
    right: Vec3
    half_height: float = 40.0
    pan: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def matrices(self, pane_width: int, pane_height: int) -> tuple[Mat4, Mat4]:
        """Return this pane's current (view, projection) matrices.

        Parameters
        ----------
            pane_width : int
                pane width, in the same units as pane_height
            pane_height : int
                pane height, used to keep the projection square-pixelled

        Returns
        -------
            tuple[Mat4, Mat4]
                view matrix (including pan) and orthographic projection
        """
        half_width = self.half_height * pane_width / max(pane_height, 1)
        project = ortho(
            -half_width, half_width, -self.half_height, self.half_height, 0.05, 1500.0
        )
        view = look_at(self.eye + self.pan, self.target + self.pan, self.up)
        return view, project

    def pan_by(self, dx_pixels: float, dy_pixels: float, pane_height: float) -> None:
        """Shift this pane's pan offset by a screen-space drag delta.

        Parameters
        ----------
            dx_pixels : float
                horizontal drag delta in screen pixels (positive is right)
            dy_pixels : float
                vertical drag delta in screen pixels (positive is down)
            pane_height : float
                current pane height, to convert pixels to world units
        """
        units_per_pixel = (2.0 * self.half_height) / max(pane_height, 1.0)
        self.pan = (
            self.pan
            + self.right * (-dx_pixels * units_per_pixel)
            + self.up * (dy_pixels * units_per_pixel)
        )


class BvhViewport(QOpenGLWindow):
    """The OpenGL viewport used by the main application window."""

    _MOVE_KEYS = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D}
    _DIVIDER_WIDTH = 2

    def __init__(self) -> None:
        super().__init__()
        self.camera = FirstPersonCamera(
            Vec3(0, 12, 100),
            Vec3(0, 12, 0),
            Vec3(0, 1, 0),
            45.0,
        )
        self.camera.speed = 25.0
        self.window_width = 1024
        self.window_height = 640
        self.scene = BvhScene()
        self.trace = False
        self.four_view = False
        self.ortho_views: dict[int, OrthoView] = {
            TOP_VIEW: OrthoView(
                eye=Vec3(0, 500, 0),
                target=Vec3(0, 0, 0),
                up=Vec3(0, 0, -1),
                right=Vec3(1, 0, 0),
            ),
            FRONT_VIEW: OrthoView(
                eye=Vec3(0, 20, 500),
                target=Vec3(0, 20, 0),
                up=Vec3(0, 1, 0),
                right=Vec3(1, 0, 0),
            ),
            SIDE_VIEW: OrthoView(
                eye=Vec3(500, 20, 0),
                target=Vec3(0, 20, 0),
                up=Vec3(0, 1, 0),
                right=Vec3(0, 0, -1),
            ),
        }
        self._maximized_pane: int | None = None
        self._panning_pane: int | None = None
        self.keys_pressed: set[Qt.Key] = set()
        self._rotating_camera = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()
        self._last_frame_time = 0.0
        self._text_ready = False

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.18, 0.19, 0.20, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.scene.initialize_gl()
        Text.add_font(
            "BVHViewer",
            str(DEMO_DIR.parent / "font" / "Arial.ttf"),
            18,
        )
        Text.set_screen_size(self.window_width, self.window_height)
        self._text_ready = True

    def paintGL(self) -> None:
        self.makeCurrent()
        if self.four_view:
            gl.glClearColor(0.08, 0.08, 0.08, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        frame_time = self._frame_timer.elapsed() * 0.001
        delta_time = min(max(frame_time - self._last_frame_time, 0.0), 0.05)
        self._last_frame_time = frame_time
        self.advance_camera(delta_time)
        if self.four_view:
            gl.glClearColor(0.18, 0.19, 0.20, 1.0)
            gl.glEnable(gl.GL_SCISSOR_TEST)
            for rectangle, view, project in self._pane_draws():
                gl.glViewport(*rectangle)
                gl.glScissor(*rectangle)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                self.scene.draw(view, project, Mat4(), trace=self.trace)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            self._draw_view_labels()
        else:
            gl.glViewport(0, 0, self.window_width, self.window_height)
            self.scene.draw(
                self.camera.view,
                self.camera.projection,
                Mat4(),
                trace=self.trace,
            )
        if self.keys_pressed & self._MOVE_KEYS:
            self.update()

    def resizeGL(self, width: int, height: int) -> None:
        self.window_width = int(width * self.devicePixelRatio())
        self.window_height = int(height * self.devicePixelRatio())
        if self._text_ready:
            Text.set_screen_size(self.window_width, self.window_height)
        if self.four_view:
            if self._maximized_pane is not None:
                pane_width, pane_height = self.window_width, self.window_height
            else:
                _, _, pane_width, pane_height = self._four_view_rectangles()[0]
            aspect = pane_width / max(pane_height, 1)
        else:
            aspect = self.window_width / max(self.window_height, 1)
        self._set_perspective_projection(aspect)

    def set_four_view(self, enabled: bool) -> None:
        self.four_view = enabled
        self._maximized_pane = None
        self.resizeGL(
            round(self.window_width / self.devicePixelRatio()),
            round(self.window_height / self.devicePixelRatio()),
        )
        self.update()

    def toggle_maximized_pane(self, x: float, y: float) -> None:
        """Maximize the pane under (x, y), or restore the four-view layout.

        Parameters
        ----------
            x : float
                cursor position, window-local logical pixels
            y : float
                cursor position, window-local logical pixels
        """
        if not self.four_view:
            return
        if self._maximized_pane is not None:
            self._maximized_pane = None
        else:
            index = self._pane_index_at(x, y)
            if index is None:
                return
            self._maximized_pane = index
        self.resizeGL(
            round(self.window_width / self.devicePixelRatio()),
            round(self.window_height / self.devicePixelRatio()),
        )
        self.update()

    def _set_perspective_projection(self, aspect: float) -> None:
        self.camera.aspect = aspect
        self.camera.near = 0.05
        self.camera.far = 1500.0
        self.camera._projection = self.camera.set_projection(
            self.camera.zoom,
            self.camera.aspect,
            self.camera.near,
            self.camera.far,
        )

    def _four_view_rectangles(self) -> list[tuple[int, int, int, int]]:
        divider = self._DIVIDER_WIDTH
        left_width = max(1, (self.window_width - divider) // 2)
        bottom_height = max(1, (self.window_height - divider) // 2)
        right_x = left_width + divider
        top_y = bottom_height + divider
        right_width = max(1, self.window_width - right_x)
        top_height = max(1, self.window_height - top_y)
        return [
            (0, top_y, left_width, top_height),
            (right_x, top_y, right_width, top_height),
            (0, 0, left_width, bottom_height),
            (right_x, 0, right_width, bottom_height),
        ]

    def _pane_rectangles(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Return (pane_index, rectangle) for every pane currently being drawn."""
        if self._maximized_pane is not None:
            return [
                (self._maximized_pane, (0, 0, self.window_width, self.window_height))
            ]
        return list(enumerate(self._four_view_rectangles()))

    def _pane_index_at(self, x: float, y: float) -> int | None:
        """Return the pane index under a window-local logical-pixel position.

        Parameters
        ----------
            x : float
                cursor position, window-local logical pixels
            y : float
                cursor position, window-local logical pixels (Qt: 0 at the top)

        Returns
        -------
            int or None
                pane index, or None when four-view is off or (x, y) misses every pane
        """
        if not self.four_view:
            return None
        ratio = self.devicePixelRatio()
        device_x = x * ratio
        device_y = y * ratio
        for index, (rx, ry, rw, rh) in self._pane_rectangles():
            top = self.window_height - (ry + rh)
            if rx <= device_x < rx + rw and top <= device_y < top + rh:
                return index
        return None

    def _pane_draws(self) -> list[tuple[tuple[int, int, int, int], Mat4, Mat4]]:
        draws = []
        for index, rectangle in self._pane_rectangles():
            if index == PERSPECTIVE_VIEW:
                draws.append((rectangle, self.camera.view, self.camera.projection))
            else:
                _, _, pane_width, pane_height = rectangle
                view, project = self.ortho_views[index].matrices(
                    pane_width, pane_height
                )
                draws.append((rectangle, view, project))
        return draws

    def _draw_view_labels(self) -> None:
        gl.glViewport(0, 0, self.window_width, self.window_height)
        for index, (x, y, _, height) in self._pane_rectangles():
            screen_y = self.window_height - (y + height)
            Text.render_text(
                "BVHViewer",
                x + 10,
                screen_y + 20,
                _VIEW_LABELS[index],
                Vec3(0.82, 0.84, 0.86),
            )

    def advance_camera(self, delta_time: float) -> None:
        forward = float(Qt.Key.Key_W in self.keys_pressed) - float(
            Qt.Key.Key_S in self.keys_pressed
        )
        strafe = float(Qt.Key.Key_D in self.keys_pressed) - float(
            Qt.Key.Key_A in self.keys_pressed
        )
        if forward != 0.0 or strafe != 0.0:
            self.camera.move(forward, strafe, delta_time)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.add(event.key())
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.discard(event.key())
            self.update()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            if not self._is_perspective_position(position.x(), position.y()):
                self._rotating_camera = False
                return
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self._rotating_camera = True
            return
        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            position = event.position()
            index = self._pane_index_at(position.x(), position.y())
            if index is not None and index != PERSPECTIVE_VIEW:
                self._panning_pane = index
                self._last_mouse_x = position.x()
                self._last_mouse_y = position.y()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._rotating_camera and event.buttons() & Qt.MouseButton.LeftButton:
            position = event.position()
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self.camera.process_mouse_movement(diff_x, -diff_y)
            self.update()
            return
        if self._panning_pane is not None and event.buttons() & (
            Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton
        ):
            position = event.position()
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            _, _, _, pane_height = dict(self._pane_rectangles())[self._panning_pane]
            pane_height_logical = pane_height / self.devicePixelRatio()
            self.ortho_views[self._panning_pane].pan_by(
                diff_x, diff_y, pane_height_logical
            )
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating_camera = False
            return
        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning_pane = None
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        position = event.position()
        index = self._pane_index_at(position.x(), position.y())
        if index is not None and index != PERSPECTIVE_VIEW:
            view = self.ortho_views[index]
            wheel_steps = event.angleDelta().y() / 120.0
            view.half_height = max(5.0, min(view.half_height * 0.9**wheel_steps, 500.0))
        else:
            self.camera.process_mouse_scroll(event.angleDelta().y() * 0.01)
        self.update()

    def _is_perspective_position(self, x: float, y: float) -> bool:
        if not self.four_view:
            return True
        return self._pane_index_at(x, y) == PERSPECTIVE_VIEW


class MainWindow(QMainWindow):
    """The BVH viewport, menus and animation transport in one application."""

    def __init__(self, bvh_path: Path = DEFAULT_BVH) -> None:
        super().__init__()
        self._bvh_path: Path | None = None
        self.setWindowTitle("BVHViewer")
        self.setMinimumSize(720, 480)
        self.resize(1100, 760)
        self.setStyleSheet(_APP_STYLE)

        self.viewport = BvhViewport()
        viewport_container = QWidget.createWindowContainer(self.viewport, self)
        viewport_container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        viewport_container.setFocus()

        self.timeline = TimelineWidget(self)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(viewport_container, 1)
        layout.addWidget(self.timeline)
        self.setCentralWidget(central)

        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self.advance_playback)

        self._create_actions()
        self._create_menus()
        self._connect_timeline()
        self.statusBar().showMessage("No BVH file loaded")
        self.load_bvh(bvh_path, show_error=False)

    def _create_actions(self) -> None:
        self.open_action = QAction("&Open BVH...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setStatusTip("Load a BVH motion-capture file")
        self.open_action.triggered.connect(self.open_bvh_dialog)

        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)

        self.play_action = QAction("Play / Pause", self)
        self.play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.play_action.triggered.connect(self._handle_space)

        self.first_action = QAction("First Frame", self)
        self.first_action.setShortcut(QKeySequence(Qt.Key.Key_Home))
        self.first_action.triggered.connect(self.go_to_first_frame)

        self.previous_action = QAction("Previous Frame", self)
        self.previous_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.previous_action.triggered.connect(self.step_backward)

        self.next_action = QAction("Next Frame", self)
        self.next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.next_action.triggered.connect(self.step_forward)

        self.last_action = QAction("Last Frame", self)
        self.last_action.setShortcut(QKeySequence(Qt.Key.Key_End))
        self.last_action.triggered.connect(self.go_to_last_frame)

        self.trace_action = QAction("Trace Motion", self)
        self.trace_action.setCheckable(True)
        self.trace_action.setShortcut(QKeySequence(Qt.Key.Key_T))
        self.trace_action.toggled.connect(self._set_trace)

        self.four_view_action = QAction("Four Views", self)
        self.four_view_action.setCheckable(True)
        self.four_view_action.setShortcut(QKeySequence(Qt.Key.Key_4))
        self.four_view_action.toggled.connect(self.viewport.set_four_view)

        self.fullscreen_action = QAction("Full Screen", self)
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.setShortcut(QKeySequence(Qt.Key.Key_F))
        self.fullscreen_action.toggled.connect(self._set_fullscreen)

    def _create_menus(self) -> None:
        self.menuBar().setNativeMenuBar(False)
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.open_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        playback_menu = self.menuBar().addMenu("&Playback")
        playback_menu.addAction(self.play_action)
        playback_menu.addSeparator()
        playback_menu.addActions(
            [
                self.first_action,
                self.previous_action,
                self.next_action,
                self.last_action,
            ]
        )

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.trace_action)
        view_menu.addAction(self.four_view_action)
        view_menu.addAction(self.fullscreen_action)

    def _connect_timeline(self) -> None:
        self.timeline.frame_requested.connect(self.seek_frame)
        self.timeline.playback_range_changed.connect(self._playback_range_changed)
        self.timeline.fps_changed.connect(self.set_playback_fps)
        self.timeline.first_requested.connect(self.go_to_first_frame)
        self.timeline.previous_requested.connect(self.step_backward)
        self.timeline.play_toggled.connect(self.toggle_playback)
        self.timeline.next_requested.connect(self.step_forward)
        self.timeline.last_requested.connect(self.go_to_last_frame)

    def current_path(self) -> Path | None:
        """Return the loaded BVH path."""
        return self._bvh_path

    def open_bvh_dialog(self) -> None:
        """Ask for a BVH file and load it into the viewport."""
        directory = self._bvh_path.parent if self._bvh_path else DEMO_DIR / "bvh"
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open BVH File",
            str(directory),
            "BVH motion capture (*.bvh);;All files (*)",
        )
        if filename:
            self.load_bvh(Path(filename))

    def load_bvh(self, path: str | Path, show_error: bool = True) -> bool:
        """Load a BVH file and configure the timeline for its clip.

        Parameters
        ----------
            path : str or Path
                BVH file to load
            show_error : bool
                show a dialog if the file cannot be loaded

        Returns
        -------
            bool
                true when the new file was loaded
        """
        resolved = Path(path).expanduser().resolve()
        try:
            character = Bvh(resolved)
        except (OSError, ValueError, BvhParseError) as error:
            if show_error:
                QMessageBox.critical(
                    self,
                    "Unable to open BVH file",
                    f"Could not load {resolved.name}.\n\n{error}",
                )
            return False

        self.viewport.scene.set_character(character)
        self._bvh_path = resolved
        self.timeline.set_clip(character.num_frames, character.frame_time)
        self.set_playback_fps(self.timeline.playback_fps())
        self.set_playing(True)
        self.setWindowTitle(f"BVHViewer — {resolved.name}")
        self._sync_frame_display()
        self.viewport.update()
        return True

    def set_playing(self, playing: bool) -> None:
        """Start or stop clip playback.

        Parameters
        ----------
            playing : bool
                true to run the playback timer
        """
        if not self.viewport.scene.characters:
            playing = False
        self.viewport.scene.paused = not playing
        if playing:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()
        self.timeline.set_playing(playing)

    def toggle_playback(self) -> None:
        """Toggle between playing and paused."""
        self.set_playing(self.viewport.scene.paused)
        self._sync_frame_display()

    def _handle_space(self) -> None:
        """Maximize the pane under the mouse, or play/pause everywhere else."""
        local_pos = self.viewport.mapFromGlobal(QCursor.pos())
        over_viewport = (
            0 <= local_pos.x() < self.viewport.width()
            and 0 <= local_pos.y() < self.viewport.height()
        )
        if over_viewport and self.viewport.four_view:
            self.viewport.toggle_maximized_pane(local_pos.x(), local_pos.y())
            return
        self.toggle_playback()

    def set_playback_fps(self, fps: float) -> None:
        """Set the playback rate and update the animation timer.

        Parameters
        ----------
            fps : float
                requested playback frames per second
        """
        bounded_fps = max(1.0, min(float(fps), 240.0))
        self.timeline.set_playback_fps(bounded_fps)
        self._playback_timer.setInterval(max(1, round(1000.0 / bounded_fps)))
        self._sync_frame_display()

    def playback_fps(self) -> float:
        """Return the selected playback frame rate."""
        return self.timeline.playback_fps()

    def playback_interval_ms(self) -> int:
        """Return the current playback timer interval in milliseconds."""
        return self._playback_timer.interval()

    def seek_frame(self, frame: int) -> None:
        """Pause and display the requested timeline frame.

        Parameters
        ----------
            frame : int
                frame requested by the scrubber
        """
        self.set_playing(False)
        self.viewport.scene.seek(frame)
        self._sync_frame_display()
        self.viewport.update()

    def go_to_first_frame(self) -> None:
        """Pause and jump to the start of the clip."""
        range_start, _ = self.timeline.playback_range()
        self.seek_frame(range_start)

    def go_to_last_frame(self) -> None:
        """Pause and jump to the end of the clip."""
        _, range_end = self.timeline.playback_range()
        self.seek_frame(range_end)

    def step_forward(self) -> None:
        """Pause and move forward by one frame."""
        range_start, range_end = self.timeline.playback_range()
        current = self.viewport.scene.current_frame_number()
        target = range_start if current < range_start else min(current + 1, range_end)
        self.seek_frame(target)

    def step_backward(self) -> None:
        """Pause and move backwards by one frame."""
        range_start, range_end = self.timeline.playback_range()
        current = self.viewport.scene.current_frame_number()
        target = range_end if current > range_end else max(current - 1, range_start)
        self.seek_frame(target)

    def advance_playback(self) -> None:
        """Advance one timer tick within the selected playback range."""
        range_start, range_end = self.timeline.playback_range()
        self.viewport.scene.advance_in_range(range_start, range_end)
        self._sync_frame_display()
        self.viewport.update()

    def _sync_frame_display(self) -> None:
        frame = self.viewport.scene.current_frame_number()
        last_frame = max(0, self.viewport.scene.frame_count() - 1)
        range_start, range_end = self.timeline.playback_range()
        self.timeline.set_frame(frame)
        filename = self._bvh_path.name if self._bvh_path else "No clip"
        state = "Playing" if not self.viewport.scene.paused else "Paused"
        self.statusBar().showMessage(
            f"{filename}    Frame {frame} / {last_frame}    "
            f"Range {range_start} - {range_end}    "
            f"{self.playback_fps():.2f} fps    {state}"
        )

    def _playback_range_changed(self, range_start: int, range_end: int) -> None:
        del range_start, range_end
        self._sync_frame_display()

    def _set_trace(self, enabled: bool) -> None:
        self.viewport.trace = enabled
        self.viewport.update()

    def _set_fullscreen(self, enabled: bool) -> None:
        if enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._playback_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by the Qt event loop."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def _configure_surface_format() -> None:
    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bvh",
        type=Path,
        default=DEFAULT_BVH,
        help="path to the .bvh file to play (default: %(default)s)",
    )
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create and run the BVH viewer application.

    Parameters
    ----------
        argv : list[str] or None
            command-line arguments without the executable name

    Returns
    -------
        int
            Qt application exit status
    """
    args = _parse_args(argv)
    _configure_surface_format()
    app_type = DebugApplication if args.debug else QApplication
    app = app_type(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("BVHViewer")

    window = MainWindow(args.bvh)
    window.show()

    if args.smoketest is not None:

        def finish_smoketest() -> None:
            print("SMOKETEST OK")
            app.quit()

        QTimer.singleShot(args.smoketest, finish_smoketest)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
