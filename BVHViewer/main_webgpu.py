#!/usr/bin/env -S uv run --script
"""A WebGPU version of the PyNGL BVH animation viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import wgpu
from bvh_scene_webgpu import BvhWebGPUScene
from main import (
    DEFAULT_BVH,
    FRONT_VIEW,
    PERSPECTIVE_VIEW,
    SIDE_VIEW,
    TOP_VIEW,
    DebugApplication,
    OrthoView,
    _parse_args,
)
from main import (
    MainWindow as ViewerMainWindow,
)
from ncca.ngl import FirstPersonCamera, Mat4, PerspMode, Vec3, look_at, ortho
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

SHADER_PATH = Path(__file__).with_name("bvh_webgpu.wgsl")
_VIEW_LABELS = ("TOP", "PERSPECTIVE", "FRONT", "SIDE")


def viewport_clear_colour(four_view: bool) -> tuple[float, float, float, float]:
    del four_view
    return (0.18, 0.19, 0.20, 1.0)


class WebGPUOrthoView(OrthoView):
    """An orthographic pane using WebGPU's zero-to-one depth range."""

    def matrices(self, pane_width: int, pane_height: int) -> tuple[Mat4, Mat4]:
        half_width = self.half_height * pane_width / max(pane_height, 1)
        project = ortho(
            -half_width,
            half_width,
            -self.half_height,
            self.half_height,
            0.05,
            1500.0,
            PerspMode.WebGPU,
        )
        view = look_at(self.eye + self.pan, self.target + self.pan, self.up)
        return view, project


class BvhWebGPUViewport(WebGPUWidget):
    """The WebGPU viewport used by the BVH application window."""

    _MOVE_KEYS: ClassVar = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D}
    _DIVIDER_WIDTH = 2

    def __init__(self) -> None:
        super().__init__()
        self.camera = FirstPersonCamera(
            Vec3(0, 12, 100),
            Vec3(0, 12, 0),
            Vec3(0, 1, 0),
            45.0,
            PerspMode.WebGPU,
        )
        self.camera.speed = 25.0
        self.window_width, self.window_height = self.texture_size
        self.scene = BvhWebGPUScene()
        self.trace = False
        self.four_view = False
        self.ortho_views: dict[int, WebGPUOrthoView] = {
            TOP_VIEW: WebGPUOrthoView(
                eye=Vec3(0, 500, 0),
                target=Vec3(0, 0, 0),
                up=Vec3(0, 0, -1),
                right=Vec3(1, 0, 0),
            ),
            FRONT_VIEW: WebGPUOrthoView(
                eye=Vec3(0, 20, 500),
                target=Vec3(0, 20, 0),
                up=Vec3(0, 1, 0),
                right=Vec3(1, 0, 0),
            ),
            SIDE_VIEW: WebGPUOrthoView(
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
        self._initialise_webgpu()

    def _initialise_webgpu(self) -> None:
        self.device = get_default_device()
        self._create_render_buffer()
        self.scene.initialise_webgpu(self.device, SHADER_PATH)
        self.start_update_timer(16)
        self.update()

    def paintWebGPU(self) -> None:
        frame_time = self._frame_timer.elapsed() * 0.001
        delta_time = min(max(frame_time - self._last_frame_time, 0.0), 0.05)
        self._last_frame_time = frame_time
        self.advance_camera(delta_time)
        self.scene.prepare_frame()

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": viewport_clear_colour(self.four_view),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        for index, rectangle, view, project in self._pane_draws():
            x, y, width, height = rectangle
            render_pass.set_viewport(x, y, width, height, 0.0, 1.0)
            render_pass.set_scissor_rect(x, y, width, height)
            self.scene.draw_webgpu(
                render_pass,
                view,
                project,
                Mat4(),
                index,
                trace=self.trace,
            )
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        if self.four_view:
            self._draw_view_labels()
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.window_width = width
        self.window_height = height
        if self.four_view and self._maximized_pane is None:
            _, _, pane_width, pane_height = self._four_view_rectangles()[0]
            aspect = pane_width / max(pane_height, 1)
        else:
            aspect = width / max(height, 1)
        self._set_perspective_projection(aspect)
        self.update()

    def _set_perspective_projection(self, aspect: float) -> None:
        self.camera.aspect = aspect
        self.camera.near = 0.05
        self.camera.far = 1500.0
        self.camera._projection = self.camera.set_projection(
            self.camera.zoom,
            aspect,
            self.camera.near,
            self.camera.far,
            PerspMode.WebGPU,
        )

    def set_four_view(self, enabled: bool) -> None:
        self.four_view = enabled
        self._maximized_pane = None
        self.resizeWebGPU(self.window_width, self.window_height)

    def toggle_maximized_pane(self, x: float, y: float) -> None:
        if not self.four_view:
            return
        if self._maximized_pane is not None:
            self._maximized_pane = None
        else:
            index = self._pane_index_at(x, y)
            if index is None:
                return
            self._maximized_pane = index
        self.resizeWebGPU(self.window_width, self.window_height)

    def _four_view_rectangles(self) -> list[tuple[int, int, int, int]]:
        divider = self._DIVIDER_WIDTH
        left_width = max(1, (self.window_width - divider) // 2)
        top_height = max(1, (self.window_height - divider) // 2)
        right_x = left_width + divider
        bottom_y = top_height + divider
        right_width = max(1, self.window_width - right_x)
        bottom_height = max(1, self.window_height - bottom_y)
        return [
            (0, 0, left_width, top_height),
            (right_x, 0, right_width, top_height),
            (0, bottom_y, left_width, bottom_height),
            (right_x, bottom_y, right_width, bottom_height),
        ]

    def _pane_rectangles(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        if self._maximized_pane is not None:
            return [
                (self._maximized_pane, (0, 0, self.window_width, self.window_height))
            ]
        if self.four_view:
            return list(enumerate(self._four_view_rectangles()))
        return [(PERSPECTIVE_VIEW, (0, 0, self.window_width, self.window_height))]

    def _pane_draws(self) -> list[tuple[int, tuple[int, int, int, int], Mat4, Mat4]]:
        draws = []
        for index, rectangle in self._pane_rectangles():
            if index == PERSPECTIVE_VIEW:
                draws.append(
                    (index, rectangle, self.camera.view, self.camera.projection)
                )
            else:
                _, _, pane_width, pane_height = rectangle
                view, project = self.ortho_views[index].matrices(
                    pane_width, pane_height
                )
                draws.append((index, rectangle, view, project))
        return draws

    def _pane_index_at(self, x: float, y: float) -> int | None:
        if not self.four_view:
            return None
        device_x = x * self.ratio
        device_y = y * self.ratio
        for index, (rx, ry, width, height) in self._pane_rectangles():
            if rx <= device_x < rx + width and ry <= device_y < ry + height:
                return index
        return None

    def _draw_view_labels(self) -> None:
        colour = QColor(209, 214, 219)
        for index, (x, y, _, _) in self._pane_rectangles():
            self.render_text(
                round(x / self.ratio) + 10,
                round(y / self.ratio) + 20,
                _VIEW_LABELS[index],
                12,
                colour=colour,
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
        if event.key() == Qt.Key_Escape:
            # The viewport is embedded in the QMainWindow rather than being the
            # top level thing itself, so self.close() would take the viewport
            # away and leave the shell and its transport sitting there. Closing
            # every top level window instead sends the QMainWindow a real close
            # event, so its own closeEvent still gets to stop the timers.
            QApplication.closeAllWindows()
            return
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
        position = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self.four_view
                and self._pane_index_at(position.x(), position.y()) != PERSPECTIVE_VIEW
            ):
                return
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self._rotating_camera = True
            return
        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            index = self._pane_index_at(position.x(), position.y())
            if index is not None and index != PERSPECTIVE_VIEW:
                self._panning_pane = index
                self._last_mouse_x = position.x()
                self._last_mouse_y = position.y()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if self._rotating_camera and event.buttons() & Qt.MouseButton.LeftButton:
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
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            _, _, _, pane_height = dict(self._pane_rectangles())[self._panning_pane]
            self.ortho_views[self._panning_pane].pan_by(
                diff_x,
                diff_y,
                pane_height / self.ratio,
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


class MainWindow(ViewerMainWindow):
    """The existing BVH application shell with a WebGPU viewport."""

    def __init__(self, bvh_path: Path = DEFAULT_BVH) -> None:
        super().__init__(bvh_path, viewport=BvhWebGPUViewport())

    def load_bvh(self, path: str | Path, show_error: bool = True) -> bool:
        loaded = super().load_bvh(path, show_error)
        if loaded:
            self.setWindowTitle(f"BVHViewer WebGPU — {Path(path).name}")
        return loaded


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app_type = DebugApplication if args.debug else QApplication
    app = app_type(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("BVHViewer WebGPU")

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
