#!/usr/bin/env -S uv run --script
"""A WebGPU version of the PyNGL skinned-mesh import demo.

See ``main.py`` for the OpenGL version this mirrors, and ``mesh.py`` for
the impasse-based loader and skinning maths shared by both.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import ClassVar

import wgpu
from main import (
    DEFAULT_MODEL,
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
from mesh import SkinnedMesh
from ncca.ngl import FirstPersonCamera, Mat4, PerspMode, Vec3, look_at, ortho
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from webgpu_renderer import SkinWebGPURenderer
from wgpu.utils import get_default_device

_VIEW_LABELS = ("TOP", "PERSPECTIVE", "FRONT", "SIDE")


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
            5000.0,
            PerspMode.WebGPU,
        )
        view = look_at(self.eye + self.pan, self.target + self.pan, self.up)
        return view, project


class SkinWebGPUViewport(WebGPUWidget):
    """The WebGPU viewport: loads the mesh, skins it on the GPU, and draws it.

    Camera/navigation duplicates ``main.py``'s ``SkinViewport`` -- see the
    design spec for why this isn't shared: ``QOpenGLWindow`` and
    ``WebGPUWidget`` are unrelated Qt base classes, the same reason
    ``BVHViewer/main_webgpu.py`` duplicates ``BvhViewport`` rather than
    subclassing it.
    """

    _MOVE_KEYS: ClassVar = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D}
    _DIVIDER_WIDTH = 2

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("Skinned Mesh Import WebGPU")
        self.window_width, self.window_height = self.texture_size
        self.model_path = model_path
        self.mesh = SkinnedMesh(str(model_path))
        self.current_frame = 0

        self.four_view = False
        self._maximized_pane: int | None = None
        self._panning_pane: int | None = None
        self.keys_pressed: set[Qt.Key] = set()
        self._rotating_camera = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()
        self._last_frame_time = 0.0

        self._compute_view_setup()
        self._initialise_webgpu()

    # -------------------------------------------------------- camera setup

    def _compute_view_setup(self) -> None:
        """(Re)build the model-axis correction, camera and ortho panes for the current mesh."""
        z_up = self.model_path.suffix.lower() == ".md5mesh"
        self.model_matrix = Mat4().rotate_x(-90.0) if z_up else Mat4()

        bbox_min, bbox_max = self.mesh.bounding_box()

        def to_display(corner: list[float]) -> Vec3:
            if z_up:
                return Vec3(corner[0], corner[2], -corner[1])
            return Vec3(corner[0], corner[1], corner[2])

        a = to_display(bbox_min)
        b = to_display(bbox_max)
        lo = Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
        hi = Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))
        centre = Vec3((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5)
        height = float(max(hi.y - lo.y, 0.001))
        distance = height * 2.5

        eye = Vec3(centre.x, centre.y, hi.z + height * 1.5)
        self.camera = FirstPersonCamera(
            eye, centre, Vec3(0.0, 1.0, 0.0), 45.0, PerspMode.WebGPU
        )
        self.camera.speed = height * 0.4
        self._look_at(eye, centre)

        half_height = height * 0.75
        self.ortho_views: dict[int, WebGPUOrthoView] = {
            TOP_VIEW: WebGPUOrthoView(
                eye=Vec3(centre.x, hi.y + distance, centre.z),
                target=centre,
                up=Vec3(0.0, 0.0, -1.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            FRONT_VIEW: WebGPUOrthoView(
                eye=Vec3(centre.x, centre.y, hi.z + distance),
                target=centre,
                up=Vec3(0.0, 1.0, 0.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            SIDE_VIEW: WebGPUOrthoView(
                eye=Vec3(hi.x + distance, centre.y, centre.z),
                target=centre,
                up=Vec3(0.0, 1.0, 0.0),
                right=Vec3(0.0, 0.0, -1.0),
                half_height=half_height,
            ),
        }

    def _look_at(self, eye: Vec3, target: Vec3) -> None:
        """Point the FirstPersonCamera at ``target`` -- its constructor ignores ``look``."""
        direction = (target - eye).normalized()
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, direction.y))))
        yaw = math.degrees(math.atan2(direction.z, direction.x))
        self.camera.yaw = yaw
        self.camera.pitch = pitch
        self.camera._update_camera_vectors()

    # ------------------------------------------------------------- webgpu

    def _initialise_webgpu(self) -> None:
        self.device = get_default_device()
        self._create_render_buffer()
        self.renderer = SkinWebGPURenderer(self.device)
        self.renderer.set_mesh(self.mesh)
        self.start_update_timer(16)
        self.set_frame(0)
        self.update()

    def set_frame(self, frame: int) -> None:
        """Pose the mesh at the given timeline frame and request a repaint."""
        self.current_frame = frame
        time_seconds = frame / self.mesh.ticks_per_second()
        self.renderer.update_bones(self.mesh.bone_transforms(time_seconds))
        self.update()

    def load_model(self, model_path: Path) -> None:
        """Replace the current mesh with the one at ``model_path``.

        Raises whatever ``SkinnedMesh`` raises without touching the
        currently-loaded mesh -- same contract as ``SkinViewport.load_model``.
        """
        new_mesh = SkinnedMesh(str(model_path))
        self.model_path = model_path
        self.mesh = new_mesh
        self._compute_view_setup()
        # _compute_view_setup() just replaced self.camera with a fresh
        # FirstPersonCamera, whose constructor defaults to near=0.1/far=100
        # -- short of this demo's own guard, whose eye-to-target distance is
        # ~110 units, so the reloaded mesh would sit entirely past the far
        # clip plane and never appear. Re-derive the projection from the
        # current window size, the same call set_four_view() and
        # toggle_maximized_pane() already use to refresh it after a state
        # change with no resize event of its own.
        self.resizeWebGPU(self.window_width, self.window_height)
        self.renderer.set_mesh(self.mesh)
        self.set_frame(0)

    def paintWebGPU(self) -> None:
        frame_time = self._frame_timer.elapsed() * 0.001
        delta_time = min(max(frame_time - self._last_frame_time, 0.0), 0.05)
        self._last_frame_time = frame_time
        self._advance_camera(delta_time)

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.25, 0.25, 0.28, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        for index, rectangle, view, project, eye in self._pane_draws():
            x, y, width, height = rectangle
            render_pass.set_viewport(x, y, width, height, 0.0, 1.0)
            render_pass.set_scissor_rect(x, y, width, height)
            # The light sits at the viewer's own position (a "headlamp"),
            # the same trick main.py's _draw_mesh uses (light.position set
            # to the eye transformed by its own view matrix).
            self.renderer.update_camera(
                index, project @ view, self.model_matrix, eye, eye
            )
            self.renderer.render(render_pass, index)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        if self.four_view:
            self._draw_view_labels()
        self._update_colour_buffer()

        if self.keys_pressed & self._MOVE_KEYS:
            self.update()

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
        self.camera.far = 5000.0
        self.camera._projection = self.camera.set_projection(
            self.camera.zoom,
            aspect,
            self.camera.near,
            self.camera.far,
            PerspMode.WebGPU,
        )

    # --------------------------------------------------------- four-view

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

    def _pane_draws(
        self,
    ) -> list[tuple[int, tuple[int, int, int, int], Mat4, Mat4, Vec3]]:
        draws = []
        for index, rectangle in self._pane_rectangles():
            if index == PERSPECTIVE_VIEW:
                draws.append(
                    (
                        index,
                        rectangle,
                        self.camera.view,
                        self.camera.projection,
                        self.camera.eye,
                    )
                )
            else:
                _, _, pane_width, pane_height = rectangle
                ortho_view = self.ortho_views[index]
                view, project = ortho_view.matrices(pane_width, pane_height)
                draws.append(
                    (index, rectangle, view, project, ortho_view.eye + ortho_view.pan)
                )
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

    # ------------------------------------------------------------- input

    def _advance_camera(self, delta_time: float) -> None:
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
                diff_x, diff_y, pane_height / self.ratio
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
            view.half_height = max(
                5.0, min(view.half_height * 0.9**wheel_steps, 5000.0)
            )
        else:
            self.camera.process_mouse_scroll(event.angleDelta().y() * 0.01)
        self.update()


class MainWindow(ViewerMainWindow):
    """The existing SkinnedMeshImport application shell with a WebGPU viewport."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        super().__init__(model_path, viewport=SkinWebGPUViewport(model_path))

    def _update_title(self) -> None:
        self.setWindowTitle(
            f"SkinnedMeshImport WebGPU — {self.viewport.model_path.name}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app_type = DebugApplication if args.debug else QApplication
    app = app_type(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("SkinnedMeshImport WebGPU")

    window = MainWindow(Path(args.model))
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
