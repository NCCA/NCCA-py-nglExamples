#!/usr/bin/env -S uv run --script
"""
LookAtDemos: ngl::lookAt and ngl::perspective/ortho (OpenGL).

Combines NGL9Demos' SimpleLookAt and MultipleViews demos. Tab switches
between a single interactive perspective camera and a 2x2 grid comparing
that same perspective view against three fixed orthographic reference views
(top, front, side) of the identical scene.

Controls:
    Tab  toggle simple / multi-view mode
    LMB rotate  RMB pan  wheel zoom (perspective view only)  Space reset  Esc quit
"""

import argparse
import sys
import traceback

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Vec3, logger, look_at, ortho, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("LookAtDemos (OpenGL)")
        self.multi_view: bool = False
        self.mouse_global_tx: Mat4 = Mat4()

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 4.0, 4.0, 40)

    def _load_matrices(self, view: Mat4, project: Mat4, model: Mat4) -> None:
        ShaderLib.set_uniform("MVP", project @ view @ model)
        mv = view @ model
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(mv).inverse().transposed())

    def _draw_scene(self, view: Mat4, project: Mat4, global_tx: Mat4) -> None:
        self._load_matrices(view, project, global_tx)
        Primitives.draw("troll")
        self._load_matrices(view, project, global_tx @ Mat4().translate(0, -1.0, 0))
        Primitives.draw("grid")

    def _viewport_rect(self, quadrant: str) -> tuple[int, int, int, int]:
        half_w = self.window_width // 2
        half_h = self.window_height // 2
        return {
            "top": (0, half_h, half_w, half_h),
            "persp": (half_w, half_h, half_w, half_h),
            "front": (0, 0, half_w, half_h),
            "side": (half_w, 0, half_w, half_h),
        }[quadrant]

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        if not self.multi_view:
            gl.glViewport(0, 0, self.window_width, self.window_height)
            view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
            aspect = self.window_width / self.window_height
            project = perspective(45.0, aspect, 0.05, 350.0)
            self._draw_scene(view, project, self.mouse_global_tx)
            return

        # Each ortho pane gets its own -aspect..aspect bound, same as the
        # perspective pane's fov/aspect below -- a fixed -1..1 box stretches
        # X on a non-square pane (each quadrant here is half_w x half_h) and
        # would falsely suggest orthographic projection itself distorts
        # shapes.
        x, y, w, h = self._viewport_rect("top")
        gl.glViewport(x, y, w, h)
        aspect = w / h
        view = look_at(Vec3(0, 2, 0), Vec3(0, 0, 0), Vec3(0, 0, -1))
        self._draw_scene(view, ortho(-aspect, aspect, -1, 1, 0.1, 100), Mat4())

        x, y, w, h = self._viewport_rect("front")
        gl.glViewport(x, y, w, h)
        aspect = w / h
        view = look_at(Vec3(0, 0, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self._draw_scene(view, ortho(-aspect, aspect, -1, 1, 0.01, 200), Mat4())

        x, y, w, h = self._viewport_rect("side")
        gl.glViewport(x, y, w, h)
        aspect = w / h
        view = look_at(Vec3(2, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self._draw_scene(view, ortho(-aspect, aspect, -1, 1, 0.1, 100), Mat4())

        x, y, w, h = self._viewport_rect("persp")
        gl.glViewport(x, y, w, h)
        view = look_at(Vec3(0, 1, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))
        project = perspective(45.0, w / h, 0.01, 100.0)
        self._draw_scene(view, project, self.mouse_global_tx)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Tab:
            self.multi_view = not self.multi_view
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
