#!/usr/bin/env -S uv run --script
"""
ViewToWorldTransform: click to place objects in world space (OpenGL).

Shift+left-click unprojects the mouse position at the camera's far plane
into a world-space point (view_to_world.unproject_point) and places a cube
there. The HUD shows the last click's screen and world coordinates.

Controls:
    Shift+LMB  place a cube at the unprojected point
    Space  clear placed cubes
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from view_to_world import unproject_point


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
        self.setTitle("ViewToWorldTransform (OpenGL)")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.click_positions: list[Vec3] = []
        self.last_click_screen: tuple[int, int] | None = None

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        self.view = look_at(Vec3(0, 0, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 16
        )

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        tx = Transform()
        for position in self.click_positions:
            tx.set_position(position.x, position.y, position.z)
            mv = self.view @ global_tx @ tx.matrix()
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(mv).inverse().transposed()
            )
            Primitives.draw("cube")

        if self.last_click_screen is not None and self.click_positions:
            last = self.click_positions[-1]
            sx, sy = self.last_click_screen
            text = f"Pos=({sx},{sy}) World=({last.x:.2f},{last.y:.2f},{last.z:.2f})"
            Text.render_text("Arial", 10, 10, text, Vec3(1.0, 1.0, 1.0))

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 50.0)
        Text.set_screen_size(w, h)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self.click_positions.clear()
            self.last_click_screen = None
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            position = event.position()
            x, y = int(position.x()), int(position.y())
            view_projection = (self.project @ self.view).to_numpy()
            world = unproject_point(
                x, y, self.window_width, self.window_height, view_projection
            )
            self.click_positions.append(Vec3(*world))
            self.last_click_screen = (x, y)
            self.update()
        else:
            super().mousePressEvent(event)


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
