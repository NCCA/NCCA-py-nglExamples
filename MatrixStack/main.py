#!/usr/bin/env -S uv run --script
"""
MatrixStack: an OpenGL-style push/pop matrix stack (OpenGL).

Demonstrates building a scene graph by hand with push_matrix()/pop_matrix()
instead of a Transform tree: three trolls, a ring of orbiting spheres, and a
reference grid, each pushed/popped around a shared stack.

Controls:
    I/O  increase / decrease the sphere ring's vertical wave frequency
    W/S  wireframe / solid
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import math
import sys
import traceback

import OpenGL.GL as gl
from matrix_stack import MatrixStack
from ncca.ngl import Mat3, Prims, Vec3, logger, look_at, perspective
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
        self.setTitle("MatrixStack (OpenGL)")
        self.stack = MatrixStack()
        self.rotation: float = 0.0
        self.freq: float = 1.0

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        self.stack.set_view(look_at(Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0)))

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 10.0, 10.0, 100)
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)

        self.startTimer(10)

    def load_matrices_to_shader(self) -> None:
        normal_matrix = Mat3.from_mat4(self.stack.mv()).inverse().transposed()
        ShaderLib.set_uniform("MVP", self.stack.mvp())
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        self.stack.push_matrix()
        self.stack.translate(0, 0, self.model_position.z)
        self.stack.translate(self.model_position.x, self.model_position.y, 0)
        self.stack.rotate_axis_angle(self.spin_x_face, 1, 0, 0)
        self.stack.rotate_axis_angle(self.spin_y_face, 0, 1, 0)

        self.stack.push_matrix()
        self.stack.translate(0, -0.65, 0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(-1.0, -1.85, -1.0)
        self.stack.rotate_axis_angle(45, 0, 1, 0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(1.0, -1.85, -1.0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        i = 0.0
        while i < 2.0 * math.pi:
            self.stack.push_matrix()
            x = math.cos(i) * 2.0
            z = math.sin(i) * 2.0
            y = math.sin(i * self.freq) * 0.5
            ShaderLib.set_uniform("Colour", abs(x), abs(y), abs(z), 1.0)
            self.stack.rotate_axis_angle(self.rotation, 0, 1, 0)
            self.stack.translate(x, y, z)
            self.stack.push_matrix()
            self.stack.scale(0.04, 0.04, 0.04)
            self.stack.rotate_axis_angle(self.rotation * 2, 0, 1, 0)
            self.load_matrices_to_shader()
            Primitives.draw("sphere")
            self.stack.pop_matrix()
            self.stack.pop_matrix()
            i += 0.05

        self.stack.push_matrix()
        self.stack.translate(0, -1.2, 0)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        self.load_matrices_to_shader()
        Primitives.draw("grid")
        self.stack.pop_matrix()
        self.stack.pop_matrix()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.stack.set_projection(perspective(45.0, float(w) / h, 0.05, 350.0))

    def timerEvent(self, event) -> None:
        self.rotation += 1.0
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_I:
            self.freq += 1.0
        elif key == Qt.Key_O:
            self.freq -= 1.0
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
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
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
