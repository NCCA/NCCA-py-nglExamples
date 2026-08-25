#!/usr/bin/env -S uv run --script
"""Spotlight demo: 4 animated cone-attenuation spotlights over a grid of teapots."""

import argparse
import math
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import Primitives, PySideEventHandlingMixin, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

_NUM_LIGHTS = 4
_LIGHT_COLOURS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.3, 0.4, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
]
_LIGHT_CENTRES = [(-4.0, -4.0), (4.0, -4.0), (-4.0, 4.0), (4.0, 4.0)]
_LIGHT_RADII = [2.5, 3.0, 2.0, 3.5]
_LIGHT_HEIGHT = 6.0


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
        self.setTitle("Spotlight")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.animate: bool = True
        self.time: float = 0.0
        self.grid_positions: list[tuple[float, float]] = [
            (float(x), float(z)) for x in range(-6, 7, 3) for z in range(-6, 7, 3)
        ]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 8, 16), Vec3(0, 0, 0), Vec3(0, 1, 0))

        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "Spotlight",
            str(shader_dir / "SpotlightVertex.glsl"),
            str(shader_dir / "SpotlightFragment.glsl"),
        )
        ShaderLib.use("Spotlight")
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 1.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 1.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 1.0)
        ShaderLib.set_uniform("material.shininess", 51.2)

        for i in range(_NUM_LIGHTS):
            r, g, b, a = _LIGHT_COLOURS[i]
            ShaderLib.set_uniform(f"light[{i}].diffuse", r, g, b, a)
            ShaderLib.set_uniform(f"light[{i}].specular", r, g, b, a)
            ShaderLib.set_uniform(f"light[{i}].direction", 0.0, -1.0, 0.0)
            ShaderLib.set_uniform(
                f"light[{i}].spotCosCutoff", math.cos(math.radians(25.0))
            )
            ShaderLib.set_uniform(
                f"light[{i}].spotCosInnerCutoff", math.cos(math.radians(15.0))
            )
            ShaderLib.set_uniform(f"light[{i}].spotExponent", 8.0)
            ShaderLib.set_uniform(f"light[{i}].constantAttenuation", 1.0)
            ShaderLib.set_uniform(f"light[{i}].linearAttenuation", 0.02)
            ShaderLib.set_uniform(f"light[{i}].quadraticAttenuation", 0.005)

        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "ground", 30, 30, 20, 20, Vec3(0, 1, 0))

        self._update_lights()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _update_lights(self) -> None:
        ShaderLib.use("Spotlight")
        for i in range(_NUM_LIGHTS):
            cx, cz = _LIGHT_CENTRES[i]
            radius = _LIGHT_RADII[i]
            phase = self.time + i * 1.7
            x = cx + math.cos(phase) * radius
            z = cz + math.sin(phase) * radius
            ShaderLib.set_uniform(f"light[{i}].position", x, _LIGHT_HEIGHT, z)

    def _on_tick(self) -> None:
        if self.animate:
            self.time += 0.03
            self._update_lights()
            self.update()

    def load_matrices(self, global_tx: Mat4, tx: Transform) -> None:
        m = global_tx @ tx.matrix()
        mvp = self.project @ self.view @ m
        normal_matrix = Mat3.from_mat4(m).inverse().transposed()
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("viewerPos", 0.0, 8.0, 16.0)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Spotlight")
        tx = Transform()
        self.load_matrices(global_tx, tx)
        Primitives.draw("ground")

        for x, z in self.grid_positions:
            tx.reset()
            tx.set_position(x, 0.5, z)
            self.load_matrices(global_tx, tx)
            Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 150.0)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        self.update()
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


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
