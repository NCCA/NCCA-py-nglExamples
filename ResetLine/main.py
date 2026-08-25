"""OpenGL primitive-restart line field ported from NGL9Demos."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from blade_field import RESTART_INDEX, animate_blades, create_blade_field
from ncca.ngl import Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    IndexVertexData,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """A field of animated blades drawn with one restarted line strip."""

    def __init__(self, rows: int = 120, cols: int = 120, seed: int = 7) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(),
        )
        self.setTitle("ResetLine (OpenGL primitive restart)")
        self.window_width = 1024
        self.window_height = 720
        self.project = Mat4()
        self.view = look_at(Vec3(0, 10, 10), Vec3(), Vec3(0, 1, 0))
        self.field = create_blade_field(rows=rows, cols=cols, seed=seed)
        self.vertices = self.field.vertices.copy()
        self.phase = 0.0
        self.animate = False
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(10)
        self.animation_timer.timeout.connect(self._tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.7, 0.7, 0.7, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "ResetLine",
            str(shader_dir / "LineVertex.glsl"),
            str(shader_dir / "LineFragment.glsl"),
        )
        ShaderLib.use("ResetLine")

        self.vao = VAOFactory.create_vao(VAOType.SIMPLE_INDEX, gl.GL_LINE_STRIP)
        with self.vao:
            self.vao.set_data(
                IndexVertexData(
                    self.vertices,
                    len(self.vertices),
                    self.field.indices,
                    gl.GL_UNSIGNED_INT,
                    gl.GL_DYNAMIC_DRAW,
                )
            )
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 6 * 4, 0)
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 6 * 4, 3 * 4)
        gl.glPrimitiveRestartIndex(int(RESTART_INDEX))
        self.animation_timer.start()

    def _global_transform(self) -> Mat4:
        transform = Mat4().rotate_y(self.spin_y_face) @ Mat4().rotate_x(
            self.spin_x_face
        )
        transform[3, 0] = self.model_position.x
        transform[3, 1] = self.model_position.y
        transform[3, 2] = self.model_position.z
        return transform

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use("ResetLine")
        ShaderLib.set_uniform(
            "MVP", self.project @ self.view @ self._global_transform()
        )
        gl.glEnable(gl.GL_PRIMITIVE_RESTART)
        with self.vao:
            self.vao.draw()
        gl.glDisable(gl.GL_PRIMITIVE_RESTART)

    def resizeGL(self, width: int, height: int) -> None:
        self.window_width = int(width * self.devicePixelRatio())
        self.window_height = int(height * self.devicePixelRatio())
        self.project = perspective(45.0, width / max(height, 1), 0.001, 100.0)

    def _tick(self) -> None:
        if self.animate:
            self.vertices = animate_blades(self.vertices, self.field.ranges, self.phase)
            self.phase += 0.05
            self.makeCurrent()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vao.get_buffer_id())
            gl.glBufferSubData(
                gl.GL_ARRAY_BUFFER,
                0,
                self.vertices.nbytes,
                np.ascontiguousarray(self.vertices),
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoketest", nargs="?", const=300, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow(args.rows, args.cols, args.seed)
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
