"""OpenGL three-pose OBJ morph demo ported from NGL9Demos."""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from morph_mesh import adjust_weight, advance_punch, load_morph_mesh
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

MODEL_DIR = Path(__file__).parent / "models"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """A Bruce Lee mesh blended between a base pose and two punches."""

    def __init__(self) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(),
        )
        self.setTitle("MorphObj (OpenGL)")
        self.window_width = 1024
        self.window_height = 720
        self.project = Mat4()
        self.view = look_at(Vec3(0, 10, 40), Vec3(0, 10, 0), Vec3(0, 1, 0))
        self.weight_one = 0.0
        self.weight_two = 0.0
        self.left_direction = 1
        self.right_direction = 1
        self.animation_enabled = True
        self.mesh_data = load_morph_mesh(
            MODEL_DIR / "BrucePose1.obj",
            MODEL_DIR / "BrucePose2.obj",
            MODEL_DIR / "BrucePose3.obj",
        )
        self.left_timer = QTimer(self)
        self.left_timer.setInterval(4)
        self.left_timer.timeout.connect(self._advance_left)
        self.right_timer = QTimer(self)
        self.right_timer.setInterval(4)
        self.right_timer.timeout.connect(self._advance_right)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "Morph",
            str(shader_dir / "MorphVertex.glsl"),
            str(shader_dir / "MorphFragment.glsl"),
        )
        ShaderLib.use("Morph")
        ShaderLib.set_uniform("lightPosition", Vec3(2, 20, 2))

        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self.vao:
            self.vao.set_data(VertexData(self.mesh_data, len(self.mesh_data)))
            for location in range(6):
                self.vao.set_vertex_attribute_pointer(
                    location, 3, gl.GL_FLOAT, 18 * 4, location * 3 * 4
                )
            self.vao.set_num_indices(len(self.mesh_data))

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 18
        )

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
        model_view = self.view @ self._global_transform()
        ShaderLib.use("Morph")
        ShaderLib.set_uniform("MVP", self.project @ model_view)
        ShaderLib.set_uniform("MV", model_view)
        ShaderLib.set_uniform(
            "normalMatrix", Mat3.from_mat4(model_view).inverse().transposed()
        )
        ShaderLib.set_uniform("weight1", self.weight_one)
        ShaderLib.set_uniform("weight2", self.weight_two)
        with self.vao:
            self.vao.draw()
        Text.render_text(
            "Arial",
            10,
            25,
            f"Q/W pose one {self.weight_one:.2f}   A/S pose two {self.weight_two:.2f}",
            Vec3(1, 1, 1),
        )
        Text.render_text("Arial", 10, 50, "Z/X punch   Space pause", Vec3(1, 1, 1))

    def resizeGL(self, width: int, height: int) -> None:
        self.window_width = int(width * self.devicePixelRatio())
        self.window_height = int(height * self.devicePixelRatio())
        self.project = perspective(45.0, width / max(height, 1), 0.05, 350.0)
        Text.set_screen_size(width, height)

    def _advance_left(self) -> None:
        if not self.animation_enabled:
            return
        self.weight_one, self.left_direction, active = advance_punch(
            self.weight_one, self.left_direction
        )
        if not active:
            self.left_timer.stop()
        self.update()

    def _advance_right(self) -> None:
        if not self.animation_enabled:
            return
        self.weight_two, self.right_direction, active = advance_punch(
            self.weight_two, self.right_direction
        )
        if not active:
            self.right_timer.stop()
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Q:
            self.weight_one = adjust_weight(self.weight_one, -0.1)
        elif key == Qt.Key_W:
            self.weight_one = adjust_weight(self.weight_one, 0.1)
        elif key == Qt.Key_A:
            self.weight_two = adjust_weight(self.weight_two, -0.1)
        elif key == Qt.Key_S:
            self.weight_two = adjust_weight(self.weight_two, 0.1)
        elif key == Qt.Key_Z and not self.left_timer.isActive():
            self.weight_one = 0.0
            self.left_direction = 1
            self.left_timer.start()
        elif key == Qt.Key_X and not self.right_timer.isActive():
            self.weight_two = 0.0
            self.right_direction = 1
            self.right_timer.start()
        elif key == Qt.Key_Space:
            self.animation_enabled = not self.animation_enabled
        else:
            super().keyPressEvent(event)
        self.update()

    def closeEvent(self, event) -> None:
        self.left_timer.stop()
        self.right_timer.stop()
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
    window = MainWindow()
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
