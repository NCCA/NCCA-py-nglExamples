#!/usr/bin/env -S uv run --script
"""ShadedGrid demo: an animated wave grid with a geometry-shader normal visualization."""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib, VAOFactory, VAOType
from ncca.ngl.opengl.abstract_vao import VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from wave_grid import build_wave_grid

_GRID_N = 40
_GRID_SIZE = 4.0


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
        self.setTitle("ShadedGrid")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.animate: bool = True
        self.offset: float = 0.0
        self.normal_size: float = 0.1
        self.draw_face_normals: bool = True
        self.draw_vertex_normals: bool = True
        self.vertex_count: int = 0

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 3, 6), Vec3(0, 0, 0), Vec3(0, 1, 0))

        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "Phong",
            str(shader_dir / "PhongVertex.glsl"),
            str(shader_dir / "PhongFragment.glsl"),
        )
        ShaderLib.use("Phong")
        ShaderLib.set_uniform("material.ambient", 0.329412, 0.223529, 0.027451, 1.0)
        ShaderLib.set_uniform("material.diffuse", 0.780392, 0.568627, 0.113725, 1.0)
        ShaderLib.set_uniform("material.specular", 0.992157, 0.941176, 0.807843, 1.0)
        ShaderLib.set_uniform("material.shininess", 57.8974)
        ShaderLib.set_uniform("light[0].position", 3.0, 2.0, 2.0)
        ShaderLib.set_uniform("light[0].ambient", 0.1, 0.1, 0.1, 1.0)
        ShaderLib.set_uniform("light[0].diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light[0].specular", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light[1].position", -3.0, 1.5, 2.0)
        ShaderLib.set_uniform("light[1].ambient", 0.05, 0.05, 0.05, 1.0)
        ShaderLib.set_uniform("light[1].diffuse", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("light[1].specular", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("light[2].position", 0.0, 1.0, -3.0)
        ShaderLib.set_uniform("light[2].ambient", 0.05, 0.05, 0.05, 1.0)
        ShaderLib.set_uniform("light[2].diffuse", 0.4, 0.4, 0.4, 1.0)
        ShaderLib.set_uniform("light[2].specular", 0.4, 0.4, 0.4, 1.0)

        ShaderLib.load_shader(
            "NormalViz",
            str(shader_dir / "normalVertex.glsl"),
            str(shader_dir / "normalFragment.glsl"),
            str(shader_dir / "normalGeo.glsl"),
        )
        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("vertNormalColour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("faceNormalColour", 1.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("drawFaceNormals", self.draw_face_normals)
        ShaderLib.set_uniform("drawVertexNormals", self.draw_vertex_normals)

        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        self._upload_grid()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _upload_grid(self) -> None:
        data = build_wave_grid(_GRID_N, _GRID_SIZE, self.offset)
        self.vertex_count = len(data) // 8
        self.vao.bind()
        self.vao.set_data(VertexData(data, self.vertex_count))
        stride = 8 * 4
        self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)
        self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, stride, 3 * 4)
        self.vao.set_vertex_attribute_pointer(2, 2, gl.GL_FLOAT, stride, 6 * 4)
        self.vao.set_num_indices(self.vertex_count)
        self.vao.unbind()

    def _on_tick(self) -> None:
        if self.animate:
            self.offset += 0.02
            self._upload_grid()
            self.update()

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

        mvp = self.project @ self.view @ global_tx
        normal_matrix = Mat3.from_mat4(global_tx).inverse().transposed()

        self.vao.bind()

        ShaderLib.use("Phong")
        ShaderLib.set_uniform("M", global_tx)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("viewerPos", 0.0, 3.0, 6.0)
        self.vao.draw()

        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalSize", self.normal_size)
        self.vao.draw()

        self.vao.unbind()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 150.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.normal_size += 0.01
        elif key == Qt.Key_Minus:
            self.normal_size = max(0.0, self.normal_size - 0.01)
        elif key == Qt.Key_1:
            self.draw_face_normals = not self.draw_face_normals
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("drawFaceNormals", self.draw_face_normals)
        elif key == Qt.Key_2:
            self.draw_vertex_normals = not self.draw_vertex_normals
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("drawVertexNormals", self.draw_vertex_normals)
        elif key == Qt.Key_U:
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
