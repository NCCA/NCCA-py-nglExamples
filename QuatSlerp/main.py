#!/usr/bin/env -S uv run --script
"""QuatSlerp demo: slerp between two quaternion orientations of a teapot, with a
Qt Designer side panel (start/end rotation, quaternion readouts, interpolation
slider) matching the original NGL9Demos/QuatSlerp UI."""

import sys
import traceback

import OpenGL.GL as gl
from ncca.ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    Quaternion,
    ShaderLib,
    Vec3,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.widgets import Vec3Widget
from PySide6.QtCore import QFile, Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget


class Loader(QUiLoader):
    """Custom QUiLoader so the promoted Vec3Widget resolves to the real class."""

    def createWidget(
        self, class_name: str, parent: QWidget | None = None, name: str = ""
    ) -> QWidget:
        if class_name == "Vec3Widget":
            return Vec3Widget(parent)
        return super().createWidget(class_name, parent, name)


class QuatSlerpScene(QOpenGLWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.mouse_global_tx: Mat4 = Mat4()
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()

        self.window_width: int = 1024
        self.window_height: int = 720

        self.rotate: bool = False
        self.translate: bool = False
        self.spin_x_face: int = 0
        self.spin_y_face: int = 0
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.1

        self.start_rotation = Vec3(45, 90, 80)
        self.end_rotation = Vec3(-300, 270, 360)
        self.start_quat = Quaternion.from_mat4(self._euler_mat4(self.start_rotation))
        self.end_quat = Quaternion.from_mat4(self._euler_mat4(self.end_rotation))
        self.interp: float = 0.0

    @staticmethod
    def _euler_mat4(rotation: Vec3) -> Mat4:
        return (
            Mat4().rotate_z(rotation.z)
            @ Mat4().rotate_y(rotation.y)
            @ Mat4().rotate_x(rotation.x)
        )

    def set_start_rotation(self, rotation: Vec3) -> None:
        self.start_rotation = rotation
        self.start_quat = Quaternion.from_mat4(self._euler_mat4(rotation))
        self.update()

    def set_end_rotation(self, rotation: Vec3) -> None:
        self.end_rotation = rotation
        self.end_quat = Quaternion.from_mat4(self._euler_mat4(rotation))
        self.update()

    def set_interp(self, value: float) -> None:
        self.interp = value
        self.update()

    def interpolated_quat(self) -> Quaternion:
        return self.start_quat.slerp(self.end_quat, self.interp)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 8), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()

    def _model_matrix(self, quat: Quaternion, position: Vec3) -> Mat4:
        """Build a model matrix from a quaternion rotation plus a position.

        NOTE: ncca.ngl's Transform class only builds rotation from Euler
        angles (set_rotation), it has no way to set rotation from a Mat4, so
        we assemble the model matrix directly instead of going via Transform.
        """
        model = quat.to_mat4()
        model[3, 0] = position.x
        model[3, 1] = position.y
        model[3, 2] = position.z
        return model

    def load_matrices_to_shader(self, model: Mat4) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        mv = self.view @ self.mouse_global_tx @ model
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        interp_quat = self.interpolated_quat()

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)

        model = self._model_matrix(interp_quat, Vec3(0, 0, 0))
        self.load_matrices_to_shader(model)
        Primitives.draw("teapot")

        model = self._model_matrix(self.start_quat, Vec3(-2, 0, 0))
        self.load_matrices_to_shader(model)
        Primitives.draw("teapot")

        model = self._model_matrix(self.end_quat, Vec3(2, 0, 0))
        self.load_matrices_to_shader(model)
        Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True
        self.setFocus()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        num_pixels = event.angleDelta()
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
        self.update()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QuatSlerp")
        self._load_ui()

        self.scene = QuatSlerpScene()
        self.scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.centralWidget().layout().addWidget(self.scene, 0, 0)

        self.start_rotation_widget.set_range(-360.0, 360.0)
        self.start_rotation_widget.set_single_step(1.0)
        self.start_rotation_widget.set_value(self.scene.start_rotation)
        self.end_rotation_widget.set_range(-360.0, 360.0)
        self.end_rotation_widget.set_single_step(1.0)
        self.end_rotation_widget.set_value(self.scene.end_rotation)

        self.start_rotation_widget.valueChanged.connect(self._start_rotation_changed)
        self.end_rotation_widget.valueChanged.connect(self._end_rotation_changed)
        self.interp_slider.valueChanged.connect(self._interp_changed)

        self._update_readouts()

    def _load_ui(self) -> None:
        loader = Loader()
        ui_file = QFile("ui/MainWindow.ui")
        ui_file.open(QFile.ReadOnly)
        loaded_ui = loader.load(ui_file, self)
        self.setCentralWidget(loaded_ui)
        for child in loaded_ui.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self, name, child)
        ui_file.close()

    def _start_rotation_changed(self, value: Vec3) -> None:
        self.scene.set_start_rotation(value)
        self._update_readouts()

    def _end_rotation_changed(self, value: Vec3) -> None:
        self.scene.set_end_rotation(value)
        self._update_readouts()

    def _interp_changed(self, value: int) -> None:
        self.scene.set_interp(value / 1000.0)
        self._update_readouts()

    def _update_readouts(self) -> None:
        self.start_quat_edit.setText(str(self.scene.start_quat))
        self.end_quat_edit.setText(str(self.scene.end_quat))
        interp_quat = self.scene.interpolated_quat()
        self.interp_quat_edit.setText(str(interp_quat))
        self.matrix_edit.setPlainText(str(interp_quat.to_mat4()))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
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
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv
    if "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
