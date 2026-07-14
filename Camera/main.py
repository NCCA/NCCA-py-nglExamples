#!/usr/bin/env -S uv run --script
"""Camera demo: 4 selectable UVN cameras viewing a lit PBR scene."""

import argparse
import sys
import traceback

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger
from ncca.ngl.opengl import Primitives, ShaderLib, Text
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from uvn_camera import UVNCamera


class MainWindow(QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.mouse_global_tx: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Camera")

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

        self.cameras: list[UVNCamera] = []
        self.camera_index: int = 3
        self.rotation: float = 0.0
        self.light_on: list[bool] = [True, True, True]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        aspect = self.window_width / self.window_height
        self.cameras = [
            UVNCamera(
                Vec3(0, 0, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0
            ),
            UVNCamera(
                Vec3(0, 20, 0.001),
                Vec3(0, 0, 0),
                Vec3(0, 1, 0),
                65.0,
                aspect,
                0.2,
                150.0,
            ),
            UVNCamera(
                Vec3(20, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0
            ),
            UVNCamera(
                Vec3(8, 6, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0
            ),
        ]

        ShaderLib.load_shader(
            "PBR", "shaders/PBRVertex.glsl", "shaders/PBRFragment.glsl"
        )
        ShaderLib.use("PBR")
        # Matches NGL9Demos/Camera/src/NGLScene.cpp's own material/light
        # values (this shader is shared with ShadingModels, whose defaults
        # are unrelated and don't belong here).
        ShaderLib.set_uniform("lightPositions[0]", -2.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightColours[0]", 100.0, 100.0, 100.0)
        ShaderLib.set_uniform("lightPositions[1]", 2.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightColours[1]", 100.0, 100.0, 100.0)
        ShaderLib.set_uniform("lightPositions[2]", 0.0, 2.0, 1.0)
        ShaderLib.set_uniform("lightColours[2]", 100.0, 100.0, 100.0)
        ShaderLib.set_uniform("albedo", 0.950, 0.71, 0.29)
        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)
        ShaderLib.set_uniform("exposure", 2.2)

        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "ground", 30, 30, 20, 20, Vec3(0, 1, 0))

        Text.add_font("Arial", "Arial.ttf", 18)

    def _update_lights(self) -> None:
        for i in range(3):
            c = 100.0 if self.light_on[i] else 0.0
            ShaderLib.set_uniform(f"lightColours[{i}]", c, c, c)

    def load_matrices(self, camera: UVNCamera, tx: Transform) -> None:
        m = tx.matrix()
        mv = camera.view @ self.mouse_global_tx @ m
        mvp = camera.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("camPos", camera.eye.x, camera.eye.y, camera.eye.z)

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

        camera = self.cameras[self.camera_index]
        ShaderLib.use("PBR")
        self._update_lights()

        tx = Transform()
        tx.set_position(0, -0.5, 0)
        tx.set_scale(15, 1, 15)
        self.load_matrices(camera, tx)
        Primitives.draw("ground")

        tx.reset()
        tx.set_position(0, 0, 0)
        self.load_matrices(camera, tx)
        Primitives.draw("teapot")

        tx.reset()
        tx.set_position(-3, 0.25, 0)
        tx.set_rotation(0, self.rotation, 0)
        self.load_matrices(camera, tx)
        Primitives.draw("cube")

        tx.reset()
        tx.set_position(3, 0, 0)
        tx.set_scale(0.5, 0.5, 0.5)
        self.load_matrices(camera, tx)
        Primitives.draw("football")

        self._draw_help_text(camera)

        self.rotation = (self.rotation + 1.0) % 360.0
        self.update()

    def _draw_help_text(self, camera: UVNCamera) -> None:
        yellow = Vec3(1.0, 1.0, 0.0)
        white = Vec3(1.0, 1.0, 1.0)

        y = 20
        for line in (
            "Arrow keys : move active camera's eye",
            "1-4 : switch active camera",
            "r / y / p : roll / yaw / pitch active camera",
            "z / x / c : toggle lights 0 / 1 / 2",
            "+ / - : adjust field of view",
        ):
            Text.render_text("Arial", 10, y, line, white)
            y += 20

        y += 10
        Text.render_text(
            "Arial",
            10,
            y,
            f"Active camera {self.camera_index}  FOV {camera.fov:.1f}",
            yellow,
        )
        y += 24
        Text.render_text("Arial", 10, y, "View Matrix", white)
        for row in range(4):
            y += 20
            r = camera.view[row]
            Text.render_text(
                "Arial",
                10,
                y,
                f"[ {r[0]:+0.4f} {r[1]:+0.4f} {r[2]:+0.4f} {r[3]:+0.4f} ]",
                yellow,
            )

        y += 24
        lights = " ".join(
            f"L{i}:{'On' if on else 'Off'}" for i, on in enumerate(self.light_on)
        )
        Text.render_text("Arial", 10, y, lights, white)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        aspect = float(w) / h
        for camera in self.cameras:
            camera.set_shape(camera.fov, aspect, camera.near, camera.far)
        Text.set_screen_size(self.window_width, self.window_height)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        camera = self.cameras[self.camera_index]
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            self.camera_index = key - Qt.Key_1
        elif key == Qt.Key_Left:
            camera.move_eye(-0.5, 0, 0)
        elif key == Qt.Key_Right:
            camera.move_eye(0.5, 0, 0)
        elif key == Qt.Key_Up:
            camera.move_eye(0, 0, -0.5)
        elif key == Qt.Key_Down:
            camera.move_eye(0, 0, 0.5)
        elif key == Qt.Key_R:
            camera.roll(3.0)
        elif key == Qt.Key_Y:
            camera.yaw(3.0)
        elif key == Qt.Key_P:
            camera.pitch(3.0)
        elif key == Qt.Key_Z:
            self.light_on[0] = not self.light_on[0]
        elif key == Qt.Key_X:
            self.light_on[1] = not self.light_on[1]
        elif key == Qt.Key_C:
            self.light_on[2] = not self.light_on[2]
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            camera.set_shape(camera.fov + 1.0, camera.aspect, camera.near, camera.far)
        elif key == Qt.Key_Minus:
            camera.set_shape(camera.fov - 1.0, camera.aspect, camera.near, camera.far)
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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        # Use the vertical wheel delta (angleDelta().y()) -- .x() is only
        # populated by horizontal scroll gestures, which left plain vertical
        # wheel/trackpad scrolling doing almost nothing and made stray
        # horizontal trackpad noise cause small unintended zoom jumps.
        # Scaling by the delta magnitude (120 = one standard wheel notch)
        # instead of a fixed step also makes fast scrolling zoom
        # proportionally rather than being capped to one step per event.
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
