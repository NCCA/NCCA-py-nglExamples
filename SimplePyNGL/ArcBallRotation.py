#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.
Enhanced with Maya/Blender-style mouse controls with faster, more responsive rotation.
"""

import math
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    Prims,
    ShaderLib,
    Vec3,
    logger,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

PBR_SHADER = "pbr"


class ArcballCamera:
    """
    An arcball camera system with improved responsiveness and Maya/Blender-style controls.
    """

    def __init__(self):
        # Camera position and target
        self.eye = Vec3(0.0, 1.0, 4.0)
        self.target = Vec3(0.0, 0.0, 0.0)
        self.up = Vec3(0.0, 1.0, 0.0)

        # Arcball parameters
        self.distance = 4.0
        self.min_distance = 0.1
        self.max_distance = 100.0

        # Rotation as quaternion for smooth, gimbal-lock-free rotation
        self.quaternion = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]

        # Mouse interaction state
        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = [0, 0]

        # IMPROVED SENSITIVITY - Much more responsive!
        self.rotate_sensitivity = 1.0  # Increased from 0.005
        self.pan_sensitivity = 0.01  # Increased from 0.001
        self.zoom_sensitivity = 0.15  # Slightly increased

        # Track mouse speed for acceleration
        self.last_mouse_move_time = 0
        self.mouse_velocity = [0, 0]

    def screen_to_arcball(self, x, y, width, height):
        """
        Convert screen coordinates to arcball vector with improved responsiveness.
        """
        # Normalize to [-1, 1] range but make it more sensitive near edges
        x = (2.0 * x - width) / width
        y = (height - 2.0 * y) / height

        # Calculate length with improved sensitivity
        length = math.sqrt(x * x + y * y)
        if length > 1.0:
            x /= length
            y /= length
            length = 1.0

        # Enhanced arcball with more responsive feel
        z = math.sqrt(1.0 - length * length)

        # Return enhanced vector with sensitivity applied
        return [x * self.rotate_sensitivity, y * self.rotate_sensitivity, z]

    def quat_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

    def quat_normalize(self, q):
        """Normalize a quaternion."""
        length = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if length > 0:
            return [q[0] / length, q[1] / length, q[2] / length, q[3] / length]
        return q

    def quat_to_matrix(self, q):
        """Convert quaternion to 4x4 rotation matrix."""
        w, x, y, z = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return Mat4(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                0,
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                0,
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
                0,
                0,
                0,
                0,
                1,
            ]
        )

    def rotate_vector_by_quaternion(self, vec, quat):
        """Rotate a vector by a quaternion."""
        # Convert Vec3 to [x, y, z]
        v = [vec.x, vec.y, vec.z]
        w, x, y, z = quat

        # Calculate rotation using q * v * q^-1
        uv = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]]

        uuv = [y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0]]

        return Vec3(
            v[0] + 2.0 * (uv[0] * w + uuv[0]),
            v[1] + 2.0 * (uv[1] * w + uuv[1]),
            v[2] + 2.0 * (uv[2] * w + uuv[2]),
        )

    def start_rotation(self, x, y, width, height):
        """Start rotation operation."""
        self.is_rotating = True
        self.last_arcball = self.screen_to_arcball(x, y, width, height)

    def update_rotation(self, x, y, width, height):
        """Update rotation with improved sensitivity and acceleration."""
        if not self.is_rotating:
            return

        current_arcball = self.screen_to_arcball(x, y, width, height)

        # Calculate rotation quaternion
        dot_product = sum(
            a * b for a, b in zip(self.last_arcball, current_arcball, strict=False)
        )

        # Handle case where vectors are opposite (180-degree rotation)
        if dot_product < -0.999999:
            # Find an orthogonal vector for 180-degree rotation
            if abs(self.last_arcball[0]) < 0.1:
                axis = [0, -self.last_arcball[2], self.last_arcball[1]]
            else:
                axis = [-self.last_arcball[2], 0, self.last_arcball[0]]

            axis = self.quat_normalize(axis)
            rotation_quat = [0, axis[0], axis[1], axis[2]]
        else:
            # Regular arcball rotation with acceleration
            cross_product = [
                self.last_arcball[1] * current_arcball[2]
                - self.last_arcball[2] * current_arcball[1],
                self.last_arcball[2] * current_arcball[0]
                - self.last_arcball[0] * current_arcball[2],
                self.last_arcball[0] * current_arcball[1]
                - self.last_arcball[1] * current_arcball[0],
            ]

            rotation_quat = [
                dot_product,
                cross_product[0],
                cross_product[1],
                cross_product[2],
            ]
            rotation_quat = self.quat_normalize(rotation_quat)

        # Apply rotation
        self.quaternion = self.quat_multiply(rotation_quat, self.quaternion)
        self.quaternion = self.quat_normalize(self.quaternion)

        self.last_arcball = current_arcball
        self.last_mouse_pos = [x, y]

    def end_rotation(self):
        """End rotation operation."""
        self.is_rotating = False

    def start_panning(self, x, y):
        """Start panning operation."""
        self.is_panning = True
        self.last_mouse_pos = [x, y]

    def update_panning(self, x, y):
        """Update panning based on mouse movement."""
        if not self.is_panning:
            return

        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]

        # Get camera's right and up vectors from current view
        current_view = self.get_view_matrix()

        # Extract right and up vectors from view matrix
        right = Vec3(
            current_view[0][0], current_view[1][0], current_view[2][0]
        ).normalized()
        up = Vec3(
            current_view[0][1], current_view[1][1], current_view[2][1]
        ).normalized()

        # Scale pan by distance and field of view with improved sensitivity
        fov_scale = math.tan(math.radians(22.5))  # Half of 45-degree FOV
        pan_scale = self.distance * fov_scale * self.pan_sensitivity

        # Update target position (camera position follows target)
        self.target = self.target - right * dx * pan_scale + up * dy * pan_scale
        self.eye = self.eye - right * dx * pan_scale + up * dy * pan_scale

        self.last_mouse_pos = [x, y]

    def end_panning(self):
        """End panning operation."""
        self.is_panning = False

    def zoom(self, delta):
        """Zoom in/out based on mouse wheel with improved responsiveness."""
        # More responsive zoom with exponential scaling
        zoom_factor = math.exp(-delta * self.zoom_sensitivity * 0.001)
        self.distance = max(
            self.min_distance, min(self.max_distance, self.distance * zoom_factor)
        )

        # Maintain direction from target to eye
        direction = (self.eye - self.target).normalized()
        self.eye = self.target + direction * self.distance

    def get_view_matrix(self):
        """Get the current view matrix with applied rotation."""
        # Calculate the rotated position relative to target
        relative_eye = self.eye - self.target

        # Rotate the relative position by the quaternion
        rotated_relative = self.rotate_vector_by_quaternion(
            relative_eye, self.quaternion
        )

        # Calculate new eye position
        rotated_eye = self.target + rotated_relative

        # Calculate up vector rotation
        rotated_up = self.rotate_vector_by_quaternion(self.up, self.quaternion)

        return look_at(rotated_eye, self.target, rotated_up)

    def reset(self):
        """Reset camera to default position."""
        self.eye = Vec3(0.0, 2.0, 4.0)
        self.target = Vec3(0.0, 0.0, 0.0)
        self.up = Vec3(0.0, 1.0, 0.0)
        self.distance = 4.0
        self.quaternion = [1.0, 0.0, 0.0, 0.0]


class MainWindow(QOpenGLWindow):
    """
    The main window for the OpenGL application with improved Maya/Blender-style controls.
    """

    def __init__(self, parent: object = None) -> None:
        super().__init__()

        # --- Camera System ---
        self.camera = ArcballCamera()
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()

        # --- Window and UI Attributes ---
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("SimplePyNGL - Fast Maya/Blender Style Controls")

    def initializeGL(self) -> None:
        """Initialize OpenGL context and create geometry."""
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        # Initialize camera
        self.view = self.camera.get_view_matrix()

        # Load shaders and create geometry
        ShaderLib.load_shader(
            PBR_SHADER, "shaders/PBRVertex.glsl", "shaders/PBRFragment.glsl"
        )
        ShaderLib.use(PBR_SHADER)

        eye = self.camera.eye
        to = self.camera.target
        up = self.camera.up
        self.view = look_at(eye, to, up)
        ShaderLib.set_uniform("camPos", eye)

        self.light_pos = Vec3(0.0, 2.0, 2.0)
        ShaderLib.set_uniform("lightPosition", self.light_pos)
        ShaderLib.set_uniform("lightColor", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("exposure", 2.2)
        ShaderLib.set_uniform("albedo", 0.950, 0.71, 0.29)
        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)

        Primitives.create(Prims.TRIANGLE_PLANE, "floor", 20, 20, 1, 1, Vec3(0, 1, 0))
        ShaderLib.print_registered_uniforms(PBR_SHADER)

        ShaderLib.use(DefaultShader.CHECKER)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("checkOn", True)
        ShaderLib.set_uniform("lightPos", self.light_pos)
        ShaderLib.set_uniform("colour1", 0.9, 0.9, 0.9, 1.0)
        ShaderLib.set_uniform("colour2", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("checkSize", 60.0)
        ShaderLib.print_registered_uniforms(DefaultShader.CHECKER)

        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "floor", 20, 20, 1, 1, Vec3(0, 1, 0))

    def load_matrices_to_shader(self) -> None:
        """Load transformation matrices to shader."""
        ShaderLib.use(PBR_SHADER)

        # Create identity model matrix since we're using camera rotation now
        model = Mat4.identity()

        transform_dtype = np.dtype(
            [
                ("MVP", np.float32, (4, 4)),
                ("normal_matrix", np.float32, (4, 4)),
                ("M", np.float32, (4, 4)),
            ]
        )

        t = np.zeros(1, dtype=transform_dtype)

        MVP = self.project @ self.view @ model
        normal_matrix = self.view @ model
        normal_matrix = normal_matrix.inverse().transposed()

        t[0]["MVP"] = MVP.to_numpy()
        t[0]["normal_matrix"] = normal_matrix.to_numpy()
        t[0]["M"] = model.to_numpy()
        ShaderLib.set_uniform_buffer("TransformUBO", data=t.data, size=t.data.nbytes)

    def paintGL(self) -> None:
        """Main rendering loop."""
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Update view matrix from camera
        self.view = self.camera.get_view_matrix()

        self.load_matrices_to_shader()
        Primitives.draw("teapot")

        ShaderLib.use(DefaultShader.CHECKER)
        tx = Mat4().translate(0.0, -0.45, 0.0)
        mvp = self.project @ self.view @ tx
        normal_matrix = Mat3.from_mat4(mvp)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        Primitives.draw("floor")

    def resizeGL(self, w: int, h: int) -> None:
        """Handle window resize."""
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

    def keyPressEvent(self, event) -> None:
        """Handle keyboard input with speed adjustment keys."""
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            # Reset camera to default position
            self.camera.reset()
        elif key == Qt.Key_F:
            # Frame the object (focus on origin)
            self.camera.target = Vec3(0.0, 0.0, 0.0)
            self.camera.eye = self.camera.target + Vec3(0.0, 2.0, 4.0)
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:  # + key
            # Increase rotation speed
            self.camera.rotate_sensitivity *= 1.2
            print(f"Rotation speed: {self.camera.rotate_sensitivity:.2f}")
        elif key == Qt.Key_Minus:  # - key
            # Decrease rotation speed
            self.camera.rotate_sensitivity *= 0.8
            print(f"Rotation speed: {self.camera.rotate_sensitivity:.2f}")

        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement for rotation and panning."""
        position = event.position()
        x, y = int(position.x()), int(position.y())

        # Update rotation
        if self.camera.is_rotating:
            self.camera.update_rotation(x, y, self.window_width, self.window_height)

        # Update panning
        elif self.camera.is_panning:
            self.camera.update_panning(x, y)

        self.update()

    def mousePressEvent(self, event) -> None:
        """Handle mouse button presses."""
        position = event.position()
        x, y = int(position.x()), int(position.y())

        # Maya/Blender-style button mapping:
        # Left button: Rotate (Orbit)
        # Middle button: Pan
        # Right button: Context menu (handled by Qt) or zoom
        if event.button() == Qt.LeftButton:
            self.camera.start_rotation(x, y, self.window_width, self.window_height)
        elif event.button() == Qt.MiddleButton:
            self.camera.start_panning(x, y)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse button releases."""
        if event.button() == Qt.LeftButton:
            self.camera.end_rotation()
        elif event.button() == Qt.MiddleButton:
            self.camera.end_panning()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        self.camera.zoom(delta)
        self.update()


class DebugApplication(QApplication):
    """Enhanced debug application with better error reporting."""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in debug mode with FAST Maya/Blender-style controls")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Set up OpenGL context
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    # Choose application type
    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create and show window
    window = MainWindow()
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
