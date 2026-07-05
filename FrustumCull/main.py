#!/usr/bin/env -S uv run --script
"""FrustumCull: culls a 3D grid of spheres against a test camera's view
frustum (6-plane extraction + sphere/plane distance test) and only draws the
spheres that are inside or intersecting."""

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
    VAOFactory,
    VAOType,
    Vec3,
    logger,
)
from ncca.ngl.abstract_vao import VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from uvn_camera import FrustumCamera

# Camera key-movement modes, matching NGL9Demos/FrustumCull's e/l/b/s toggle.
MOVE_EYE = "eye"
MOVE_LOOK = "look"
MOVE_BOTH = "both"
MOVE_SLIDE = "slide"
KEY_INCREMENT = 0.2


class MainWindow(QOpenGLWindow):
    """
    The main window for the OpenGL application.

    Inherits from QOpenGLWindow to provide a canvas for OpenGL rendering within a PySide6 GUI.
    It handles user input (mouse, keyboard) for camera control and manages the OpenGL context.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initializes the main window and sets up default scene parameters.
        """
        super().__init__()
        # --- Camera and Transformation Attributes ---
        self.mouse_global_tx: Mat4 = (
            Mat4()
        )  # Global transformation matrix controlled by the mouse
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)
        self.model_position: Vec3 = Vec3()  # Position of the model in world space

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height
        self.setTitle("FrustumCull")

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.original_x_rotation: int = (
            0  # Initial X position of the mouse when a rotation starts
        )
        self.original_y_rotation: int = (
            0  # Initial Y position of the mouse when a rotation starts
        )
        self.original_x_pos: int = (
            0  # Initial X position of the mouse when a translation starts
        )
        self.original_y_pos: int = (
            0  # Initial Y position of the mouse when a translation starts
        )
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming

        self.active_camera_index: int = 1
        self.last_drawn: int = 0
        self.move_mode: str = MOVE_EYE

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        aspect = self.window_width / self.window_height
        self.test_camera = FrustumCamera(
            Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0), 45.0, aspect, 2.0, 15.0
        )
        self.observer_camera = FrustumCamera(
            Vec3(0, 40, 0.001), Vec3(0, 0, 0), Vec3(0, 1, 0), 60.0, aspect, 0.5, 100.0
        )
        self.active_camera_index = 1

        ShaderLib.load_shader(
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        )
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 12)

        self.grid_positions = [
            Vec3(x, y, z)
            for x in range(-20, 21, 4)
            for y in range(-8, 9, 4)
            for z in range(-20, 21, 4)
        ]

        # 12-edge wireframe box (4 near + 4 far + 4 connecting), rebuilt from the
        # test camera's frustum corners each frame since the camera can move.
        self.frustum_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)

    # Indices into FrustumCamera.corners == [ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr]
    _FRUSTUM_EDGES = (
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),  # near face
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),  # far face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # connecting edges
    )

    def _draw_frustum(self, camera: FrustumCamera) -> None:
        corners = camera.corners
        verts: list[float] = []
        for a, b in self._FRUSTUM_EDGES:
            for corner in (corners[a], corners[b]):
                verts.extend((corner.x, corner.y, corner.z))
        data = np.array(verts, dtype=np.float32)
        vertex_count = len(self._FRUSTUM_EDGES) * 2

        with self.frustum_vao as vao:
            vao.set_data(VertexData(data, vertex_count))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(vertex_count)
            vao.draw()

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

        camera = (
            self.test_camera if self.active_camera_index == 0 else self.observer_camera
        )

        ShaderLib.use("Phong")
        drawn = 0
        for pos in self.grid_positions:
            state = self.test_camera.is_sphere_in_frustum(pos, 1.0)
            if state == "OUTSIDE":
                continue
            drawn += 1
            mv = (
                camera.view
                @ self.mouse_global_tx
                @ Mat4().translate(pos.x, pos.y, pos.z)
            )
            mvp = camera.project @ mv
            normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            ShaderLib.set_uniform("lightPos", 10.0, 10.0, 10.0)
            ShaderLib.set_uniform("viewerPos", camera.eye.x, camera.eye.y, camera.eye.z)
            Primitives.draw("sphere")

        self.last_drawn = drawn
        self.setTitle(f"FrustumCull - drawn {drawn}/{len(self.grid_positions)}")

        # Draw the test camera's frustum as a wireframe box, always in the
        # test camera's own colour, viewed from whichever camera is active.
        ShaderLib.use(DefaultShader.COLOUR)
        mvp = camera.project @ camera.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        self._draw_frustum(self.test_camera)

        # Mark the test camera's eye position with a small cube when viewing
        # from the observer camera, so its location is visible top-down.
        if self.active_camera_index == 1:
            ShaderLib.use("Phong")
            eye = self.test_camera.eye
            mv = (
                camera.view
                @ self.mouse_global_tx
                @ Mat4().translate(eye.x, eye.y, eye.z)
                @ Mat4().scale(0.5, 0.5, 0.5)
            )
            mvp = camera.project @ mv
            normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            ShaderLib.set_uniform("lightPos", 10.0, 10.0, 10.0)
            ShaderLib.set_uniform("viewerPos", camera.eye.x, camera.eye.y, camera.eye.z)
            Primitives.draw("cube")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        aspect = float(w) / h
        self.test_camera.set_shape(
            self.test_camera.fov, aspect, self.test_camera.near, self.test_camera.far
        )
        self.observer_camera.set_shape(
            self.observer_camera.fov,
            aspect,
            self.observer_camera.near,
            self.observer_camera.far,
        )

    def _move_test_camera(self, dx: float, dy: float, dz: float) -> None:
        """Move the test camera using whichever mode (keys e, l, b, /) is active."""
        camera = self.test_camera
        if self.move_mode == MOVE_EYE:
            camera.move_eye(dx, dy, dz)
        elif self.move_mode == MOVE_LOOK:
            camera.move_look(dx, dy, dz)
        elif self.move_mode == MOVE_BOTH:
            camera.move_both(dx, dy, dz)
        elif self.move_mode == MOVE_SLIDE:
            camera.slide(dx, dy, dz)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_W:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE
            )  # Switch to wireframe rendering
        elif key == Qt.Key_S:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_FILL
            )  # Switch to solid fill rendering
        elif key == Qt.Key_1:
            self.active_camera_index = 0
        elif key == Qt.Key_2:
            self.active_camera_index = 1
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        # Switch which of the test camera's move_* methods Left/Right/Up/
        # Down/I/O drive, matching NGL9Demos/FrustumCull's e/l/b/s toggle.
        elif key == Qt.Key_E:
            self.move_mode = MOVE_EYE
        elif key == Qt.Key_L:
            self.move_mode = MOVE_LOOK
        elif key == Qt.Key_B:
            self.move_mode = MOVE_BOTH
        elif key == Qt.Key_Slash:
            self.move_mode = MOVE_SLIDE
        elif key == Qt.Key_Left:
            self._move_test_camera(KEY_INCREMENT, 0, 0)
        elif key == Qt.Key_Right:
            self._move_test_camera(-KEY_INCREMENT, 0, 0)
        elif key == Qt.Key_Up:
            self._move_test_camera(0, KEY_INCREMENT, 0)
        elif key == Qt.Key_Down:
            self._move_test_camera(0, -KEY_INCREMENT, 0)
        elif key == Qt.Key_O:
            self._move_test_camera(0, 0, KEY_INCREMENT)
        elif key == Qt.Key_I:
            self._move_test_camera(0, 0, -KEY_INCREMENT)
        elif key == Qt.Key_R:
            self.test_camera.roll(3.0)
        elif key == Qt.Key_P:
            self.test_camera.pitch(3.0)
        elif key == Qt.Key_Y:
            self.test_camera.yaw(3.0)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            c = self.test_camera
            c.set_shape(c.fov + 1.0, c.aspect, c.near, c.far)
        elif key == Qt.Key_Minus:
            c = self.test_camera
            c.set_shape(max(1.0, c.fov - 1.0), c.aspect, c.near, c.far)
        # Trigger a redraw to apply changes
        self.update()
        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement events for camera control.

        Args:
            event: The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        # Translate (pan) the scene if the right mouse button is pressed
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
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        position = event.position()
        # Left button initiates rotation
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        # Right button initiates translation
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handles mouse button release events to stop rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        # Stop rotating when the left button is released
        if event.button() == Qt.LeftButton:
            self.rotate = False
        # Stop translating when the right button is released
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming.

        Args:
            event: The QWheelEvent object.
        """
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
    """
    A custom QApplication subclass for improved debugging.

    By default, Qt's event loop can suppress exceptions that occur within event handlers
    (like paintGL or mouseMoveEvent), making it very difficult to debug as the application
    may simply crash or freeze without any error message. This class overrides the `notify`
    method to catch these exceptions, print a full traceback to the console, and then
    re-raise the exception to halt the program, making the error immediately visible.
    """

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            # Attempt to process the event as usual
            return super().notify(receiver, event)
        except Exception:
            # If an exception occurs, print the full traceback
            traceback.print_exc()
            # Re-raise the exception to stop the application
            raise


if __name__ == "__main__":
    # --- Application Entry Point ---

    # Create a QSurfaceFormat object to request a specific OpenGL context
    format: QSurfaceFormat = QSurfaceFormat()
    # Request 4x multisampling for anti-aliasing
    format.setSamples(4)
    # Request OpenGL version 4.1 as this is the highest supported on macOS
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    # Request a Core Profile context, which removes deprecated, fixed-function pipeline features
    format.setProfile(QSurfaceFormat.CoreProfile)
    # Request a 24-bit depth buffer for proper 3D sorting
    format.setDepthBufferSize(24)
    # Set default format for all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    # Apply this format to all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv

    # Check for a "--debug" command-line argument to run the DebugApplication
    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    # Start the application's event loop
    sys.exit(app.exec())
