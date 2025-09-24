#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import sys
import traceback
from dataclasses import dataclass

import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    ShaderLib,
    Transform,
    Vec3,
    Vec4,
    logger,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


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
        self.setTitle("VAOPrimitives")

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

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        # Set the background color to a dark grey
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        # Enable depth testing, which ensures that objects closer to the camera obscure those further away
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Enable multisampling for anti-aliasing, which smooths jagged edges
        gl.glEnable(gl.GL_MULTISAMPLE)
        # Set up the camera's view matrix.
        # It looks from (0, 1, 4) towards (0, 0, 0) with the 'up' direction along the Y-axis.
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        # Add a ground plane
        Primitives.create_triangle_plane("ground", 10, 10, 20, 20, Vec3(0, 1, 0))
        Primitives.create_sphere("sphere", 0.3, 32)
        Primitives.create_cone("cone", 0.5, 1.0, 20, 20)
        Primitives.create_capsule("capsule", 0.2, 0.4, 20)
        Primitives.create_cylinder("cylinder", 0.2, 0.4, 20, 20)
        Primitives.create_torus("torus", 0.1, 0.3, 20, 20)
        Primitives.create_disk("disk", 0.5, 20)

    def load_matrices_to_shader(self, transform) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        ShaderLib.use(DefaultShader.DIFFUSE)
        MV = self.view @ self.mouse_global_tx @ transform.get_matrix()
        mvp = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Apply rotation based on user input
        rot_x: Mat4 = Mat4().rotate_x(self.spin_x_face)
        rot_y: Mat4 = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Apply translation
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        # draw ground
        tx = Transform()
        tx.set_position(0, -0.5, 0)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        self.load_matrices_to_shader(tx)

        def render_mesh(tx, prim, pos, scale, rot, colour):
            tx.reset()
            tx.set_position(pos)
            tx.set_scale(scale)
            tx.set_rotation(rot)
            ShaderLib.set_uniform("Colour", colour[0], colour[1], colour[2], colour[3])
            self.load_matrices_to_shader(tx)
            Primitives.draw(prim)

        @dataclass
        class Primitive:
            name: str
            position: Vec3
            scale: Vec3
            rotation: Vec3
            color: Vec4

        # fmt: off
        prims = [
            Primitive("ground",Vec3(0, -0.5, 0),Vec3(10, 1, 10),Vec3(0, 0, 0),Vec4(1.0, 1.0, 1.0, 1.0)),
            Primitive("teapot",Vec3(0, 0, 0),Vec3(1, 1, 1),Vec3(0, 0, 0),Vec4(1.0, 0.0, 0.0, 1.0)),
            Primitive("cube",Vec3(-1, -0.2, 0),Vec3(0.5, 0.5, 0.5),Vec3(0, 45, 0),Vec4(0.0, 1.0, 0.0, 1.0)),
            Primitive("sphere",Vec3(1, -0.2, 0),Vec3(0.5, 0.5, 0.5),Vec3(0, 0, 0),Vec4(0.0, 0.0, 1.0, 1.0)),
            Primitive("bunny",Vec3(2, -0.5, 0),Vec3(0.1, 0.1, 0.1),Vec3(0, -90, 0),Vec4(1.0, 0.0, 1.0, 1.0)),
            Primitive("buddah",Vec3(3, -0.5, 0),Vec3(0.1, 0.1, 0.1),Vec3(0, -90, 0),Vec4(0.0, 1.0, 1.0, 1.0)),
            Primitive("dragon",Vec3(-1, -0.2, -1),Vec3(0.1, 0.1, 0.1),Vec3(0, -90, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("troll",Vec3(1, 0.1, -1),Vec3(1, 1, 1),Vec3(0, -90, 0),Vec4(1.0, 0.0, 0.0, 1.0)),
            Primitive("tetrahedron",Vec3(-1, 0.5, -2),Vec3(0.5, 0.5, 0.5),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("octahedron",Vec3(1, 0.5, -2),Vec3(0.5, 0.5, 0.5),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("icosahedron",Vec3(-2.5, 0.5, -2),Vec3(0.5, 0.5, 0.5),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("dodecahedron",Vec3(2.5, 0.5, -2),Vec3(0.5, 0.5, 0.5),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("cone",Vec3(0, 0, -3),Vec3(1, 1, 1),Vec3(-90, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("capsule",Vec3(-1, 0, -3),Vec3(1, 1, 1),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("cylinder",Vec3(1, 0, -3),Vec3(1, 1, 1),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("torus",Vec3(2, 0, -3),Vec3(1, 1, 1),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
            Primitive("disk",Vec3(-2, 0, -3),Vec3(1, 1, 1),Vec3(0, 0, 0),Vec4(1.0, 1.0, 0.0, 1.0)),
        ]
        [render_mesh( tx, prim.name, prim.position, prim.scale, prim.rotation, prim.color )  for prim in prims]
        # fmt: on

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
        # Update the projection matrix to match the new aspect ratio.
        # This creates a perspective projection with a 45-degree field of view.
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

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
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spinXFace = 0
            self.spinYFace = 0
            self.model_position.set(0, 0, 0)
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
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
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
    # Start the application's event loop
    sys.exit(app.exec())
