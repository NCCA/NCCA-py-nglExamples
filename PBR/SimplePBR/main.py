#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import logging
import sys
import traceback

import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Transform,
    Vec3,
    Vec3Array,
    logger,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

PBR_SHADER = "pbr"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
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
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )  # --- Camera and Transformation Attributes ---
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width¦
        self.window_height: int = 720  # Window height
        self.setTitle("Blank PySide6 py-ngl")
        self.transform = Transform()

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
        # Use a simple colour shader
        if not ShaderLib.load_shader(
            PBR_SHADER,
            vert="shaders/PBRVertex.glsl",
            frag="shaders/PBRFragment.glsl",
        ):
            logging.error("Error loading shaders")
            self.close()
        ShaderLib.use(PBR_SHADER)
        ShaderLib.print_registered_uniforms()
        eye = Vec3(0, 5, 13)
        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        # // now load to our new camera
        self.view = look_at(eye, to, up)
        self.project = perspective(45.0, 720.0 / 576.0, 0.05, 350.0)

        ShaderLib.set_uniform("albedo", 0.5, 0.0, 0.0)
        ShaderLib.set_uniform("ao", 1.0)
        ShaderLib.set_uniform("camPos", eye)
        ShaderLib.set_uniform("exposure", 1.0)
        light_colors = Vec3Array(
            [
                Vec3(300.0, 300.0, 300.0),
                Vec3(300.0, 300.0, 300.0),
                Vec3(300.0, 300.0, 300.0),
                Vec3(300.0, 300.0, 300.0),
            ]
        )

        self.light_positions = Vec3Array(
            [
                Vec3(-10.0, 4.0, -10.0),
                Vec3(10.0, 4.0, -10.0),
                Vec3(-10.0, 4.0, 10.0),
                Vec3(10.0, 4.0, 10.0),
            ]
        )
        for i in range(4):
            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_positions[i])
            ShaderLib.set_uniform(f"lightColors[{i}]", light_colors[i])
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        Primitives.create_sphere("sphere", 0.5, 40)
        Primitives.create_triangle_plane("floor", 20, 20, 10, 10, Vec3(0, 1, 0))

    def load_matrices_to_shader(self):
        M = self.mouse_global_tx @ self.transform.get_matrix()
        MV = self.view @ M
        MVP = self.project @ MV

        normalMatrix = Mat3.from_mat4(MV)
        normalMatrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normalMatrix)
        ShaderLib.set_uniform("M", M)

    def load_matrices_to_colour_shader(self):
        M = self.mouse_global_tx @ self.transform.get_matrix()
        MV = self.view @ M
        MVP = self.project @ MV
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", MVP)

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
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x

        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z

        # Render the floor
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 0.8, 0.8, 0.8, 1.0)
        self.transform.reset()
        self.transform.set_position(0, -1.5, 0)
        self.load_matrices_to_colour_shader()
        Primitives.draw("floor")
        # render lights as speheres
        for i in range(4):
            ShaderLib.use(DefaultShader.COLOUR)
            self.transform.reset()
            self.transform.set_position(
                self.light_positions[i][0],
                self.light_positions[i][1],
                self.light_positions[i][2],
            )
            self.load_matrices_to_colour_shader()
            Primitives.draw("sphere")

        # Render spheres with different materials
        ShaderLib.use(PBR_SHADER)
        for row in range(7):
            for col in range(7):
                metallic = row / 6.0
                roughness = max(0.05, col / 6.0)

                self.transform.reset()
                self.transform.set_position((col - 3) * 2.5, -1.0, (row - 3) * 2.5)

                ShaderLib.set_uniform("metallic", metallic)
                ShaderLib.set_uniform("roughness", roughness)

                self.load_matrices_to_shader()
                Primitives.draw("sphere")

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
    print("starting")
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
