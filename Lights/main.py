#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import List

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Random, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib, ShaderType
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QWidget

PBR_SHADER = "pbr"
VERTEX_SHADER_NAME = f"{PBR_SHADER}VertexShader"
FRAGMENT_SHADER_NAME = f"{PBR_SHADER}FragmentShader"


@dataclass
class Light:
    """Data class to represent a light source."""

    position: Vec3
    colour: Vec3


class MainWindow(QOpenGLWindow):
    """
    The main window for the OpenGL application.

    Inherits from QOpenGLWindow to provide a canvas for OpenGL rendering within a PySide6 GUI.
    It handles user input (mouse, keyboard) for camera control and manages the OpenGL context.
    """

    def __init__(self, parent: QWidget = None) -> None:
        """
        Initializes the main window and sets up default scene parameters.
        """
        super().__init__(parent)
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
        self.setTitle("SimplePyNGL")

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
        self.transform = Transform()
        self.teapot_scale = 8.0
        self.teapot_rotation = 0.0
        self.rotation_timer = self.startTimer(20)
        self.light_change_timer = self.startTimer(1000)
        self.num_lights = 8
        self.show_lights = True
        self.light_array: List[Light] = []

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

        # We now create our view matrix for a static camera
        self.eye = Vec3(0.0, 10.0, 20.0)
        to = Vec3(0.0, 0.0, 0.0)
        up = Vec3(0.0, 1.0, 0.0)
        # now load to our new camera
        self.view = look_at(self.eye, to, up)
        self._load_and_edit_shader()
        Primitives.load_default_primitives()
        self.create_lights()
        ShaderLib.print_registered_uniforms(PBR_SHADER)

    def _load_and_edit_shader(self) -> None:
        """
        Loads and configures the PBR shader program.

        This method creates the shader program, attaches vertex and fragment shaders,
        and sets the number of lights by modifying the shader source.
        """
        ShaderLib.create_shader_program(PBR_SHADER)

        ShaderLib.attach_shader(VERTEX_SHADER_NAME, ShaderType.VERTEX)
        ShaderLib.attach_shader(FRAGMENT_SHADER_NAME, ShaderType.FRAGMENT)
        ShaderLib.load_shader_source(VERTEX_SHADER_NAME, "shaders/PBRVertex.glsl")
        ShaderLib.load_shader_source(FRAGMENT_SHADER_NAME, "shaders/PBRFragment.glsl")
        # the shader has a tag called @numLights, edit this and set to 8
        ShaderLib.edit_shader(FRAGMENT_SHADER_NAME, "@numLights", f"{self.num_lights}")
        ShaderLib.compile_shader(VERTEX_SHADER_NAME)
        ShaderLib.compile_shader(FRAGMENT_SHADER_NAME)
        ShaderLib.attach_shader_to_program(PBR_SHADER, VERTEX_SHADER_NAME)
        ShaderLib.attach_shader_to_program(PBR_SHADER, FRAGMENT_SHADER_NAME)

        ShaderLib.link_program_object(PBR_SHADER)
        ShaderLib.use(PBR_SHADER)
        self._load_shader_defaults()

    def create_lights(self) -> None:
        """
        Creates and configures the lights for the scene.

        This method generates a specified number of lights with random positions and colors,
        and uploads their data to the shader uniforms.
        """
        ShaderLib.use(PBR_SHADER)
        self.light_array = []
        for i in range(self.num_lights):
            self.light_array.append(
                Light(
                    position=Random.get_random_point(20, 20, 20),
                    colour=Vec3(0.1, 0.1, 0.1) + Random.get_random_colour3() * 100,
                )
            )

            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_array[i].position)
            ShaderLib.set_uniform(f"lightColours[{i}]", self.light_array[i].colour)

    def _load_shader_defaults(self) -> None:
        """Loads default uniform values to the PBR shader."""
        ShaderLib.use(PBR_SHADER)
        ShaderLib.set_uniform("camPos", 0.0, 10.0, 20.0)
        ShaderLib.set_uniform("albedo", 0.950, 0.71, 0.29)
        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)

    def load_matrices_to_shader(self) -> None:
        """Loads model, view, and projection matrices to the PBR shader."""
        ShaderLib.use(PBR_SHADER)

        M = self.mouse_global_tx @ self.transform.matrix()
        MV = self.view @ M
        MVP = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("M", M)

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
        self.load_matrices_to_shader()
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        if self.show_lights:
            self.transform.reset()
            for light in self.light_array:
                self.transform.set_position(
                    light.position.x, light.position.y, light.position.z
                )
                self.load_matrices_to_colour_shader(light.colour)
                Primitives.draw("cube")

        self.transform.reset()
        self.transform.set_scale(
            self.teapot_scale, self.teapot_scale, self.teapot_scale
        )
        self.transform.set_rotation(
            self.teapot_rotation, self.teapot_rotation, self.teapot_rotation
        )
        self.load_matrices_to_shader()
        Primitives.draw("teapot")

    def load_matrices_to_colour_shader(self, colour: Vec3) -> None:
        """
        Loads matrices and color to the default color shader for drawing light indicators.

        Args:
            colour: The color of the light.
        """
        ShaderLib.use(DefaultShader.COLOUR)
        MVP = self.project @ self.view @ self.mouse_global_tx @ self.transform.matrix()
        ShaderLib.set_uniform("MVP", MVP)
        c = colour / 200.0
        ShaderLib.set_uniform("Colour", c.x, c.y, c.z, 1.0)

    def update_lights(self) -> None:
        """Updates the number of lights in the scene."""
        logger.info(f"Number of Lights {self.num_lights}")
        self.num_lights = int(np.clip(self.num_lights, 1, 120))
        logger.info(f"Number of Lights {self.num_lights}")
        ShaderLib.reset_edits(FRAGMENT_SHADER_NAME)

        ShaderLib.edit_shader(FRAGMENT_SHADER_NAME, "@numLights", f"{self.num_lights}")
        ShaderLib.compile_shader(VERTEX_SHADER_NAME)
        ShaderLib.compile_shader(FRAGMENT_SHADER_NAME)
        ShaderLib.link_program_object(PBR_SHADER)
        ShaderLib.use(PBR_SHADER)
        self.create_lights()
        self.setTitle(f"Number of Lights {self.num_lights}")
        self._load_shader_defaults()

    def timerEvent(self, event: QTimerEvent) -> None:
        """
        Handles timer events for animations.

        Args:
            event: The QTimerEvent object.
        """
        if event.timerId() == self.rotation_timer:
            self.teapot_rotation += 1
            self.update()
        elif event.timerId() == self.light_change_timer:
            self.create_lights()
            # re-draw GL
            self.update()

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

    def keyPressEvent(self, event: QKeyEvent) -> None:
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
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)

        elif key == Qt.Key_1:
            self.num_lights -= 1
            self.update_lights()
        elif key == Qt.Key_2:
            self.num_lights += 1
            self.update_lights()
        elif key == Qt.Key_L:
            self.show_lights = not self.show_lights

        # Trigger a redraw to apply changes
        self.update()
        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
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

    def wheelEvent(self, event: QWheelEvent) -> None:
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

    def __init__(self, argv: List[str]) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
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

    # --- Application Entry Point ---

    # Create a QSurfaceFormat object to request a specific OpenGL context
    format = QSurfaceFormat()
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    # Start the application's event loop
    sys.exit(app.exec())
