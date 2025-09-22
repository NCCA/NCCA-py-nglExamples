#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import logging
import math
import random
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    FirstPersonCamera,
    Mat2,
    Mat3,
    Mat4,
    Primitives,
    Random,
    ShaderLib,
    Transform,
    Vec3,
    Vec3Array,
    logger,
    look_at,
)
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from texture_pack import TexturePack

PBR_SHADER = "pbr"


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

        # --- Window and UI Attributes ---
        self.width: int = 1024  # Window width¦
        self.height: int = 720  # Window height
        self.setTitle("Blank PySide6 py-ngl")
        self.transform = Transform()
        self.mouse_global_tx: Mat4 = (
            Mat4()
        )  # Global transformation matrix controlled by the mouse

        # --- Window and UI Attributes ---
        self.width: int = 1024  # Window width¦
        self.height: int = 720  # Window height
        self.setTitle("PBR Texture")
        self.model_position: Vec3 = Vec3()  # Position of the model in world space

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
        self.keys_pressed: set = set()
        eye = Vec3(0, 5, 13)
        look = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.camera = FirstPersonCamera(eye, look, up, 45.0)
        self.camera.set_projection(45.0, 720.0 / 576.0, 0.05, 350.0)
        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 0.0
        self.seed = 12345
        self.light_on = [True, True, True, True]

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
        ShaderLib.set_uniform("albedoMap", 0)
        ShaderLib.set_uniform("normalMap", 1)
        ShaderLib.set_uniform("metallicMap", 2)
        ShaderLib.set_uniform("roughnessMap", 3)
        ShaderLib.set_uniform("aoMap", 4)
        ShaderLib.print_registered_uniforms()

        light_colors = Vec3Array(
            [
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
            ]
        )

        self.light_positions = Vec3Array(
            [
                Vec3(-5.0, 4.0, -5.0),
                Vec3(5.0, 4.0, -5.0),
                Vec3(-5.0, 4.0, 5.0),
                Vec3(5.0, 4.0, 5.0),
            ]
        )
        for i in range(4):
            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_positions[i])
            ShaderLib.set_uniform(f"lightColors[{i}]", light_colors[i])
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        Primitives.create_sphere("sphere", 0.5, 40)
        Primitives.create_triangle_plane("floor", 30, 30, 10, 10, Vec3(0, 1, 0))
        TexturePack.load_json("textures/textures.json")
        Primitives.load_default_primitives()

    def load_matrices_to_shader(self):
        M = self.transform.get_matrix()
        MV = self.camera.view @ M
        MVP = self.camera.get_vp() @ M

        normalMatrix = Mat3.from_mat4(MV)
        normalMatrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normalMatrix)
        ShaderLib.set_uniform("M", M)
        texture_rotation = math.radians((Random.random_number(180.0)))
        cosTheta = math.cos(texture_rotation)
        sinTheta = math.sin(texture_rotation)
        texRot = Mat2.from_list([cosTheta, sinTheta, -sinTheta, cosTheta])
        ShaderLib.set_uniform("textureRotation", texRot)
        ShaderLib.set_uniform("camPos", self.camera.eye)

    def load_matrices_to_colour_shader(self):
        M = self.mouse_global_tx @ self.transform.get_matrix()
        MV = self.camera.view @ M
        MVP = self.camera.projection @ MV
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", MVP)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.width, self.height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # first we reset the movement values
        xDirection = 0.0
        yDirection = 0.0
        # now we loop for each of the pressed keys in the the set
        # and see which ones have been pressed. If they have been pressed
        # we set the movement value to be an incremental value
        for key in self.keys_pressed:
            if key == Qt.Key_Left:
                yDirection = -1.0
            elif key == Qt.Key_Right:
                yDirection = 1.0
            elif key == Qt.Key_Up:
                xDirection = 1.0
            elif key == Qt.Key_Down:
                xDirection = -1.0
            else:
                pass
        current_frame = self.timer.elapsed() * 0.001
        delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        if len(self.keys_pressed) != 0:
            self.camera.move(xDirection, yDirection, delta_time)

        ShaderLib.use(DefaultShader.COLOUR)
        # render lights as spheres
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
        # render rows*column number of spheres with varying metallic/roughness values scaled by rows and columns respectively

        Random.set_seed_value(self.seed)

        textures = ["copper", "greasy", "panel", "rusty", "wood"]
        for row in np.arange(-10, 10, 1.6):
            for col in np.arange(-10, 10, 1.6):
                TexturePack.activate_texture_pack(random.choice(textures))
                self.transform.set_position(col, 0.0, row)
                self.transform.set_rotation(
                    0.0, Random.random_positive_number() * 360.0, 0.0
                )
                self.load_matrices_to_shader()
                Primitives.draw("teapot")

        TexturePack.activate_texture_pack("greasy")

        self.transform.reset()
        self.transform.set_position(0.0, -0.5, 0.0)
        self.load_matrices_to_shader()
        Primitives.draw("floor")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        # Update the projection matrix to match the new aspect ratio.
        # This creates a perspective projection with a 45-degree field of view.
        self.camera.set_projection(45.0, w / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """

        def _set_light(num, mode):
            ShaderLib.use(PBR_SHADER)
            if mode:
                colour = Vec3(255.0, 255.0, 255.0)
            else:
                colour = Vec3(0.0, 0.0, 0.0)
            ShaderLib.set_uniform(num, colour)

        key = event.key()
        self.keys_pressed.add(key)
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_R:
            self.seed = random.randint(0, 1000000)
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        elif key == Qt.Key_1:
            self.light_on[0] ^= True
            _set_light("lightColors[0]", self.light_on[0])
        elif key == Qt.Key_2:
            self.light_on[1] ^= True
            _set_light("lightColors[1]", self.light_on[1])
        elif key == Qt.Key_3:
            self.light_on[2] ^= True
            _set_light("lightColors[2]", self.light_on[2])
        elif key == Qt.Key_4:
            self.light_on[3] ^= True
            _set_light("lightColors[3]", self.light_on[3])
        # Trigger a redraw to apply changes
        self.update()
        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """
        Handles keyboard release events.

        Args:
            event: The QKeyEvent object containing information about the key release.
        """
        key = event.key()
        self.keys_pressed.remove(key)
        # Trigger a redraw to apply changes
        self.update()
        # Call the base class implementation for any unhandled events
        super().keyReleaseEvent(event)

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
            self.camera.process_mouse_movement(diff_x, diff_y)

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
            self.camera.process_mouse_scroll(1.0)
        elif num_pixels.x() < 0:
            self.camera.process_mouse_scroll(-1.0)
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
