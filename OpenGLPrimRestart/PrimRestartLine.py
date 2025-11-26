#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import math
import random
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import (
    IndexVertexData,
    Mat4,
    PySideEventHandlingMixin,
    Random,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
    lerp,
    logger,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


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
        self.setTitle("Primitive Restart Line")
        self.animate = True
        self.blade_update = 0.0

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
        self.view = look_at(Vec3(0, 2, 12), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.load_shader(
            "LineShader", "shaders/LineVertex.glsl", "shaders/LineFragment.glsl"
        )
        ShaderLib.use("LineShader")
        # This will be a list of Vec3Array objects
        self.blades, self.num_to_render = self.create_blades(20.0, 20.0, 120, 120)
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE_INDEX, gl.GL_LINE_STRIP)
        self.startTimer(16)

    def create_lines(self, pos: Vec3) -> Vec3Array:
        points = Vec3Array()
        # store the inital point (root)
        points.append(pos)
        num_lines = random.randint(2, 12)
        height = 1.0 + Random.random_positive_number(1.5)
        step = height / num_lines
        p = pos.copy()
        for _ in range(num_lines):
            p.y += step
            p.x += Random.random_number(0.1)
            p.z += Random.random_number(0.1)
            points.append(
                p.copy()
            )  # note copy is very important else we get the same Vec3
        return points, num_lines

    def create_blades(self, row_size, col_size, rows, cols):
        blades = []
        total_points = 0
        # pre calculate the steps for x,y
        z_positions = np.linspace(-col_size * 0.5, col_size * 0.5, cols, endpoint=False)
        x_positions = np.linspace(-row_size * 0.5, row_size * 0.5, rows, endpoint=False)
        for z in z_positions:
            for x in x_positions:
                blade, num_points = self.create_lines(Vec3(x, 0, z))
                blades.append(blade)
                total_points += num_points
        return blades, total_points

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
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        ShaderLib.set_uniform("MVP", self.project @ self.view @ self.mouse_global_tx)
        self.render_blades()

    def render_blades(self):
        # Use the largest number for a prim restart index.
        restart_index = (
            np.iinfo(np.uint32).max - 1
        )  # 4294967294 (Python int or numpy scalar)
        base_colour = Vec3(0.1, 0.2, 0.1)
        tip_colour = Vec3(0.0, 1.0, 0.0)
        indices = []
        idx = 0
        # going to store x,y,z and r,g,b for each vertex point.
        points = []
        for blade in self.blades:
            # We will lerp the colour between the base and tip colours
            t = 0.0
            t_step = 1.0 / len(blade)
            for p in blade:
                points.append(p.x)
                points.append(p.y)
                points.append(p.z)
                points.append(lerp(base_colour, tip_colour, t).x)
                points.append(lerp(base_colour, tip_colour, t).y)
                points.append(lerp(base_colour, tip_colour, t).z)
                t += t_step
                indices.append(idx)
                idx += 1
            indices.append(restart_index)
        with self.vao:
            gl.glPrimitiveRestartIndex(restart_index)
            gl.glEnable(gl.GL_PRIMITIVE_RESTART)

            data = IndexVertexData(
                data=points,
                size=len(indices),
                indices=indices,
                index_type=gl.GL_UNSIGNED_INT,
            )
            self.vao.set_data(data)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 6 * 4, 0)
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 6 * 4, 3 * 4)
            self.vao.set_num_indices(self.num_to_render)
            self.vao.draw()

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        self.update()

    def timerEvent(self, event):
        if self.animate:
            print("Timer event triggered")
            for blade in self.blades:
                for point in blade:
                    point.x += math.sin(self.blade_update * point.y) * 0.01
                    point.z += math.cos(self.blade_update * point.y) * 0.01
        self.blade_update += 0.05
        print("Blade update:", self.blade_update)
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
