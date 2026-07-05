#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import ctypes
import math
import random
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat4, Random, Vec3, Vec3Array, lerp, logger, look_at, perspective
from ncca.ngl.opengl import (
    IndexVertexData,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
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
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 2, 12), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.load_shader(
            "LineShader", "shaders/LineVertex.glsl", "shaders/LineFragment.glsl"
        )
        ShaderLib.use("LineShader")

        # Build blades (same as before)
        self.blades, self.num_to_render = self.create_blades(20.0, 20.0, 120, 120)

        # Prepare GPU-side buffers once
        self._prepare_gpu_buffers()

        # Use a lower timer frequency if needed (e.g. ~60Hz -> 16ms)
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

    def _prepare_gpu_buffers(self):
        """
        Build interleaved vertex numpy arrays and index array once,
        create VAO/VBO/EBO and upload initial data with DYNAMIC_DRAW.
        """
        # Build lists into numpy arrays (vectorized approach)
        base_colour = np.array([0.1, 0.2, 0.1], dtype=np.float32)
        tip_colour = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        vertex_list = []  # temporary list of rows to be stacked to numpy
        index_list = []
        idx = 0

        # Count total vertices and build arrays
        for blade in self.blades:
            n = len(blade)
            # Create t values for colours along blade [0..1]
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
            # For each point in blade append [x,y,z,r,g,b]
            for i, p in enumerate(blade):
                vertex_list.append(
                    (
                        p.x,
                        p.y,
                        p.z,
                        float(base_colour[0] + (tip_colour[0] - base_colour[0]) * t[i]),
                        float(base_colour[1] + (tip_colour[1] - base_colour[1]) * t[i]),
                        float(base_colour[2] + (tip_colour[2] - base_colour[2]) * t[i]),
                    )
                )
                index_list.append(idx)
                idx += 1
            # primitive restart marker
            # Use 0xFFFFFFFF (max uint32) as restart index; GL_UNSIGNED_INT
            index_list.append(np.iinfo(np.uint32).max)
        # Convert to contiguous numpy arrays
        self.vertex_data = np.asarray(vertex_list, dtype=np.float32)  # shape (N,6)
        self.num_vertices = self.vertex_data.shape[0]
        self.index_data = np.asarray(index_list, dtype=np.uint32)
        self.num_indices = self.index_data.size

        # Save base X/Z so animation can compute offsets relative to original positions
        self.base_x = self.vertex_data[:, 0].copy()
        self.base_z = self.vertex_data[:, 2].copy()
        # the Y column (vertex_data[:,1]) remains fixed and used for phase
        self.vertex_stride = 6 * 4  # bytes

        # Create VAO, VBO, EBO
        self._vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao_id)

        # VBO (positions + colours)
        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.vertex_data.nbytes,
            self.vertex_data.ravel(),
            gl.GL_DYNAMIC_DRAW,
        )

        # position attribute 0 (vec3)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_stride, ctypes.c_void_p(0)
        )

        # color attribute 1 (vec3) offset 3 floats
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_stride, ctypes.c_void_p(3 * 4)
        )

        # EBO (indices)
        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self.index_data.nbytes,
            self.index_data,
            gl.GL_STATIC_DRAW,
        )

        # Enable primitive restart once
        gl.glPrimitiveRestartIndex(np.iinfo(np.uint32).max)
        gl.glEnable(gl.GL_PRIMITIVE_RESTART)

        # Unbind VAO
        gl.glBindVertexArray(0)

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
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z
        ShaderLib.set_uniform("MVP", self.project @ self.view @ self.mouse_global_tx)
        self.render_blades()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Update stored window size and the projection matrix when the window is resized.
        """
        # consider device pixel ratio for HiDPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        # avoid division by zero
        if h == 0:
            h = 1
        self.project = perspective(45.0, float(w) / float(h), 0.01, 350.0)

    def render_blades(self):
        """
        Draw using the prepared VAO/EBO. We don't rebuild data here.
        """
        gl.glBindVertexArray(self._vao_id)
        # shader MVP is already set in paintGL
        gl.glDrawElements(
            gl.GL_LINE_STRIP,
            int(self.num_indices),
            gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
        )
        gl.glBindVertexArray(0)

    def timerEvent(self, event):
        if self.animate:
            # Vectorized update of x,z offsets using numpy (no Python loops)
            # amplitude for movement
            amp = 0.01
            y = self.vertex_data[:, 1]  # Y column (phase)
            phase = self.blade_update * y
            new_x = self.base_x + np.sin(phase) * amp
            new_z = self.base_z + np.cos(phase) * amp

            # Update the interleaved vertex_data in-place for columns 0 and 2
            self.vertex_data[:, 0] = new_x
            self.vertex_data[:, 2] = new_z

            # Upload only the position components to the GPU VBO.
            # Because it's interleaved, we must either reupload full VBO or update ranges.
            # We'll update the full VBO buffer which is still faster than recreating it.
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
            # Use contiguous bytes from vertex_data
            gl.glBufferSubData(
                gl.GL_ARRAY_BUFFER, 0, self.vertex_data.nbytes, self.vertex_data.ravel()
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # advance animation state
        self.blade_update += 0.05
        # Request redraw
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
