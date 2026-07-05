#!/usr/bin/env -S uv run --script
"""
AnimatedTextures: GPU billboards built in a geometry shader, sampling animated
fire sprite-sheet textures. Demonstrates geometry-shader billboard expansion
and multi-texture-unit binding with py-ngl.
"""

import math
import random
import sys
import traceback
from dataclasses import dataclass

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    Prims,
    ShaderLib,
    Texture,
    Transform,
    VAOFactory,
    VAOType,
    Vec3,
    Vec4,
    VertexData,
    logger,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt, QTimer
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
        self.setTitle("AnimatedTextures")

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
        self.makeCurrent()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.view = look_at(Vec3(0, 4, 20), Vec3(0, 2, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Billboard",
            "shaders/BillboardVert.glsl",
            "shaders/BillboardFrag.glsl",
            geo="shaders/BillboardGeo.glsl",
        )

        # Load the 3 sprite-sheet textures, binding each to its own texture unit.
        self.textures = []
        self.texture_ids = []
        for i, fname in enumerate(["map1.png", "map2.png", "map3.png"]):
            tex = Texture(f"textures/{fname}")
            tex.set_multi_texture(i)
            tex_id = tex.set_texture_gl()
            self.textures.append(tex)
            self.texture_ids.append(tex_id)

        # Build 500 randomly-placed billboard points around a circle.
        data = []
        for i in range(500):
            radius = random.uniform(8.0, 9.0)
            angle = math.radians(i)
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = random.uniform(0.0, 8.0)
            tex_index = float(random.randint(0, 2))
            offset = random.uniform(0.0, 10.0)
            data.extend([x, y, z, tex_index, offset])

        arr = np.array(data, dtype=np.float32)
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_POINTS)
        with self.vao:
            vertex_data = VertexData(data=arr, size=500)
            self.vao.set_data(vertex_data)
            self.vao.set_vertex_attribute_pointer(0, 4, gl.GL_FLOAT, 5 * 4, 0)
            self.vao.set_vertex_attribute_pointer(1, 1, gl.GL_FLOAT, 5 * 4, 4 * 4)

        self.time: float = 0.0
        self.animate: bool = True

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x: Mat4 = Mat4().rotate_x(self.spin_x_face)
        rot_y: Mat4 = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Billboard")
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("cameraPos", 0.0, 4.0, 20.0)
        ShaderLib.set_uniform("time", self.time)

        for i, tex_id in enumerate(self.texture_ids):
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        ShaderLib.set_uniform("tex1", 0)
        ShaderLib.set_uniform("tex2", 1)
        ShaderLib.set_uniform("tex3", 2)

        with self.vao:
            self.vao.draw()

        if self.animate:
            self.time += 0.1
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.1, 100.0)

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
            self.animate = not self.animate
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
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
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
        """
        Handles mouse button release events to stop rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming.

        Args:
            event: The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
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
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # --- Application Entry Point ---

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv

    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
