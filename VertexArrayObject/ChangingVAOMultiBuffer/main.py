#!/usr/bin/env -S uv run --script
"""
ChangingVAO Example

This script demonstrates how to use OpenGL and Qt to render a dynamic set of 3D lines.
The vertex data changes over time, showing how to update a Vertex Array Object (VAO) each frame.
User input allows for interactive rotation, translation, and zoom.
"""

import logging
import sys

import OpenGL.GL as gl
from ngl import (
    Mat4,
    PySideEventHandlingMixin,
    Random,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    Vec3,
    Vec4Array,
    VertexData,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

COLOUR_SHADER = "ColourShader"
DATA_SIZE = 12345


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """
    Main application window for rendering dynamic 3D lines with OpenGL.

    Handles OpenGL initialization, rendering, and user input for interactive control.
    The vertex data is updated periodically to demonstrate dynamic VAO usage.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initialize the window and set up default parameters.
        """
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Changing VAO")
        self.view: Mat4 = Mat4()  # View matrix
        self.project: Mat4 = Mat4()  # Projection matrix
        self.data: list[float] = []  # Dynamic vertex data

    def initializeGL(self) -> None:
        """
        Set up OpenGL context, load shaders, and initialize scene.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Set background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
        gl.glEnable(gl.GL_MULTISAMPLE)  # Enable anti-aliasing
        # Use a simple colour shader
        if not ShaderLib.load_shader(
            COLOUR_SHADER,
            vert="shaders/ColourVertex.glsl",
            frag="shaders/ColourFragment.glsl",
        ):
            logging.error("Error loading shaders")
            self.close()
        ShaderLib.use(COLOUR_SHADER)
        ShaderLib.print_registered_uniforms()
        ShaderLib.print_properties()

        # Set up camera/view matrix
        self.view = look_at(Vec3(0, 1, 40), Vec3(0, 0, 0), Vec3(0, 1, 0))

        # Create VAO for lines and Colour
        self.vao = VAOFactory.create_vao(VAOType.MULTI_BUFFER, gl.GL_LINES)
        with self.vao:
            # As this is a Multi buffer VAO we can add two initial buffer one for Vertex and one for Color
            data = VertexData(data=None, size=0)  # empty data
            self.vao.set_data(data, 0)  # index 0 for Vertex buffer
            # colours will be the same each time so set once.
            colours = Vec4Array()
            for i in range(DATA_SIZE):
                colours.append(Random.get_random_colour4())

            colour_data: VertexData = VertexData(
                data=colours.to_list(), size=len(colours)
            )

            self.vao.set_data(colour_data, 1)  # index 1 for Color buffer
            self.vao.set_vertex_attribute_pointer(1, 4, gl.GL_FLOAT, 0, 0)
        # # Set up text rendering for displaying data size
        self.text = Text("../fonts/Arial.ttf", 18)
        self.text.set_screen_size(self.window_width, self.window_height)

        # Start a timer to update the vertex data periodically
        self.startTimer(220.0)

    def loadMatricesToShader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self) -> None:
        """
        Render the scene. Called automatically by Qt.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(COLOUR_SHADER)

        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x

        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        self.loadMatricesToShader()

        # Bind VAO and update vertex data
        with self.vao:
            data = VertexData(data=self.data, size=len(self.data) // 3)
            self.vao.set_data(data, 0)

            # Set vertex attribute pointer for position (3 floats per vertex)
            # We must do this each time as we change the data.
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
            self.vao.draw()

        # Render text showing the current data size
        self.text.set_colour(1, 1, 1)
        self.text.render_text(10, 18, f"Data Size {(len(self.data) / 2)}")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle window resizing and update the projection matrix.

        Args:
            w: New window width.
            h: New window height.
        """
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        self.text.set_screen_size(self.window_width, self.window_height)

    def keyPressEvent(self, event) -> None:
        """
        Handle keyboard input for controlling the scene.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)  # Wireframe mode
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)  # Solid mode
        elif key == Qt.Key_Space:
            # Reset rotation and position
            self.spinXFace = 0
            self.spinYFace = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Periodically called by Qt to update the vertex data with random values.

        This demonstrates how to update a VAO with new data each frame.
        """
        size = 100 + int(Random.random_positive_number(12000))
        # Clear old data
        del self.data[:]
        for i in range(0, size * 2):
            p = Random.get_random_vec3() * 5
            self.data.append(p.x)
            self.data.append(p.y)
            self.data.append(p.z)
        self.update()


if __name__ == "__main__":
    # Set up Qt application and OpenGL format
    app = QApplication(sys.argv)
    format = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    window = MainWindow()
    window.setFormat(format)
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
