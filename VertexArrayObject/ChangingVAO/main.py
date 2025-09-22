#!/usr/bin/env -S uv run --script
"""
ChangingVAO Example

This script demonstrates how to use OpenGL and Qt to render a dynamic set of 3D lines.
The vertex data changes over time, showing how to update a Vertex Array Object (VAO) each frame.
User input allows for interactive rotation, translation, and zoom.
"""

import sys

import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    Mat4,
    PySideEventHandlingMixin,
    Random,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    Vec3,
    VertexData,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


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
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("Changing VAO")
        self.modelPos: Vec3 = Vec3()  # Model position in world space
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

        # Set up camera/view matrix
        self.view = look_at(Vec3(0, 1, 40), Vec3(0, 0, 0), Vec3(0, 1, 0))

        # Use a simple color shader
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        # Create VAO for lines
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)

        # # Set up text rendering for displaying data size
        self.text = Text("../fonts/Arial.ttf", 18)
        print(f"{self.width=} {self.height=}")
        self.text.set_screen_size(self.width, self.height)

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
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x

        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        # Bind VAO and update vertex data
        self.loadMatricesToShader()

        with self.vao:
            data = VertexData(data=self.data, size=len(self.data) // 3)
            self.vao.set_data(data)

            # Set vertex attribute pointer for position (3 floats per vertex)
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
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        self.text.set_screen_size(self.width, self.height)

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
