#!/usr/bin/env -S uv run --script
"""
ChangingVAO Example

This script demonstrates how to use OpenGL and Qt to render a dynamic set of 3D lines.
The vertex data changes over time, showing how to update a Vertex Array Object (VAO) each frame.
User input allows for interactive rotation, translation, and zoom.
"""

import sys
import traceback

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
    logger,
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
        self.window_width: int = 1024
        self.window_height: int = 720
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
        Text.add_font("Arial", "../fonts/Arial.ttf", 48)
        print(f"{self.window_width=} {self.window_height=}")
        Text.set_screen_size(self.window_width, self.window_height)

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
        # self.text.render_text(10, 18, f"Data Size {(len(self.data) / 2)}")
        Text.render_dynamic_text(
            "Arial", 10, 48, f"Data Size {(len(self.data) / 2)}", Vec3(1, 1, 1)
        )

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
        Text.set_screen_size(self.window_width, self.window_height)

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
    # Set up Qt application and OpenGL format
    app = DebugApplication(sys.argv)

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
