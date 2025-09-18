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
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(QOpenGLWindow):
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
        self.mouseGlobalTX: Mat4 = Mat4()
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("Changing VAO")
        self.spinXFace: int = 0  # Rotation around X axis
        self.spinYFace: int = 0  # Rotation around Y axis
        self.rotate: bool = False
        self.translate: bool = False
        self.origX: int = 0
        self.origY: int = 0
        self.origXPos: int = 0
        self.origYPos: int = 0
        self.INCREMENT: float = 0.01  # Translation increment per pixel
        self.ZOOM: float = 2.1  # Zoom increment
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
        self.text.set_screen_size(self.width, self.height)

        # Start a timer to update the vertex data periodically
        self.startTimer(220.0)

    def loadMatricesToShader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        mvp = self.project @ self.view @ self.mouseGlobalTX
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
        rotX = Mat4().rotate_x(self.spinXFace)
        rotY = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX

        # Update model position
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z

        self.loadMatricesToShader()

        # Bind VAO and update vertex data
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
        # self.text.set_screen_size(self.width, self.height)

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
            self.modelPos.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handle mouse movement for rotation and translation.
        """
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diffx = position.x() - self.origX
            diffy = position.y() - self.origY
            self.spinXFace += int(0.5 * diffy)
            self.spinYFace += int(0.5 * diffx)
            self.origX = position.x()
            self.origY = position.y()
            self.update()

        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diffX = int(position.x() - self.origXPos)
            diffY = int(position.y() - self.origYPos)
            self.origXPos = position.x()
            self.origYPos = position.y()
            self.modelPos.x += self.INCREMENT * diffX
            self.modelPos.y -= self.INCREMENT * diffY
            self.update()

    def mousePressEvent(self, event) -> None:
        """
        Handle mouse button press events to start rotation or translation.
        """
        if event.button() == Qt.LeftButton:
            position = event.position()
            self.origX = position.x()
            self.origY = position.y()
            self.rotate = True

        elif event.button() == Qt.RightButton:
            position = event.position()
            self.origXPos = position.x()
            self.origYPos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handle mouse button release events to stop rotation or translation.
        """
        if event.button() == Qt.LeftButton:
            self.rotate = False

        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handle mouse wheel events for zooming in/out.
        """
        numPixels = event.angleDelta()
        if numPixels.x() > 0:
            self.modelPos.z += self.ZOOM
        elif numPixels.x() < 0:
            self.modelPos.z -= self.ZOOM
        self.update()

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
