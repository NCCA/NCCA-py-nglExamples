#!/usr/bin/env -S uv run --script

import sys

import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    Mat4,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
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
    Main application window for rendering a simple Boid model using OpenGL.

    Handles user interaction (mouse, keyboard), manages OpenGL context,
    and draws a simple geometry using shaders and VAO.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initialize the window and set up default parameters.
        """
        super().__init__()
        self.mouseGlobalTX: Mat4 = Mat4()  # Transformation matrix for mouse interaction
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("Boid")
        self.spinXFace: int = 0  # Rotation around X axis
        self.spinYFace: int = 0  # Rotation around Y axis
        self.rotate: bool = False  # Is the model being rotated?
        self.translate: bool = False  # Is the model being translated?
        self.origX: int = 0  # Original X position for rotation
        self.origY: int = 0  # Original Y position for rotation
        self.origXPos: int = 0  # Original X position for translation
        self.origYPos: int = 0  # Original Y position for translation
        self.INCREMENT: float = 0.01  # Translation increment per pixel
        self.ZOOM: float = 0.1  # Zoom increment
        self.modelPos: Vec3 = Vec3()  # Model position in world space
        self.view: Mat4 = Mat4()  # View matrix
        self.project: Mat4 = Mat4()  # Projection matrix

    def initializeGL(self) -> None:
        """
        Called once to initialize the OpenGL context.
        Sets up background color, depth testing, shaders, and geometry.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Set background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing for 3D
        gl.glEnable(gl.GL_MULTISAMPLE)  # Enable anti-aliasing
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))  # Camera setup
        ShaderLib.use(DefaultShader.COLOUR)  # Use color shader
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)  # Set default color
        self.buildVAO()  # Build geometry

    def buildVAO(self) -> None:
        """
        Creates and sets up the Vertex Array Object (VAO) for the Boid geometry.
        """
        # Define vertices for the Boid model
        # fmt : off
        verts = Vec3Array(
            [
                Vec3(0.0, 1.0, 1.0),
                Vec3(0.0, 0.0, -1.0),
                Vec3(-0.5, 0.0, 1.0),
                Vec3(0.0, 1.0, 1.0),
                Vec3(0.0, 0.0, -1.0),
                Vec3(0.5, 0.0, 1.0),
                Vec3(0.0, 1.0, 1.0),
                Vec3(0.0, 0.0, 1.5),
                Vec3(-0.5, 0.0, 1.0),
                Vec3(0.0, 1.0, 1.0),
                Vec3(0.0, 0.0, 1.5),
                Vec3(0.5, 0.0, 1.0),
            ]
        )
        # fmt : on
        # Create VAO and bind vertex data
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self.vao:
            data: VertexData = VertexData(data=verts.to_list(), size=len(verts))
            self.vao.set_data(data)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)

    def loadMatricesToShader(self) -> None:
        """
        Loads the Model-View-Projection (MVP) matrix to the shader.
        """
        mvp: Mat4 = self.project @ self.view @ self.mouseGlobalTX
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self) -> None:
        """
        Called every frame to draw the scene.
        Handles clearing, setting up transformations, and drawing geometry.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        # Apply rotation based on user input
        rotX: Mat4 = Mat4().rotate_x(self.spinXFace)
        rotY: Mat4 = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX
        # Apply translation
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z
        self.loadMatricesToShader()
        with self.vao:
            self.vao.draw()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called when the window is resized.
        Updates the viewport and projection matrix.
        """
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard events for controlling the model and rendering mode.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)  # Wireframe mode
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)  # Fill mode
        elif key == Qt.Key_Space:
            self.spinXFace = 0
            self.spinYFace = 0
            self.modelPos.set(0, 0, 0)  # Reset model position and rotation
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement for rotating and translating the model.
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
        Handles mouse button press events to start rotation or translation.
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
        Handles mouse button release events to stop rotation or translation.
        """
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming in and out.
        """
        numPixels = event.pixelDelta()
        if numPixels.x() > 0:
            self.modelPos.z += self.ZOOM
        elif numPixels.x() < 0:
            self.modelPos.z -= self.ZOOM
        self.update()


if __name__ == "__main__":
    # Entry point for the application
    app: QApplication = QApplication(sys.argv)
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)  # Enable anti-aliasing
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)  # Set depth buffer size
    QSurfaceFormat.setDefaultFormat(format)  # Apply format globally

    window: MainWindow = MainWindow()
    window.setFormat(format)
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
