#!/usr/bin/env -S uv run --script

import sys

import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
    VertexData,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
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
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Boid")
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
        mvp: Mat4 = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self) -> None:
        """
        Called every frame to draw the scene.
        Handles clearing, setting up transformations, and drawing geometry.
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

        self.loadMatricesToShader()
        with self.vao:
            self.vao.draw()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called when the window is resized.
        Updates the viewport and projection matrix.
        """
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)


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
