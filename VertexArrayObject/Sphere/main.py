#!/usr/bin/env -S uv run --script
"""
BoidShaded Example

This script demonstrates how to use OpenGL and Qt to render a simple 3D object (a "boid") with Phong shading.
It sets up a window, handles user input for rotation/translation/zoom, and manages OpenGL resources.
"""

import math
import sys
import traceback

import OpenGL.GL as gl
from ngl import (
    Mat4,
    ShaderLib,
    Texture,
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

TEXTURE_SHADER = "TextureShader"


class MainWindow(QOpenGLWindow):
    """

    Handles OpenGL initialization, rendering, and user input for interactive control.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initialize the window and set up default parameters.
        """
        super().__init__()
        self.mouseGlobalTX: Mat4 = Mat4()
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("VAO Sphere with Texture")
        self.spinXFace: int = 0  # Rotation around X axis
        self.spinYFace: int = 0  # Rotation around Y axis
        self.rotate: bool = False
        self.translate: bool = False
        self.origX: int = 0
        self.origY: int = 0
        self.origXPos: int = 0
        self.origYPos: int = 0
        self.INCREMENT: float = 0.01  # Translation increment per pixel
        self.ZOOM: float = 0.1  # Zoom increment
        self.modelPos: Vec3 = Vec3()  # Model position in world space
        self.view: Mat4 = Mat4()  # View matrix
        self.project: Mat4 = Mat4()  # Projection matrix

    def initializeGL(self) -> None:
        """
        Set up OpenGL context, load shaders, and initialize scene.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Set background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
        gl.glEnable(gl.GL_MULTISAMPLE)  # Enable anti-aliasing

        # Set up camera/view matrix
        eye = Vec3(0, 1, 4)
        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.view = look_at(eye, to, up)

        # Load and use Phong shader
        if not ShaderLib.load_shader(
            TEXTURE_SHADER, "shaders/TextureVertex.glsl", "shaders/TextureFragment.glsl"
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use(TEXTURE_SHADER)
        texture = Texture("textures/earth.png")
        self.texture_id = texture.set_texture_gl()

        self.build_vao_sphere()

    def build_vao_sphere(self, radius: float = 1.0, precision: int = 100):
        """
        Creates a sphere VAO using triangle strips.
        based on an algorithm by Paul Bourke.
        http://astronomy.swin.edu.au/~pbourke/opengl/sphere/

        Args:
            radius: The radius of the sphere.
            precision: The number of divisions around the sphere. Higher is more detailed.

        Returns:
            A configured ngl.AbstractVAO containing the sphere geometry.
        """
        # In NGL, "simpleVAO" is a basic VAO that holds interleaved data in a single buffer.
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLE_STRIP)

        if radius < 0:
            radius = -radius
        if precision < 4:
            precision = 4

        vertex_data = []

        for i in range(precision // 2):
            theta1 = i * (2 * math.pi) / precision - (math.pi / 2)
            theta2 = (i + 1) * (2 * math.pi) / precision - (math.pi / 2)

            for j in range(precision + 1):
                theta3 = j * (2 * math.pi) / precision

                # Vertex 1 (for the top of the strip)
                nx1 = math.cos(theta2) * math.cos(theta3)
                ny1 = math.sin(theta2)
                nz1 = math.cos(theta2) * math.sin(theta3)
                x1 = radius * nx1
                y1 = radius * ny1
                z1 = radius * nz1
                u1 = j / precision
                v1 = 1.0 - (2 * (i + 1) / precision)
                vertex_data.extend([x1, y1, z1, nx1, ny1, nz1, u1, v1])

                # Vertex 2 (for the bottom of the strip)
                nx2 = math.cos(theta1) * math.cos(theta3)
                ny2 = math.sin(theta1)
                nz2 = math.cos(theta1) * math.sin(theta3)
                x2 = radius * nx2
                y2 = radius * ny2
                z2 = radius * nz2
                u2 = j / precision
                v2 = 1.0 - (2 * i / precision)
                vertex_data.extend([x2, y2, z2, nx2, ny2, nz2, u2, v2])

        num_verts = len(vertex_data) // 8

        with self.vao:
            data = VertexData(data=vertex_data, size=len(vertex_data))
            self.vao.set_data(data)

            # Stride is 8 floats * 4 bytes/float = 32 bytes
            stride = 8 * 4

            # Set attribute pointers for the interleaved data
            # Attribute 0: Vertex (x, y, z)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)
            # Attribute 1: Normal (nx, ny, nz) - offset is 3 floats (12 bytes)
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, stride, 3 * 4)
            # Attribute 2: UV (u, v) - offset is 6 floats (24 bytes)
            self.vao.set_vertex_attribute_pointer(2, 2, gl.GL_FLOAT, stride, 6 * 4)

            # Set the number of vertices to draw
            self.vao.set_num_indices(num_verts)

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
        ShaderLib.use(TEXTURE_SHADER)

        # Apply rotation based on user input
        rotX = Mat4().rotate_x(self.spinXFace)
        rotY = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX
        # Update model position
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z
        self.loadMatricesToShader()
        # Draw geometry
        with self.vao:
            self.vao.draw()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle window resizing and update the projection matrix.

        Args:
            w: New window width.
            h: New window height.
        """

        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.1, 350.0)

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
        else:
            # Call base implementation for unhandled keys
            super().keyPressEvent(event)
        self.update()

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


def except_hook(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)  # forward to default
    sys.exit(1)  # optional: quit app on error


if __name__ == "__main__":
    # Set up Qt application and OpenGL format
    sys.excepthook = except_hook
    #    app = QApplication(sys.argv)
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
