#!/usr/bin/env -S uv run --script
"""
BoidShaded Example

This script demonstrates how to use OpenGL and Qt to render a simple 3D object (a "boid") with Phong shading.
It sets up a window, handles user input for rotation/translation/zoom, and manages OpenGL resources.
"""

import math
import sys

import OpenGL.GL as gl
from ngl import (
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    Texture,
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

TEXTURE_SHADER = "TextureShader"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """

    Handles OpenGL initialization, rendering, and user input for interactive control.
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
        self.mouse_global_tx: Mat4 = Mat4()
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("VAO Sphere with Texture")
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
        mvp = self.project @ self.view @ self.mouse_global_tx
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
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x

        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z

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
