#!/usr/bin/env -S uv run --script
"""
BoidShaded Example

This script demonstrates how to use OpenGL and Qt to render a simple 3D object (a "boid") with Phong shading.
It sets up a window, handles user input for rotation/translation/zoom, and manages OpenGL resources.
"""

import sys

import OpenGL.GL as gl
from ngl import (
    Mat3,
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
    Vec4,
    VertexData,
    calc_normal,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """
    Main application window for rendering a 3D boid with Phong shading.

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
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Boid")
        self.model_position: Vec3 = Vec3()  # Model position in world space
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
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use("Phong")

        # Set up lighting and material properties
        lightPos = Vec4(-2.0, 5.0, 2.0, 0.0)
        ShaderLib.set_uniform("light.position", lightPos)
        ShaderLib.set_uniform("light.ambient", 0.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("light.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light.specular", 0.8, 0.8, 0.8, 1.0)
        # Gold-like Phong material
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 0.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 0.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 0.0)
        ShaderLib.set_uniform("material.shininess", 51.2)
        ShaderLib.set_uniform("viewerPos", eye)

        self.buildVAO()

    def buildVAO(self) -> None:
        """
        Build the Vertex Array Object (VAO) for the boid geometry.
        """
        print("Building VAO")
        # Define vertices for the boid geometry
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
        # Calculate normals for each triangle and append them
        for i in range(0, len(verts), 3):
            n = calc_normal(verts[i], verts[i + 1], verts[i + 2])
            verts.extend([n, n, n])
        for i in range(0, len(verts)):
            print(verts[i])

        # Create and bind VAO
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self.vao:
            # Set vertex data and attribute pointers not len // 2 is the amount to
            # render not the total size. We have 4 triangles but packed with normals
            # and positions
            data = VertexData(data=verts.to_list(), size=len(verts) // 2)
            self.vao.set_data(data)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)  # Position
            self.vao.set_vertex_attribute_pointer(
                1, 3, gl.GL_FLOAT, 0, 12 * 3
            )  # Normal

        print("VAO created")

    def load_matrices_to_shader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        MV = self.view @ self.mouse_global_tx
        mvp = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("M", self.mouse_global_tx)

    def paintGL(self) -> None:
        """
        Render the scene. Called automatically by Qt.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use("Phong")

        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        self.load_matrices_to_shader()
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
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
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
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
