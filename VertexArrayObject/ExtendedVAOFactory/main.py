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
from MultiBufferIndexVAO import MultiBufferIndexVAO, VertexData
from ngl import (
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    Transform,
    VAOFactory,
    Vec3,
    Vec3Array,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

COLOUR_SHADER = "ColourShader"


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
        self.setTitle("Extended VAO")
        self.view: Mat4 = Mat4()  # View matrix
        self.project: Mat4 = Mat4()  # Projection matrix
        self.data: list[float] = []  # Dynamic vertex data
        self.index: int = 0

    def initializeGL(self) -> None:
        """
        Set up OpenGL context, load shaders, and initialize scene.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Set background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
        gl.glEnable(gl.GL_MULTISAMPLE)  # Enable anti-aliasing
        # first Register VAO creators
        VAOFactory.register_vao_creator("MultiBufferIndexVAO", MultiBufferIndexVAO)

        # Use a simple colour shader
        if not ShaderLib.load_shader(
            COLOUR_SHADER,
            vert="shaders/ColourVertex.glsl",
            frag="shaders/ColourFragment.glsl",
        ):
            logging.error("Error loading shaders")
            self.close()
        ShaderLib.use(COLOUR_SHADER)

        # Set up camera/view matrix
        self.view = look_at(Vec3(0, 1, 3), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.001, 500.0)
        self.build_vao()

        # Start a timer to update the vertex data periodically
        self.startTimer(160)

    def build_vao(self):
        # fmt: off
        verts = Vec3Array([
            Vec3(-0.26286500 , 0.0000000 , 0.42532500 ),
            Vec3(0.26286500 , 0.0000000 , 0.42532500 ),
            Vec3(-0.26286500 , 0.0000000 , -0.42532500 ),
            Vec3(0.26286500 , 0.0000000 , -0.42532500 ),
            Vec3(0.0000000 , 0.42532500 , 0.26286500 ),
            Vec3(0.0000000 , 0.42532500 , -0.26286500 ),
            Vec3(0.0000000 , -0.42532500 , 0.26286500 ),
            Vec3(0.0000000 , -0.42532500 , -0.26286500 ),
            Vec3(0.42532500 , 0.26286500 , 0.0000000 ),
            Vec3(-0.42532500 , 0.26286500 , 0.0000000 ),
            Vec3(0.42532500 , -0.26286500 , 0.0000000 ),
            Vec3(-0.42532500 , -0.26286500 , 0.0000000 )
        ])

        colours = Vec3Array([
            Vec3(1.0, 0.0, 0.0),
            Vec3(1.0, 0.55, 0.0),
            Vec3(1.0, 0.0, 1.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.29, 0.51, 0.0),
            Vec3(0.5, 0.0, 0.5),
            Vec3(1.0, 1.0, 1.0),
            Vec3(0.0, 1.0, 1.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.12, 0.56, 1.0),
            Vec3(0.86, 0.08, 0.24),
        ])

        indices = [
            0, 6, 1, 0, 11, 6, 1, 4, 0, 1, 8, 4, 1, 10, 8, 2, 5, 3, 2, 9, 5, 2, 11, 9, 3, 7, 2, 3, 10, 7,
            4, 8, 5, 4, 9, 0, 5, 8, 3, 5, 9, 4, 6, 10, 1, 6, 11, 7, 7, 10, 6, 7, 11, 2, 8, 10, 3, 9, 11, 0,
        ]

        self.num_indices = len(indices)
        # fmt: on
        self.vao = VAOFactory.create_vao("MultiBufferIndexVAO", gl.GL_TRIANGLES)

        with self.vao:
            # As this is a Multi buffer VAO we can add two initial buffer one for Vertex and one for Color
            data = VertexData(data=verts.to_list(), size=len(verts))
            self.vao.set_data(data)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)

            colour_data = VertexData(data=colours.to_list(), size=len(colours))
            self.vao.set_data(colour_data)
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 0, 0)

            self.vao.set_indices(indices, gl.GL_UNSIGNED_SHORT)
            self.vao.set_num_indices(len(indices))

    def paintGL(self) -> None:
        """
        Render the scene. Called automatically by Qt.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(COLOUR_SHADER)

        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        mouse_global_tx = rot_y @ rot_x

        # Update model position
        mouse_global_tx[3][0] = self.model_position.x
        mouse_global_tx[3][1] = self.model_position.y
        mouse_global_tx[3][2] = self.model_position.z

        with self.vao:
            t = Transform()
            t.set_position(-1.2, 0.0, 0.0)
            mvp = self.project @ self.view @ t.get_matrix() @ mouse_global_tx
            ShaderLib.set_uniform("MVP", mvp)
            self.vao.draw(0, self.index * 3)
            t.set_position(0.0, 0.0, 0.0)
            mvp = self.project @ self.view @ t.get_matrix() @ mouse_global_tx
            ShaderLib.set_uniform("MVP", mvp)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            self.vao.draw()
            t.set_position(1.2, 0.0, 0.0)
            mvp = self.project @ self.view @ t.get_matrix() @ mouse_global_tx
            ShaderLib.set_uniform("MVP", mvp)
            self.vao.draw(self.index * 3, 3)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            self.vao.draw()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle window resizing and update the projection matrix.

        Args:
            w: New window width.
            h: New window height.
        """
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def timerEvent(self, event) -> None:
        """
        Periodically called by Qt to update the index for drawing.
        """
        self.index += 3
        if self.index >= self.num_indices // 3:
            self.index = 0
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
