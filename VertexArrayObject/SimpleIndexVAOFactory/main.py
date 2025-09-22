#!/usr/bin/env -S uv run --script

import sys

import OpenGL.GL as gl
from ngl import (
    IndexVertexData,
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )

        self.width = int(1024)
        self.height = int(720)
        self.setTitle("SimpleIndexVAOFactory")
        self.view = Mat4()
        self.project = Mat4()
        self.vao = None

    def initializeGL(self):
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        eye = Vec3(0, 1, 2)
        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.view = look_at(eye, to, up)
        if not ShaderLib.load_shader(
            "Colour", "shaders/ColourVertex.glsl", "shaders/ColourFragment.glsl"
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use("Colour")

        self.buildVAO()

    def buildVAO(self):
        print("Building VAO")
        # fmt: off
        verts = Vec3Array([
        Vec3(-0.26286500, 0.0000000, 0.42532500), Vec3(1.0,0.0,0.0),
        Vec3(0.26286500, 0.0000000, 0.42532500), Vec3(1.0,0.55,0.0),
        Vec3(-0.26286500, 0.0000000, -0.42532500),  Vec3(1.0,0.0,1.0),
        Vec3(0.26286500, 0.0000000, -0.42532500),  Vec3(0.0,1.0,0.0),
        Vec3(0.0000000, 0.42532500, 0.26286500),  Vec3(0.0,0.0,1.0),
        Vec3(0.0000000, 0.42532500, -0.26286500),  Vec3(0.29,0.51,0.0),
        Vec3(0.0000000, -0.42532500, 0.26286500),  Vec3(0.5,0.0,0.5),
        Vec3(0.0000000, -0.42532500, -0.26286500),  Vec3(1.0,1.0,1.0),
        Vec3(0.42532500, 0.26286500, 0.0000000),  Vec3(0.0,1.0,1.0),
        Vec3(-0.42532500, 0.26286500, 0.0000000),  Vec3(0.0,0.0,0.0),
        Vec3(0.42532500, -0.26286500, 0.0000000),  Vec3(0.12,0.56,1.0),
        Vec3(-0.42532500, -0.26286500, 0.0000000),  Vec3(0.86,0.08,0.24)
        ])

        indices=[0,6,1,0,11,6,1,4,0,1,8,4,1,10,8,2,5,3,
            2,9,5,2,11,9,3,7,2,3,10,7,4,8,5,4,9,0,
            5,8,3,5,9,4,6,10,1,6,11,7,7,10,6,7,11,2,
            8,10,3,9,11,0]

        # fmt: on

        self.vao = VAOFactory.create_vao(VAOType.SIMPLE_INDEX, gl.GL_TRIANGLES)
        with self.vao:
            data = IndexVertexData(
                data=verts.to_list(),
                size=len(indices),
                indices=indices,
                index_type=gl.GL_UNSIGNED_SHORT,
            )
            self.vao.set_data(data)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 24, 0)
            # 12 is the offset for the second attribute 3 * 4 bytes for a Vec3 use size of Vec3
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 24, Vec3.sizeof())
            print("VAO created")

    def loadMatricesToShader(self):
        ShaderLib.use("Colour")
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self):
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
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

    def resizeGL(self, w, h):
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    format = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    # now set the depth buffer to 24 bits
    format.setDepthBufferSize(24)
    # set that as the default format for all windows
    QSurfaceFormat.setDefaultFormat(format)

    window = MainWindow()
    window.setFormat(format)
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
