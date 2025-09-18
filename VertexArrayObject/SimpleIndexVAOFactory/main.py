#!/usr/bin/env -S uv run --script

import sys

import OpenGL.GL as gl
from ngl import (
    IndexVertexData,
    Mat4,
    ShaderLib,
    VAOFactory,
    VAOType,
    Vec3,
    Vec3Array,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(QOpenGLWindow):
    def __init__(self, parent=None):
        QOpenGLWindow.__init__(self)
        self.mouseGlobalTX = Mat4()
        self.width = int(1024)
        self.height = int(720)
        self.setTitle("SimpleIndexVAOFactory")
        self.spinXFace = int(0)
        self.spinYFace = int(0)
        self.rotate = False
        self.translate = False
        self.origX = int(0)
        self.origY = int(0)
        self.origXPos = int(0)
        self.origYPos = int(0)
        self.INCREMENT = 0.01
        self.ZOOM = 0.1
        self.modelPos = Vec3()
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
        mvp = self.project @ self.view @ self.mouseGlobalTX
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self):
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        rotX = Mat4().rotate_x(self.spinXFace)
        rotY = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z
        self.loadMatricesToShader()
        with self.vao:
            self.vao.draw()

    def resizeGL(self, w, h):
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            self.spinXFace = 0
            self.spinYFace = 0
            self.modelPos.set(0, 0, 0)
        else:
            # always call the base implementation for unhandled keys
            super().keyPressEvent(event)
        self.update()

    def mouseMoveEvent(self, event):
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

    def mousePressEvent(self, event):
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rotate = False

        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event):
        numPixels = event.pixelDelta()

        if numPixels.x() > 0:
            self.modelPos.z += self.ZOOM

        elif numPixels.x() < 0:
            self.modelPos.z -= self.ZOOM
        self.update()


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
