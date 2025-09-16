#!/usr/bin/env -S uv run --script

import sys

import OpenGL.GL as gl
from ngl import (
    Mat4,
    ShaderLib,
    VAOFactory,
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
    def __init__(self, parent=None):
        QOpenGLWindow.__init__(self)
        # super(QOpenGLWindow, self).__init__(parent)
        self.mouseGlobalTX = Mat4()
        self.width = int(1024)
        self.height = int(720)
        self.setTitle("Boid")
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
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use("nglColourShader")
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        self.buildVAO()

    # def buildVAO(self):
    #     # fmt: off
    #     verts = [
    #         0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -0.5, 0.0, 1.0,
    #         0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.5, 0.0, 1.0,
    #         0.0, 1.0, 1.0, 0.0, 0.0, 1.5, -0.5, 0.0, 1.0,
    #         0.0, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5, 0.0, 1.0,
    #     ]
    #     # fmt: on

    #     self.vao = VAOFactory.create_vao("simpleVAO", gl.GL_TRIANGLES)
    #     self.vao.bind()
    #     data = VertexData(data=verts, size=len(verts) // 3)
    #     self.vao.set_data(data)
    #     self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
    #     self.vao.unbind()

    def buildVAO(self):
        # fmt: off
        # Vec3Array is handy for storing 3D vectors we can easily manipulate and access
        # but opengl just wants them in a flat list so call to_list() when needed
        verts = Vec3Array([
            Vec3(0.0, 1.0, 1.0), Vec3(0.0, 0.0, -1.0), Vec3(-0.5, 0.0, 1.0),
            Vec3(0.0, 1.0, 1.0), Vec3(0.0, 0.0, -1.0), Vec3(0.5, 0.0, 1.0),
            Vec3(0.0, 1.0, 1.0), Vec3(0.0, 0.0, 1.5), Vec3(-0.5, 0.0, 1.0),
            Vec3(0.0, 1.0, 1.0), Vec3(0.0, 0.0, 1.5), Vec3(0.5, 0.0, 1.0),
            Vec3(0.0, 1.0, 1.0), Vec3(0.0, 0.0, 1.5), Vec3(0.5, 0.0, 1.0),
        ])
        # fmt: on

        self.vao = VAOFactory.create_vao("simpleVAO", gl.GL_TRIANGLES)
        self.vao.bind()
        data = VertexData(data=verts.to_list(), size=len(verts))
        self.vao.set_data(data)
        self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
        self.vao.unbind()

    def loadMatricesToShader(self):
        mvp = self.project @ self.view @ self.mouseGlobalTX
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self):
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use("nglColourShader")
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        rotX = Mat4().rotate_x(self.spinXFace)
        rotY = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z
        self.loadMatricesToShader()
        self.vao.bind()
        self.vao.draw()
        self.vao.unbind()

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
        self.update()
        super().keyPressEvent(event)

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
