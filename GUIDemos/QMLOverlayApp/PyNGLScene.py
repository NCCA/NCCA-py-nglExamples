"""QOpenGLWidget teapot scene driven by pre-composed matrices from ncca.ngl.qml models."""

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class PyNGLScene(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_width: int = 1024
        self.window_height: int = 720
        self._model_matrix = Mat4()
        self._view_matrix = Mat4()
        self._colour = Vec3(1.0, 1.0, 0.0)

    @Slot(Mat4)
    def set_model_matrix(self, matrix: Mat4) -> None:
        self._model_matrix = matrix
        self.update()

    @Slot(Mat4)
    def set_view_matrix(self, matrix: Mat4) -> None:
        self._view_matrix = matrix
        self.update()

    @Slot(float, float, float)
    def set_colour(self, r: float, g: float, b: float) -> None:
        self._colour = Vec3(r, g, b)
        self.update()

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.project = perspective(45.0, self.width() / self.height(), 0.1, 100.0)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.DIFFUSE)
        mv = self._view_matrix @ self._model_matrix
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform(
            "Colour", self._colour.x, self._colour.y, self._colour.z, 1.0
        )
        Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
