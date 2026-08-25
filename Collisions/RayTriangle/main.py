#!/usr/bin/env -S uv run --script
"""RayTriangle: N random triangles tested every frame against one
interactively-movable ray.

Ported from NGL9Demos/Collisions/RayTriangle -- default 50 triangles
(configurable via --triangles), keyboard-moved ray endpoints, no
animation timer (matches the C++, which has none for this sub-demo).
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
)
from ncca.ngl.opengl.abstract_vao import VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import ray_triangle_intersect

_STEP = 0.5


def _random_triangle() -> tuple[Vec3, Vec3, Vec3]:
    """One triangle: a centre 10 units out along a random unit vector,
    then 3 verts independently jittered around it -- ported from the
    C++'s `c + Vec3(randomNumber(2)+0.1, ...)` per vertex."""
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    c = Vec3(*(axis * 10.0))
    verts = []
    for _ in range(3):
        verts.append(
            Vec3(
                c.x + random.uniform(-2, 2) + 0.1,
                c.y + random.uniform(-2, 2) + 0.1,
                c.z - random.uniform(0, 2) + 0.1,
            )
        )
    return tuple(verts)


def _calc_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    e1 = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
    e2 = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
    n = np.cross(e1, e2)
    length = np.linalg.norm(n)
    if length > 0:
        n = n / length
    return Vec3(*n)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_triangles: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("RayTriangle")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.triangles = [_random_triangle() for _ in range(num_triangles)]
        # One VAO per triangle, built once in initializeGL and reused every
        # frame -- the C++ original builds each Triangle's VAO once in its
        # own ctor and never touches it again. Rebuilding a fresh VAO per
        # triangle per paintGL call would leak a VBO/VAO pair 50 times a
        # frame, the same class of bug already found and fixed for the ray
        # line in Task 4.
        self.triangle_vaos: list = []
        self.ray_start = Vec3(0, 0, 0.2)
        self.ray_end = Vec3(0, 0, -20)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        # Radius 0.1, not the C++'s literal 0.05 -- the C++ additionally
        # calls setScale(2.0, 2.0, 2.0) on this marker before drawing it,
        # so its actual visual radius is 0.05 * 2.0 = 0.1. This baked mesh
        # skips the extra per-draw scale and bakes the doubled radius in
        # directly instead.
        Primitives.create(Prims.SPHERE, "smallSphere", 0.1, 10)

        # Reused every frame -- rebuilding a fresh VAO per draw call would
        # leak a VBO/VAO pair every tick.
        self.ray_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)

        for v0, v1, v2 in self.triangles:
            normal = _calc_normal(v0, v1, v2)
            data = np.array(
                [
                    v0.x,
                    v0.y,
                    v0.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                    v1.x,
                    v1.y,
                    v1.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                    v2.x,
                    v2.y,
                    v2.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
            with vao:
                vao.set_data(VertexData(data, 3))
                vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 8 * 4, 0)
                vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 8 * 4, 3 * 4)
                vao.set_num_indices(3)
            self.triangle_vaos.append(vao)

    def _draw_line(self, p0: Vec3, p1: Vec3, mvp: Mat4) -> None:
        data = np.array([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], dtype=np.float32)
        with self.ray_vao as vao:
            vao.set_data(VertexData(data, 2))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(2)
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("MVP", mvp)
            vao.draw()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        tx = Transform()
        tx.set_position(self.ray_start.x, self.ray_start.y, self.ray_start.z)
        m = global_tx @ tx.matrix()
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        Primitives.draw("cube")

        mvp_line = self.project @ self.view @ global_tx
        self._draw_line(self.ray_start, self.ray_end, mvp_line)

        ray_start_np = np.array([self.ray_start.x, self.ray_start.y, self.ray_start.z])
        ray_end_np = np.array([self.ray_end.x, self.ray_end.y, self.ray_end.z])

        ShaderLib.use(DefaultShader.DIFFUSE)
        for (v0, v1, v2), vao in zip(self.triangles, self.triangle_vaos):
            hit, hit_point = ray_triangle_intersect(
                ray_start_np,
                ray_end_np,
                np.array([v0.x, v0.y, v0.z]),
                np.array([v1.x, v1.y, v1.z]),
                np.array([v2.x, v2.y, v2.z]),
            )
            self._draw_triangle(vao, v0, hit, hit_point, global_tx)

    def _draw_triangle(
        self, vao, v0: Vec3, hit: bool, hit_point, global_tx: Mat4
    ) -> None:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if hit else gl.GL_FILL)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        m = global_tx
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        with vao:
            vao.draw()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        tx = Transform()
        tx.set_position(v0.x, v0.y, v0.z)
        tx.set_scale(0.06, 0.06, 0.06)
        m2 = global_tx @ tx.matrix()
        mv2 = self.view @ m2
        ShaderLib.set_uniform("MVP", self.project @ mv2)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m2).inverse().transposed())
        Primitives.draw("cube")

        if hit and hit_point is not None:
            tx3 = Transform()
            tx3.set_position(*hit_point)
            m3 = global_tx @ tx3.matrix()
            mv3 = self.view @ m3
            ShaderLib.set_uniform("MVP", self.project @ mv3)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m3).inverse().transposed()
            )
            Primitives.draw("smallSphere")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.ray_end.y += _STEP
        elif key == Qt.Key_Down:
            self.ray_end.y -= _STEP
        elif key == Qt.Key_Left:
            self.ray_end.x -= _STEP
        elif key == Qt.Key_Right:
            self.ray_end.x += _STEP
        elif key == Qt.Key_W:
            self.ray_start.y += _STEP
        elif key == Qt.Key_Z:
            self.ray_start.y -= _STEP
        elif key == Qt.Key_A:
            self.ray_start.x -= _STEP
        elif key == Qt.Key_S:
            self.ray_start.x += _STEP
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangles", type=int, default=50)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow(num_triangles=args.triangles)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
