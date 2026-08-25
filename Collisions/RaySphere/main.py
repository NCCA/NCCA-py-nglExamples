#!/usr/bin/env -S uv run --script
"""RaySphere: N spheres tested each tick against 2 sweeping rays.

Ported from NGL9Demos/Collisions/RaySphere -- default 50 spheres
(configurable via --spheres), 2 animated rays sweeping in opposite x
directions, wireframe-on-hit, near/far hit-point markers.
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
from collision_maths import ray_sphere_intersect


def _hit_points(ray_start: Vec3, ray_dir: Vec3, sphere_pos: Vec3, radius: float):
    """Quadratic-root near/far hit points, for drawing only -- ported from
    NGL9Demos's drawHitPoints(). Returns (near, far) Vec3 or (None, None)."""
    d = np.array([ray_dir.x, ray_dir.y, ray_dir.z])
    d = d / np.linalg.norm(d)
    p = np.array([ray_start.x, ray_start.y, ray_start.z]) - np.array(
        [sphere_pos.x, sphere_pos.y, sphere_pos.z]
    )
    a = float(d @ d)
    b = 2.0 * float(d @ p)
    c = float(p @ p) - radius * radius
    discrim = b * b - 4.0 * a * c
    if discrim < 0.0:
        return None, None
    root = discrim**0.5
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    o = np.array([ray_start.x, ray_start.y, ray_start.z])
    h1 = o + d * t1
    h2 = o + d * t2
    return Vec3(*h1), Vec3(*h2)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_spheres: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("RaySphere")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()

        self.spheres = [
            {
                "pos": Vec3(random.uniform(0, 10), random.uniform(0, 8), 0.0),
                "radius": random.uniform(0, 1) + 0.2,
                "hit": False,
            }
            for _ in range(num_spheres)
        ]
        self.ray1_start = Vec3(0, 10, 0)
        self.ray1_end = Vec3(0, -5, 0)
        self.ray2_start = Vec3(0, 0, 20)
        self.ray2_end = Vec3(0, 0, -5)
        self._sweep_forward = True
        self.animate = True
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, -25), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)
        Primitives.create(Prims.SPHERE, "smallSphere", 0.2, 10)
        # Reused every frame for both rays -- rebuilding a fresh VAO per draw
        # call would leak a VBO/VAO pair every tick.
        self.ray_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        self.animation_timer.start(50)

    def _on_tick(self) -> None:
        if not self.animate:
            return
        for s in self.spheres:
            hit1 = ray_sphere_intersect(
                np.array([self.ray1_start.x, self.ray1_start.y, self.ray1_start.z]),
                np.array(
                    [
                        self.ray1_end.x - self.ray1_start.x,
                        self.ray1_end.y - self.ray1_start.y,
                        self.ray1_end.z - self.ray1_start.z,
                    ]
                ),
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
            )
            hit2 = ray_sphere_intersect(
                np.array([self.ray2_start.x, self.ray2_start.y, self.ray2_start.z]),
                np.array(
                    [
                        self.ray2_end.x - self.ray2_start.x,
                        self.ray2_end.y - self.ray2_start.y,
                        self.ray2_end.z - self.ray2_start.z,
                    ]
                ),
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
            )
            s["hit"] = hit1 or hit2

        step = 0.5 if self._sweep_forward else -0.5
        self.ray1_end.x += step
        self.ray2_end.x -= step
        if self.ray1_end.x > 22.0:
            self._sweep_forward = False
        elif self.ray1_end.x <= -22.0:
            self._sweep_forward = True
        self.update()

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

        # Small cube markers at each ray's fixed origin.
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        for start in (self.ray1_start, self.ray2_start):
            tx = Transform()
            tx.set_position(start.x, start.y, start.z)
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("cube")

        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
            ShaderLib.use(DefaultShader.DIFFUSE)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("sphere")
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if s["hit"]:
                for ray_start, ray_end in (
                    (self.ray1_start, self.ray1_end),
                    (self.ray2_start, self.ray2_end),
                ):
                    ray_dir = Vec3(
                        ray_end.x - ray_start.x,
                        ray_end.y - ray_start.y,
                        ray_end.z - ray_start.z,
                    )
                    near, far = _hit_points(ray_start, ray_dir, s["pos"], s["radius"])
                    if near is None:
                        continue
                    for point, colour in (
                        (near, (1.0, 0.0, 0.0)),
                        (far, (0.0, 1.0, 0.0)),
                    ):
                        ShaderLib.use(DefaultShader.DIFFUSE)
                        ShaderLib.set_uniform("Colour", *colour, 1.0)
                        tx2 = Transform()
                        tx2.set_position(point.x, point.y, point.z)
                        m2 = global_tx @ tx2.matrix()
                        mv2 = self.view @ m2
                        ShaderLib.set_uniform("MVP", self.project @ mv2)
                        ShaderLib.set_uniform(
                            "normalMatrix", Mat3.from_mat4(m2).inverse().transposed()
                        )
                        Primitives.draw("smallSphere")

        for ray_start, ray_end in (
            (self.ray1_start, self.ray1_end),
            (self.ray2_start, self.ray2_end),
        ):
            mvp = self.project @ self.view @ global_tx
            self._draw_line(ray_start, ray_end, mvp)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        # Mirrors PySideEventHandlingMixin's own if/elif/else dispatch:
        # keys this demo owns are handled here and never fall through, so
        # Key_Space toggles `animate` only -- it must not also trigger the
        # mixin's own Key_Space handler (reset_camera), which would reset
        # the orbit/pan on every pause. Anything we don't recognise still
        # goes to super() as before.
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Space:
            self.animate = not self.animate
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


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
    parser.add_argument("--spheres", type=int, default=50)
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
    window = MainWindow(num_spheres=args.spheres)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
