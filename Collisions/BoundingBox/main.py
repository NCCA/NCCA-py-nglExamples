#!/usr/bin/env -S uv run --script
"""BoundingBox: N spheres bounce inside a cubic bounding box, with an
optional all-pairs sphere/sphere check.

Ported from NGL9Demos/Collisions/BoundingBox -- default 50 spheres
(configurable via --spheres), variable count at runtime (+/- keys,
minimum 1), half-extent-40 cube, optional sphere-sphere checking (S key,
default off).
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
from collision_maths import sphere_bbox_reflect, sphere_sphere_collide  # noqa: E402

_HALF_EXTENT = 40.0


def _random_unit_vec3() -> Vec3:
    v = np.random.normal(size=3)
    v = v / np.linalg.norm(v)
    return Vec3(*v)


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(
            random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)
        ),
        "dir": _random_unit_vec3(),
        "radius": random.uniform(0.5, 2.5),
        "hit": False,
    }


_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom face (y = -h)
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top face (y = +h)
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # connecting edges
)


def _box_corners(h: float) -> list[Vec3]:
    return [
        Vec3(-h, -h, -h),
        Vec3(h, -h, -h),
        Vec3(h, -h, h),
        Vec3(-h, -h, h),
        Vec3(-h, h, -h),
        Vec3(h, h, -h),
        Vec3(h, h, h),
        Vec3(-h, h, h),
    ]


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
        self.setTitle("BoundingBox")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.animate = True
        self.check_sphere_sphere = False
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.box_vao = None

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 80, 80), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)

        # The 12-edge wireframe box never changes shape (it's a fixed
        # half-extent-40 cube), so its vertex data is uploaded once here
        # rather than every paintGL call -- only the draw() call repeats.
        self.box_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        corners = _box_corners(_HALF_EXTENT)
        verts: list[float] = []
        for a, b in _BOX_EDGES:
            verts.extend((corners[a].x, corners[a].y, corners[a].z))
            verts.extend((corners[b].x, corners[b].y, corners[b].z))
        data = np.array(verts, dtype=np.float32)
        with self.box_vao as vao:
            vao.set_data(VertexData(data, len(_BOX_EDGES) * 2))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(len(_BOX_EDGES) * 2)

        self.animation_timer.start(40)

    def _on_tick(self) -> None:
        if not self.animate:
            return
        # Reset every sphere's hit flag before re-testing -- forgetting
        # this leaves a sphere permanently wireframed after its one and
        # only collision, a bug class already caught in SpherePlane.
        for s in self.spheres:
            s["hit"] = False
            s["pos"] = s["pos"] + s["dir"]

        if self.check_sphere_sphere:
            for current in self.spheres:
                for other in self.spheres:
                    if current is other:
                        continue
                    if sphere_sphere_collide(
                        np.array([other["pos"].x, other["pos"].y, other["pos"].z]),
                        other["radius"],
                        np.array(
                            [current["pos"].x, current["pos"].y, current["pos"].z]
                        ),
                        current["radius"],
                    ):
                        # Asymmetric by design, straight from the C++: only
                        # the outer/"current" sphere reverses and flags --
                        # the "other" sphere is untouched here (it gets its
                        # own turn as the outer sphere later in this pass).
                        current["dir"] = current["dir"] * -1.0
                        current["hit"] = True

        # Wall reflection always runs, regardless of the S toggle.
        for s in self.spheres:
            hit, new_dir = sphere_bbox_reflect(
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                np.array([s["dir"].x, s["dir"].y, s["dir"].z]),
                s["radius"],
                _HALF_EXTENT,
            )
            if hit:
                s["dir"] = Vec3(*new_dir)
                s["hit"] = True
        self.update()

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

        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        with self.box_vao as vao:
            vao.draw()

        ShaderLib.use(DefaultShader.DIFFUSE)
        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
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

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        # Mirrors PySideEventHandlingMixin's own if/elif/else dispatch:
        # keys this demo owns are handled here and never fall through, so
        # Key_Space toggles `animate` only -- it must not also trigger the
        # mixin's own Key_Space handler (reset_camera), which would reset
        # the orbit/pan on every pause. Key_S is likewise ours alone here
        # (sphere/sphere toggle), not the mixin's solid-fill shortcut.
        # Anything we don't recognise still goes to super() as before.
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Space:
            self.animate = not self.animate
        elif key == Qt.Key_S:
            self.check_sphere_sphere = not self.check_sphere_sphere
        elif key == Qt.Key_R:
            self.spheres = [_spawn_sphere() for _ in range(len(self.spheres))]
        elif key == Qt.Key_Minus:
            if len(self.spheres) > 1:
                self.spheres.pop()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.spheres.append(_spawn_sphere())
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
