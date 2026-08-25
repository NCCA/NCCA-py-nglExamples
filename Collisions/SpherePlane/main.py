#!/usr/bin/env -S uv run --script
"""SpherePlane: N falling spheres collide with a tiltable plane.

Ported from NGL9Demos/Collisions/SpherePlane -- default 50 spheres
(configurable via --spheres), fixed 0.2 radius, respawn every 20 ticks,
plane tilt via Up/Down (X axis) and Left/Right (Z axis). No pause
control (the C++ declares but never wires up an animate toggle here --
faithfully left as always-on).
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
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_plane_collide

_PLANE_WIDTH = 5.0
_PLANE_DEPTH = 5.0
_RESPAWN_EVERY = 20


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(random.uniform(-6, 6), 8.0, random.uniform(-6, 6)),
        "dir": Vec3(0.0, -1.0, 0.0),
        "radius": 0.2,
        "hit": False,
    }


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
        self.setTitle("SpherePlane")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.plane_xrot = 0.0
        self.plane_zrot = 0.0
        self.tick_count = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)
        Primitives.create(
            Prims.TRIANGLE_PLANE,
            "plane",
            _PLANE_WIDTH,
            _PLANE_DEPTH,
            1,
            1,
            Vec3(0, 1, 0),
        )
        self.animation_timer.start(130)

    def _plane_normal(self) -> Vec3:
        # Mat4 only multiplies with Vec4 (see ncca.ngl.mat_base.MatrixBase),
        # so the rotation is extracted into a Mat3 which does support
        # Mat3 @ Vec3 -- same pattern as Camera/uvn_camera.py's
        # _rotate_vec3() elsewhere in this repo.
        #
        # This library's Mat3 @ Vec3 multiplies as matrix_data @ v (column-
        # vector convention), but this repo's rendering convention is
        # row-vector (translation in row 3, uploaded row-major with
        # GL_FALSE -- see shader_program.py's _set_matrix_uniform).
        # For a rotation matrix that mismatch is exactly a transpose, i.e.
        # an inverse, so the untransposed multiply silently returns the
        # mirror-image normal at any nonzero tilt. Transposing first makes
        # this match what the vertex shader actually does to a row vector.
        rot = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        return Mat3.from_mat4(rot).transposed() @ Vec3(0, 1, 0)

    def _on_tick(self) -> None:
        normal = self._plane_normal()
        normal_np = np.array([normal.x, normal.y, normal.z])
        for s in self.spheres:
            s["hit"] = False
            s["pos"] = s["pos"] + s["dir"]
            hit = sphere_plane_collide(
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
                np.array([0.0, 0.0, 0.0]),
                normal_np,
                _PLANE_WIDTH,
                _PLANE_DEPTH,
            )
            if hit:
                s["dir"] = normal
                s["hit"] = True

        self.tick_count += 1
        if self.tick_count >= _RESPAWN_EVERY:
            self.tick_count = 0
            for s in self.spheres:
                fresh = _spawn_sphere()
                s["pos"], s["dir"], s["radius"], s["hit"] = (
                    fresh["pos"],
                    fresh["dir"],
                    fresh["radius"],
                    fresh["hit"],
                )
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

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        plane_tilt = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        m = global_tx @ plane_tilt
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        Primitives.draw("plane")

        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m2 = global_tx @ tx.matrix()
            mv2 = self.view @ m2
            ShaderLib.set_uniform("MVP", self.project @ mv2)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m2).inverse().transposed()
            )
            Primitives.draw("sphere")
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.plane_xrot += 1.0
        elif key == Qt.Key_Down:
            self.plane_xrot -= 1.0
        elif key == Qt.Key_Left:
            self.plane_zrot -= 1.0
        elif key == Qt.Key_Right:
            self.plane_zrot += 1.0
        self.update()
        super().keyPressEvent(event)

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
