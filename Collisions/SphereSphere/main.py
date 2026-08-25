#!/usr/bin/env -S uv run --script
"""SphereSphere: 4 fixed spheres, 2 moving and bouncing off 2 static ones.

Ported from NGL9Demos/Collisions/SphereSphere -- exact object count (4
spheres, fixed positions/radii/directions/colours) and collision rules,
no simplification.
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_sphere_collide  # noqa: E402

_SPHERES = [
    {
        "pos": Vec3(-10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(-7.0, 0.0, 0.0),
        "dir": Vec3(0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (1.0, 0.0, 0.0),
    },
    {
        "pos": Vec3(7.0, 0.0, 0.0),
        "dir": Vec3(-0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (0.0, 0.0, 1.0),
    },
]


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("SphereSphere")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [dict(s) for s in _SPHERES]
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)
        self.animation_timer.start(20)

    def _on_tick(self) -> None:
        self.spheres[2]["pos"] = self.spheres[2]["pos"] + self.spheres[2]["dir"]
        self.spheres[3]["pos"] = self.spheres[3]["pos"] + self.spheres[3]["dir"]
        self._check_collisions()
        self.update()

    def _check_collisions(self) -> None:
        s2, s3, s0, s1 = (
            self.spheres[2],
            self.spheres[3],
            self.spheres[0],
            self.spheres[1],
        )
        if sphere_sphere_collide(
            _v3(s2["pos"]), s2["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
            s3["dir"] = s3["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s0["pos"]), s0["radius"], _v3(s2["pos"]), s2["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s1["pos"]), s1["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s3["dir"] = s3["dir"] * -1.0

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
        for s in self.spheres:
            ShaderLib.set_uniform("Colour", *s["colour"], 1.0)
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

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


def _v3(v: Vec3):
    import numpy as np

    return np.array([v.x, v.y, v.z])


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
    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
