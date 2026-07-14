#!/usr/bin/env -S uv run --script
"""Sphere-traced signed distance fields (OpenGL).

The whole scene -- a ground plane, a sphere, a box and a torus melted
together with a smooth minimum, plus one sphere orbiting overhead -- is
described entirely inside a single fragment shader (shaders/RayMarchFragment
.glsl). There is no geometry: three vertices from ScreenTri's no-VBO
fullscreen-triangle trick are enough, because every pixel just fires a ray
from the camera and walks it forward in steps sized by the distance field
until it lands on a surface.

The maths that decides those distances (sd_sphere, sd_box, sd_torus,
sd_plane, smooth_min, scene) lives twice more: once in sdf_maths.py as a
tested numpy mirror, and once in RayMarch.wgsl for the WebGPU sibling of
this demo. All three are deliberately line-for-line the same function, so
you can hold the GLSL and WGSL side by side in the README and see there is
nothing backend-specific about ray marching itself.

Controls:
    S      toggle soft shadows
    O      toggle ambient occlusion
    N      visualise surface normals
    I      visualise the iteration count as a heat map
    +/-    widen/narrow the smooth-min blend radius
    Space  pause/resume the orbiting sphere
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import time
import traceback
from math import cos, radians, sin
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Vec3, logger
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib, Text
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

RAY_MARCH_SHADER = "RayMarch"
BASE_DISTANCE = 6.0
TARGET_HEIGHT = 0.8
FOV_DEGREES = 45.0


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Fullscreen ray-marched SDF scene with the standard PyNGL mouse orbit."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.3,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Ray Marching SDF (OpenGL)")
        self.vao = None

        # --- demo state driven by the keyboard ---
        self.shadows_on = True
        self.ao_on = True
        self.show_normals = False
        self.show_iterations = False
        self.smooth_k = 0.3
        self.paused = False
        self._clock_start = time.perf_counter()
        self._paused_time = 0.0

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            RAY_MARCH_SHADER,
            str(shader_dir / "RayMarchVertex.glsl"),
            str(shader_dir / "RayMarchFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()

        self.vao = gl.glGenVertexArrays(1)
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )
        self.startTimer(16)

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def _elapsed_time(self) -> float:
        if self.paused:
            return self._paused_time
        return time.perf_counter() - self._clock_start

    def _camera_basis(self):
        """Orbit camera driven by the standard mouse rotate/pan/zoom state:
        spin_x/y_face are pitch/yaw, model_position.z zooms, model_position
        .x/y pans the look-at target. Returns (pos, forward, right, up) as
        Vec3, which the shader combines with each pixel's screen offset to
        build its ray direction -- see camPos/camForward/camRight/camUp in
        RayMarchFragment.glsl.
        """
        yaw = radians(self.spin_y_face)
        pitch = radians(self.spin_x_face)
        distance = max(1.0, BASE_DISTANCE - self.model_position.z)
        target = Vec3(self.model_position.x, TARGET_HEIGHT + self.model_position.y, 0.0)
        offset = Vec3(
            distance * cos(pitch) * sin(yaw),
            distance * sin(pitch),
            distance * cos(pitch) * cos(yaw),
        )
        pos = target + offset
        forward = (target - pos).normalized()
        world_up = Vec3(0.0, 1.0, 0.0)
        right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()
        return pos, forward, right, up

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        ShaderLib.use(RAY_MARCH_SHADER)
        pos, forward, right, up = self._camera_basis()
        ShaderLib.set_uniform("camPos", pos)
        ShaderLib.set_uniform("camForward", forward)
        ShaderLib.set_uniform("camRight", right)
        ShaderLib.set_uniform("camUp", up)
        ShaderLib.set_uniform(
            "fovScale",
            float(sin(radians(FOV_DEGREES) * 0.5) / cos(radians(FOV_DEGREES) * 0.5)),
        )
        ShaderLib.set_uniform(
            "aspect", float(self.window_width) / float(self.window_height)
        )
        ShaderLib.set_uniform("time", self._elapsed_time())
        ShaderLib.set_uniform("smoothK", self.smooth_k)
        ShaderLib.set_uniform("shadowsOn", 1 if self.shadows_on else 0)
        ShaderLib.set_uniform("aoOn", 1 if self.ao_on else 0)
        ShaderLib.set_uniform("showNormals", 1 if self.show_normals else 0)
        ShaderLib.set_uniform("showIterations", 1 if self.show_iterations else 0)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

        self._draw_hud()

    def _draw_hud(self) -> None:
        state = (
            f"[S]hadows {'ON ' if self.shadows_on else 'OFF'}  "
            f"[O] AO {'ON ' if self.ao_on else 'OFF'}  "
            f"[N]ormals {'ON ' if self.show_normals else 'OFF'}  "
            f"[I]terations {'ON ' if self.show_iterations else 'OFF'}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))
        Text.render_text(
            "Arial",
            10,
            45,
            f"[+/-] smooth-min k = {self.smooth_k:.2f}   "
            f"[Space] {'paused' if self.paused else 'running'}",
            Vec3(1.0, 1.0, 1.0),
        )

    def timerEvent(self, event) -> None:
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        ratio = self.devicePixelRatio()
        self.window_width = int(w * ratio)
        self.window_height = int(h * ratio)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_S:
            self.shadows_on = not self.shadows_on
        elif key == Qt.Key_O:
            self.ao_on = not self.ao_on
        elif key == Qt.Key_N:
            self.show_normals = not self.show_normals
        elif key == Qt.Key_I:
            self.show_iterations = not self.show_iterations
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.smooth_k = min(1.5, self.smooth_k + 0.05)
        elif key == Qt.Key_Minus:
            self.smooth_k = max(0.0, self.smooth_k - 0.05)
        elif key == Qt.Key_Space:
            if not self.paused:
                self._paused_time = self._elapsed_time()
            else:
                self._clock_start = time.perf_counter() - self._paused_time
            self.paused = not self.paused
        else:
            super().keyPressEvent(event)
            return
        self.update()


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by the Qt event loop."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
