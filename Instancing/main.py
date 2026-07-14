#!/usr/bin/env -S uv run --script
"""
GPU instancing vs a Python draw-call loop (OpenGL).

The same field of N cubes drawn two ways with the *same* shader and the
*same* per-instance data:

    instanced -- one glDrawArraysInstanced call, all N cubes read their
                 offset/scale/colour from a second, per-instance VBO.
    naive     -- a Python for loop of N glDrawArrays calls, each cube's
                 offset/scale/colour pushed as a uniform.

Both draw the identical geometry from the identical layout, so the only
thing that changes between modes is *how many draw calls Python issues*.
The HUD frame-time readout is the point of the whole demo: watch it jump
by 1-2 orders of magnitude when you flip from instanced to naive at
N=4096, because each glDrawArrays call and each glUniform4f call round
trips through Python and the driver.

Controls:
    I    toggle instanced / naive draw
    +/-  double / halve the instance count N (clamped 1..65536)
    Space reset the camera, Esc quit
"""

import argparse
import ctypes
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import OpenGL.GL as gl
from instance_layout import cube, golden_spiral
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib, Text
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

INSTANCE_SHADER = "InstanceShader"
DEFAULT_N = 4096
MIN_N = 1
MAX_N = 65536
FIELD_RADIUS = 8.0
FRAME_HISTORY = 30


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """A field of cubes drawn either instanced or via a naive draw-call loop."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, -12),
        )
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Instancing: one draw call vs N (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.instanced: bool = True
        self.n: int = DEFAULT_N
        self.field_rotation: float = 0.0
        self.frame_times: deque = deque(maxlen=FRAME_HISTORY)

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.12, 0.12, 0.14, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 6.0, 16.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            INSTANCE_SHADER,
            str(shader_dir / "InstanceVertex.glsl"),
            str(shader_dir / "InstanceFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use(INSTANCE_SHADER)
        ShaderLib.set_uniform("lightDir", Vec3(0.4, 1.0, 0.6))

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

        self._build_cube_vao()
        self._build_instance_data()
        self.startTimer(16)

    def _build_cube_vao(self) -> None:
        """One VBO of cube verts (locations 0, 1), shared by every draw."""
        cube_data = cube(1.0).reshape(-1, 6)
        self.cube_vertex_count = cube_data.shape[0]
        stride = 6 * 4  # 6 floats per vertex

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.cube_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.cube_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, cube_data.nbytes, cube_data.ravel(), gl.GL_STATIC_DRAW
        )
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4)
        )

        # Second VBO: per-instance offset+scale (loc 3) and colour (loc 4).
        # glVertexAttribDivisor(loc, 1) means "advance this attribute once
        # per *instance* rather than once per vertex" -- the mechanism that
        # lets one draw call fan out into N differently placed cubes.
        # Divisor state lives on the currently bound VAO, so binding self.vao
        # first (above) keeps it from leaking into any other VAO in the app.
        self.instance_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        instance_stride = 8 * 4  # offset.xyz, scale, colour.rgba
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(
            3, 4, gl.GL_FLOAT, gl.GL_FALSE, instance_stride, ctypes.c_void_p(0)
        )
        gl.glVertexAttribDivisor(3, 1)
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(
            4, 4, gl.GL_FLOAT, gl.GL_FALSE, instance_stride, ctypes.c_void_p(4 * 4)
        )
        gl.glVertexAttribDivisor(4, 1)

        gl.glBindVertexArray(0)

    def _build_instance_data(self) -> None:
        """(Re)generate the per-instance layout and upload it to the GPU."""
        self.instance_data = golden_spiral(self.n, radius=FIELD_RADIUS)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.instance_data.nbytes,
            self.instance_data.ravel(),
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face + self.field_rotation)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        start = time.perf_counter()
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        global_tx = self.scene_global_tx()
        MV = self.view @ global_tx
        ShaderLib.use(INSTANCE_SHADER)
        ShaderLib.set_uniform("MVP", self.project @ MV)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(MV).inverse().transposed())

        gl.glBindVertexArray(self.vao)
        if self.instanced:
            ShaderLib.set_uniform("instanced", 1)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.cube_vertex_count, self.n)
        else:
            ShaderLib.set_uniform("instanced", 0)
            # The teaching point: identical geometry, identical per-instance
            # data, but pushed through Python/the driver one draw call (and
            # one pair of uniform updates) at a time instead of one call
            # for the whole field.
            for offset_scale, colour in zip(
                self.instance_data[:, 0:4], self.instance_data[:, 4:8]
            ):
                ShaderLib.set_uniform("uOffsetScale", *offset_scale.tolist())
                ShaderLib.set_uniform("uColour", *colour.tolist())
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.cube_vertex_count)
        gl.glBindVertexArray(0)

        self._draw_hud()
        self.frame_times.append(time.perf_counter() - start)

    def _draw_hud(self) -> None:
        avg_ms = (
            1000.0 * sum(self.frame_times) / len(self.frame_times)
            if self.frame_times
            else 0.0
        )
        mode = "INSTANCED (1 draw call)" if self.instanced else "NAIVE (N draw calls)"
        Text.render_text("Arial", 10, 20, f"[I] mode: {mode}", Vec3(1.0, 1.0, 1.0))
        Text.render_text(
            "Arial",
            10,
            45,
            f"[+/-] N = {self.n}   avg frame time = {avg_ms:.2f} ms",
            Vec3(1.0, 1.0, 1.0),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)
        Text.set_screen_size(w, h)

    def timerEvent(self, event) -> None:
        self.field_rotation += 0.3
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _set_n(self, n: int) -> None:
        self.n = max(MIN_N, min(MAX_N, n))
        self._build_instance_data()
        self.frame_times.clear()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_I:
            self.instanced = not self.instanced
            self.frame_times.clear()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._set_n(self.n * 2)
        elif key == Qt.Key_Minus:
            self._set_n(self.n // 2)
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, -12)
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by the Qt event loop."""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
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
