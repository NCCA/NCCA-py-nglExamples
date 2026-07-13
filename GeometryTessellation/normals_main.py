#!/usr/bin/env -S uv run --script
"""
Geometry-shader normal visualiser (OpenGL).

A teapot is drawn twice every frame: once with ordinary diffuse shading,
then a second pass with the *same* vertex data but a different program --
one whose geometry shader turns each incoming triangle into 1-3 short
line segments along its normal(s), instead of a filled triangle.

    +/-          increase / decrease the visualised normal length
    F            toggle vertex-normal mode vs. face-normal mode
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit

Teaching points:
    1. A geometry shader runs once per *primitive*, after the vertex shader
       and before rasterisation, and can emit a different primitive type
       and a different vertex count than it received --
       `layout(triangles) in; layout(line_strip, max_vertices = 6) out;`
       turns 3 triangle vertices into up to 3 independent line segments.
    2. Per-vertex ("smooth") normals are the *interpolated* average of the
       normals of every face touching a vertex; per-face ("faceted") normals
       are a single normal for the whole triangle. `F` swaps between
       emitting one line per input vertex (mode 1) and averaging the
       triangle down to a single centre + normal (mode 2) -- the same
       geometry, viewed through the two different normal conventions.
    3. This technique is a debugging tool, not a production one: it costs
       a second full draw call and a geometry shader, both of which real
       pipelines avoid where possible (see the WebGPU-absence note in the
       README -- this whole stage doesn't exist outside desktop GL/Vulkan).
"""

import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

NORMAL_LINES_SHADER = "NormalLines"

MIN_NORMAL_LENGTH = 0.02
MAX_NORMAL_LENGTH = 1.5
NORMAL_LENGTH_STEP = 0.02


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Teapot drawn with diffuse shading, then with a normal-line overlay."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Geometry Shader Normal Visualiser (OpenGL)")

        self.normal_length: float = 0.25
        self.face_mode: bool = False

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.35, 0.38, 0.42, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 1.5, 6.0), Vec3(0.0, 0.5, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            NORMAL_LINES_SHADER,
            str(shader_dir / "NormalLinesVertex.glsl"),
            str(shader_dir / "NormalLinesFragment.glsl"),
            geo=str(shader_dir / "NormalLinesGeometry.glsl"),
        )
        if not ok:
            print("error loading shaders")
            self.close()

        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        global_tx = self.scene_global_tx()
        model = Mat4()
        mv = self.view @ global_tx @ model
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()

        # ---- pass 1: ordinary diffuse-shaded teapot ----
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("Colour", 0.75, 0.55, 0.35, 1.0)
        ShaderLib.set_uniform("lightPos", Vec3(2.0, 3.0, 4.0))
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.draw("teapot")

        # ---- pass 2: same geometry, geometry-shader normal overlay ----
        ShaderLib.use(NORMAL_LINES_SHADER)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("project", self.project)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("normalLength", self.normal_length)
        ShaderLib.set_uniform("faceMode", self.face_mode)
        ShaderLib.set_uniform("lineColour", Vec3(1.0, 0.9, 0.1))
        Primitives.draw("teapot")

        self._draw_hud()

    def _draw_hud(self) -> None:
        mode = "face (flat)" if self.face_mode else "vertex (smooth)"
        state = (
            f"[F] normal mode: {mode}   [+/-] normal length: {self.normal_length:.2f}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 60.0)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_F:
            self.face_mode = not self.face_mode
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.normal_length = min(
                MAX_NORMAL_LENGTH, self.normal_length + NORMAL_LENGTH_STEP
            )
        elif key == Qt.Key_Minus:
            self.normal_length = max(
                MIN_NORMAL_LENGTH, self.normal_length - NORMAL_LENGTH_STEP
            )
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
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
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv
    if "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
