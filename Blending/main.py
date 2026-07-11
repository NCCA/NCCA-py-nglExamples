#!/usr/bin/env -S uv run --script
"""
Alpha blending and transparency (OpenGL).

Five transparent panels straddle an opaque teapot. Keyboard toggles let you
break the rendering in every classic way and see why each rule exists:

    B  toggle glEnable(GL_BLEND)          -- panels go opaque without it
    D  toggle depth *write* for the panels -- writing depth punches holes
    O  toggle back-to-front sorting        -- OVER is order dependent
    F  cycle the blend function            -- alpha / additive / premultiplied / multiply
    A/Z increase / decrease panel alpha
    Space reset the camera, Esc quit

The key teaching points:
    1. Transparency is not a material property, it is a *framebuffer
       operation*: the fragment shader just writes an alpha value and the
       blend state decides what it means.
    2. The OVER operator is order dependent, so transparent geometry must be
       drawn back to front (after all opaque geometry).
    3. Transparent geometry must be depth *tested* but not depth *written*,
       otherwise a near panel drawn first blocks a far panel drawn later.

See the OITransparency demo for what to do when sorting is not possible.
"""

import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from blend_scene import DEFAULT_ALPHA, PANEL_SIZE, PANELS, back_to_front
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

BLEND_SHADER = "BlendShader"

# (label, src factor, dst factor) presets cycled with F. The label is shown
# in the HUD so students can relate the picture to the glBlendFunc call.
BLEND_PRESETS = (
    (
        "OVER: SRC_ALPHA, ONE_MINUS_SRC_ALPHA",
        gl.GL_SRC_ALPHA,
        gl.GL_ONE_MINUS_SRC_ALPHA,
    ),
    ("ADDITIVE: SRC_ALPHA, ONE", gl.GL_SRC_ALPHA, gl.GL_ONE),
    ("PREMULTIPLIED: ONE, ONE_MINUS_SRC_ALPHA", gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA),
    ("MULTIPLY: DST_COLOR, ZERO", gl.GL_DST_COLOR, gl.GL_ZERO),
)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Transparent panel scene with togglable blend / depth / sort state."""

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
        self.setTitle("Blending and Transparency (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.blend_enabled: bool = True
        self.depth_write: bool = False
        self.sort_panels: bool = True
        self.preset: int = 0
        self.alpha: float = DEFAULT_ALPHA

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 1.5, 7.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            BLEND_SHADER,
            str(shader_dir / "BlendVertex.glsl"),
            str(shader_dir / "BlendFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use(BLEND_SHADER)
        ShaderLib.set_uniform("lightDir", Vec3(0.5, 1.0, 0.8))

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 12.0, 12.0, 24)
        # a unit quad facing +y; panels rotate it upright with their model matrix
        Primitives.create(
            Prims.TRIANGLE_PLANE, "panel", PANEL_SIZE, PANEL_SIZE, 1, 1, Vec3(0, 1, 0)
        )
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

    def _panel_matrix(self, panel) -> Mat4:
        """Model matrix standing the +y quad upright at the panel position."""
        t = Transform()
        t.set_position(*panel.position)
        t.set_rotation(90.0, 0.0, 0.0)
        return t.matrix()

    def _load_matrices(self, model: Mat4, global_tx: Mat4) -> None:
        MV = self.view @ global_tx @ model
        ShaderLib.set_uniform("MVP", self.project @ MV)
        normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_tx = self.scene_global_tx()
        tx = Transform()
        tx.set_position(0.0, -0.5, 0.0)

        # ---- 1. opaque pass: all solid geometry first, depth write on ----
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx @ tx.matrix())
        ShaderLib.set_uniform("Colour", 0.6, 0.6, 0.6, 1.0)
        Primitives.draw("grid")

        ShaderLib.use(BLEND_SHADER)
        ShaderLib.set_uniform("Colour", 0.85, 0.8, 0.75, 1.0)
        self._load_matrices(Mat4(), global_tx)
        Primitives.draw("teapot")

        # ---- 2. transparent pass ----
        if self.blend_enabled:
            gl.glEnable(gl.GL_BLEND)
            _, src, dst = BLEND_PRESETS[self.preset]
            gl.glBlendFunc(src, dst)
        # transparent geometry is depth *tested* against the opaque scene but
        # (normally) not depth *written*, so panels never occlude each other
        gl.glDepthMask(gl.GL_TRUE if self.depth_write else gl.GL_FALSE)

        matrices = [self._panel_matrix(p) for p in PANELS]
        order = range(len(PANELS))
        if self.sort_panels:
            model_views = [(self.view @ global_tx @ m).to_numpy() for m in matrices]
            order = back_to_front(model_views)
        for i in order:
            ShaderLib.set_uniform("Colour", *PANELS[i].colour, self.alpha)
            self._load_matrices(matrices[i], global_tx)
            Primitives.draw("panel")

        # ---- 3. restore state so the next frame starts clean ----
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        self._draw_hud()

    def _draw_hud(self) -> None:
        preset_name = BLEND_PRESETS[self.preset][0]
        state = (
            f"[B]lend {'ON ' if self.blend_enabled else 'OFF'}  "
            f"[D]epth write {'ON ' if self.depth_write else 'OFF'}  "
            f"[O]rder {'sorted back-to-front' if self.sort_panels else 'scene order'}  "
            f"alpha [A/Z] {self.alpha:.2f}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))
        Text.render_text(
            "Arial", 10, 45, f"[F] glBlendFunc {preset_name}", Vec3(1.0, 1.0, 1.0)
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_B:
            self.blend_enabled = not self.blend_enabled
        elif key == Qt.Key_D:
            self.depth_write = not self.depth_write
        elif key == Qt.Key_O:
            self.sort_panels = not self.sort_panels
        elif key == Qt.Key_F:
            self.preset = (self.preset + 1) % len(BLEND_PRESETS)
        elif key == Qt.Key_A:
            self.alpha = min(1.0, self.alpha + 0.05)
        elif key == Qt.Key_Z:
            self.alpha = max(0.05, self.alpha - 0.05)
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

    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
