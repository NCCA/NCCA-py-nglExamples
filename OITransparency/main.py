#!/usr/bin/env -S uv run --script
"""
Weighted blended order-independent transparency (OpenGL).

Two transparent panels intersect in an X through an opaque teapot -- a case
no per-object sort can draw correctly -- alongside ordinary front/back
panels. Three render modes show the problem and the fix:

    1  naive alpha blending in scene order (wrong where order is wrong)
    2  per-object back-to-front sorting (fixes the parallel panels, still
       wrong along the intersection of the X pair)
    3  weighted blended OIT (McGuire & Bavoil 2013): order independent,
       no sorting at all

    A/Z increase / decrease panel alpha, Space reset camera, Esc quit

How mode 3 works (three passes):
    1. opaque pass   -> opaque colour texture + depth texture
    2. transparent   -> two MRT accumulation targets, depth *test* against
       the opaque depth but no depth write, order irrelevant:
           accum  (RGBA16F, blend ONE, ONE)              a weighted SUM
           reveal (R16F, blend ZERO, ONE_MINUS_SRC_COLOR) a PRODUCT of (1-a)
    3. composite     -> full-screen triangle resolves the average and lerps
       it over the opaque colour by the total transmittance

Sums and products are commutative, which is why the result cannot depend on
draw order. See oit_common.py for a testable CPU reference implementation.

Reference: http://jcgt.org/published/0002/02/09/
"""

import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from oit_common import DEFAULT_ALPHA, PANEL_SIZE, PANELS, back_to_front
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

TRANSPARENT_SHADER = "TransparentShader"
OIT_ACCUM_SHADER = "OITAccumShader"
COMPOSITE_SHADER = "CompositeShader"

MODE_NAMES = {
    1: "1: naive alpha blend (scene order)",
    2: "2: sorted back-to-front (per object)",
    3: "3: weighted blended OIT (no sorting)",
}


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Three-pass weighted blended OIT with naive/sorted comparison modes."""

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
        self.setTitle("Weighted Blended OIT (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.mode: int = 3
        self.alpha: float = DEFAULT_ALPHA

        # GL object handles created in initializeGL / _create_fbos
        self.opaque_fbo = None
        self.accum_fbo = None
        self.textures: dict[str, int] = {}

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.view = look_at(
            Vec3(0.0, 1.8, 8.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            TRANSPARENT_SHADER,
            str(shader_dir / "TransparentVertex.glsl"),
            str(shader_dir / "TransparentFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            OIT_ACCUM_SHADER,
            str(shader_dir / "TransparentVertex.glsl"),
            str(shader_dir / "OITAccumFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            COMPOSITE_SHADER,
            str(shader_dir / "CompositeVertex.glsl"),
            str(shader_dir / "CompositeFragment.glsl"),
        )
        if not ok:
            print("error loading shaders")
            self.close()
        for shader in (TRANSPARENT_SHADER, OIT_ACCUM_SHADER):
            ShaderLib.use(shader)
            ShaderLib.set_uniform("lightDir", Vec3(0.5, 1.0, 0.8))
        ShaderLib.use(COMPOSITE_SHADER)
        ShaderLib.set_uniform("opaqueTex", 0)
        ShaderLib.set_uniform("accumTex", 1)
        ShaderLib.set_uniform("revealTex", 2)

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 12.0, 12.0, 24)
        Primitives.create(
            Prims.TRIANGLE_PLANE, "panel", PANEL_SIZE, PANEL_SIZE, 1, 1, Vec3(0, 1, 0)
        )
        # empty VAO for the full-screen composite triangle (core profile
        # requires a VAO to be bound even when no attributes are used)
        self.screen_vao = gl.glGenVertexArrays(1)

        self._create_fbos(self.window_width, self.window_height)
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    # ------------------------------------------------------------------
    # FBOs
    # ------------------------------------------------------------------
    def _create_texture(self, internal, fmt, dtype, width, height) -> int:
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, dtype, None
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def _create_fbos(self, width: int, height: int) -> None:
        """(Re)build the opaque and accumulation FBOs at this size."""
        self._delete_fbos()

        self.textures["opaque"] = self._create_texture(
            gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, width, height
        )
        self.textures["depth"] = self._create_texture(
            gl.GL_DEPTH_COMPONENT24,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_UNSIGNED_INT,
            width,
            height,
        )
        self.textures["accum"] = self._create_texture(
            gl.GL_RGBA16F, gl.GL_RGBA, gl.GL_FLOAT, width, height
        )
        self.textures["reveal"] = self._create_texture(
            gl.GL_R16F, gl.GL_RED, gl.GL_FLOAT, width, height
        )

        # opaque pass target: colour + depth
        self.opaque_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.opaque_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.textures["opaque"],
            0,
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_TEXTURE_2D,
            self.textures["depth"],
            0,
        )

        # accumulation pass target: two colour attachments sharing the SAME
        # depth texture, so transparent fragments are still tested against
        # (but never write) the opaque scene depth
        self.accum_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.accum_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.textures["accum"],
            0,
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT1,
            gl.GL_TEXTURE_2D,
            self.textures["reveal"],
            0,
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_TEXTURE_2D,
            self.textures["depth"],
            0,
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _delete_fbos(self) -> None:
        if self.opaque_fbo is not None:
            gl.glDeleteFramebuffers(2, [self.opaque_fbo, self.accum_fbo])
            gl.glDeleteTextures(len(self.textures), list(self.textures.values()))
            self.textures.clear()
            self.opaque_fbo = None
            self.accum_fbo = None

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
        """Stand the +y quad upright (rotate x), yaw it (rotate y), then
        translate. Composed explicitly so the rotation order is unambiguous:
        with MVP = P @ V @ M the rightmost transform applies first."""
        model = Mat4.rotate_y(panel.rotate_y) @ Mat4.rotate_x(90.0)
        model[3, 0], model[3, 1], model[3, 2] = panel.position
        return model

    def _load_matrices(self, shader: str, model: Mat4, global_tx: Mat4) -> None:
        ShaderLib.use(shader)
        MV = self.view @ global_tx @ model
        ShaderLib.set_uniform("MVP", self.project @ MV)
        ShaderLib.set_uniform("MV", MV)
        normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        global_tx = self.scene_global_tx()

        self._opaque_pass(global_tx)
        if self.mode == 3:
            self._oit_accum_pass(global_tx)
        else:
            self._blended_pass(global_tx, sort=self.mode == 2)
        self._composite_pass()

        Text.render_text("Arial", 10, 20, MODE_NAMES[self.mode], Vec3(1.0, 1.0, 1.0))
        Text.render_text(
            "Arial",
            10,
            45,
            f"keys 1/2/3 change mode   alpha [A/Z] {self.alpha:.2f}",
            Vec3(1.0, 1.0, 1.0),
        )

    def _opaque_pass(self, global_tx: Mat4) -> None:
        """Pass 1: all solid geometry into the opaque colour + depth FBO."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.opaque_fbo)
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.COLOUR)
        tx = Mat4().translate(0.0, -0.5, 0.0)

        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx @ tx)
        ShaderLib.set_uniform("Colour", 0.6, 0.6, 0.6, 1.0)
        Primitives.draw("grid")

        self._load_matrices(TRANSPARENT_SHADER, Mat4(), global_tx)
        ShaderLib.set_uniform("Colour", 0.85, 0.8, 0.75, 1.0)
        Primitives.draw("teapot")

    def _blended_pass(self, global_tx: Mat4, sort: bool) -> None:
        """Modes 1 and 2: blend the panels straight into the opaque buffer."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.opaque_fbo)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)

        matrices = [self._panel_matrix(p) for p in PANELS]
        order = range(len(PANELS))
        if sort:
            model_views = [(self.view @ global_tx @ m).to_numpy() for m in matrices]
            order = back_to_front(model_views)
        for i in order:
            self._load_matrices(TRANSPARENT_SHADER, matrices[i], global_tx)
            ShaderLib.set_uniform("Colour", *PANELS[i].colour, self.alpha)
            Primitives.draw("panel")

        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

    def _oit_accum_pass(self, global_tx: Mat4) -> None:
        """Mode 3, pass 2: accumulate all panels in ARBITRARY order."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.accum_fbo)
        gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])
        # clear only the colour attachments -- the depth texture still holds
        # the opaque scene depth from pass 1 and must be preserved
        gl.glClearBufferfv(gl.GL_COLOR, 0, (0.0, 0.0, 0.0, 0.0))
        gl.glClearBufferfv(gl.GL_COLOR, 1, (1.0, 1.0, 1.0, 1.0))

        gl.glEnable(gl.GL_BLEND)
        # per-attachment blend functions (core since GL 4.0):
        gl.glBlendFunci(0, gl.GL_ONE, gl.GL_ONE)  # accum: sum
        gl.glBlendFunci(1, gl.GL_ZERO, gl.GL_ONE_MINUS_SRC_COLOR)  # reveal: product
        gl.glDepthMask(gl.GL_FALSE)

        for panel in PANELS:  # note: scene order, no sorting anywhere
            self._load_matrices(OIT_ACCUM_SHADER, self._panel_matrix(panel), global_tx)
            ShaderLib.set_uniform("Colour", *panel.colour, self.alpha)
            Primitives.draw("panel")

        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

    def _composite_pass(self) -> None:
        """Final pass: full-screen triangle to the default framebuffer."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)

        ShaderLib.use(COMPOSITE_SHADER)
        ShaderLib.set_uniform("useOIT", self.mode == 3)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["opaque"])
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["accum"])
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["reveal"])

        gl.glBindVertexArray(self.screen_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        if self.opaque_fbo is not None:
            self.makeCurrent()
            self._create_fbos(self.window_width, self.window_height)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            self.mode = int(event.text())
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
