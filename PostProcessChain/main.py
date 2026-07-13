#!/usr/bin/env -S uv run --script
"""
HDR post-processing chain: bright pass, separable Gaussian bloom and
tonemapping (OpenGL).

A grid, a teapot and five emissive spheres (colours with components > 1.0,
e.g. (8, 4, 0.5)) are rendered into an RGBA16F HDR framebuffer, then a
four-pass post chain -- built on the same FBO/full-screen-triangle
machinery as ``OITransparency`` -- turns that HDR image into something an
8-bit display can show without every bright highlight just clipping to
flat white:

    1. scene      -> full-res RGBA16F FBO (grid + teapot lit N.L, spheres
                      emissive verbatim, no clamping, no gamma)
    2. bright pass -> half-res RGBA16F: max(colour - threshold, 0), sampling
                      the full-res scene texture with LINEAR filtering
    3. blur        -> separable 5-tap Gaussian, ping-ponged horizontal/
                      vertical between two half-res FBOs for `n_passes`
                      iterations
    4. tonemap     -> screen: scene + bloomStrength * bloom, * exposure,
                      then one of three operators (clamp / Reinhard / ACES
                      fitted), then gamma 2.2

    B            toggle bloom on/off
    T            cycle tonemap operator (none -> Reinhard -> ACES -> ...)
    E / Shift+E  increase / decrease exposure
    +/-          more / fewer blur passes (1..8)
    H            toggle split-screen: left half raw clamp, right half the
                 full chain -- the fastest way to see why any of this
                 matters at all
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit

Why HDR before tonemap: if the emissive spheres were clamped to [0,1] the
moment they were shaded (an ordinary 8-bit framebuffer), "bright" and
"extremely bright" would be indistinguishable -- both just (1,1,1) white,
and there would be nothing for a bright-pass threshold to find. Keeping the
scene buffer in RGBA16F lets values like (8, 4, 0.5) survive all the way to
the final pass, where the tonemap operator makes the *choice* of how to
compress that range back into [0,1] -- explicit and swappable with `T`,
instead of implicit and irreversible.
"""

import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from FrameBufferObject import FrameBufferObject
from ncca.ngl import Mat3, Mat4, Prims, Vec3, logger, look_at, perspective
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
from TextureTypes import (
    GLAttachment,
    GLTextureDataType,
    GLTextureDepthFormats,
    GLTextureFormat,
    GLTextureInternalFormat,
    GLTextureMagFilter,
    GLTextureMinFilter,
    GLTextureWrap,
)

SCENE_SHADER = "Scene"
BRIGHT_SHADER = "BrightPass"
BLUR_SHADER = "Blur"
TONEMAP_SHADER = "Tonemap"

TONEMAP_NAMES = {1: "1: none (clamp)", 2: "2: Reinhard", 3: "3: ACES fitted"}

# (position, colour) for the emissive spheres -- colours deliberately have
# components > 1.0 so there is something for the bright pass to find.
EMISSIVE_SPHERES: list[tuple[Vec3, tuple[float, float, float]]] = [
    (Vec3(-3.0, 0.8, -1.0), (8.0, 4.0, 0.5)),
    (Vec3(-1.2, 1.4, 1.5), (0.5, 6.0, 8.0)),
    (Vec3(1.0, 0.6, -2.0), (7.0, 0.6, 6.0)),
    (Vec3(2.6, 1.6, 0.8), (9.0, 9.0, 2.0)),
    (Vec3(0.0, 2.6, -0.5), (1.0, 8.0, 2.0)),
]


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Scene -> bright pass -> ping-pong blur -> tonemap composite."""

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
        self.setTitle("Post Process Chain: HDR + Bloom + Tonemap (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.bloom_enabled: bool = True
        self.operator_mode: int = 3
        self.exposure: float = 1.0
        self.n_passes: int = 4
        self.split_screen: bool = False
        self.threshold: float = 1.0
        self.bloom_strength: float = 1.0

        # GL object handles created in initializeGL / _create_fbos
        self.scene_fbo = None
        self.bright_fbo = None
        self.blur_a_fbo = None
        self.blur_b_fbo = None
        self.textures: dict[str, int] = {}
        self.half_width: int = 1
        self.half_height: int = 1

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.02, 0.02, 0.03, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.view = look_at(
            Vec3(0.0, 2.2, 8.0), Vec3(0.0, 0.8, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            SCENE_SHADER,
            str(shader_dir / "SceneVertex.glsl"),
            str(shader_dir / "SceneFragment.glsl"),
        )
        ok = (
            ShaderLib.load_shader(
                BRIGHT_SHADER,
                str(shader_dir / "CompositeVertex.glsl"),
                str(shader_dir / "BrightPassFragment.glsl"),
            )
            and ok
        )
        ok = (
            ShaderLib.load_shader(
                BLUR_SHADER,
                str(shader_dir / "CompositeVertex.glsl"),
                str(shader_dir / "BlurFragment.glsl"),
            )
            and ok
        )
        ok = (
            ShaderLib.load_shader(
                TONEMAP_SHADER,
                str(shader_dir / "CompositeVertex.glsl"),
                str(shader_dir / "TonemapFragment.glsl"),
            )
            and ok
        )
        if not ok:
            print("error loading shaders")
            self.close()

        ShaderLib.use(BRIGHT_SHADER)
        ShaderLib.set_uniform("sceneTex", 0)
        ShaderLib.use(BLUR_SHADER)
        ShaderLib.set_uniform("image", 0)
        ShaderLib.use(TONEMAP_SHADER)
        ShaderLib.set_uniform("sceneTex", 0)
        ShaderLib.set_uniform("bloomTex", 1)

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 14.0, 14.0, 28)
        Primitives.create(Prims.SPHERE, "emissiveSphere", 0.45, 24)
        # empty VAO for the full-screen post-process triangles (core
        # profile requires a VAO to be bound even with no attributes)
        self.screen_vao = gl.glGenVertexArrays(1)

        FrameBufferObject.set_default_fbo(self.defaultFramebufferObject())
        self._create_fbos(self.window_width, self.window_height)
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    # ------------------------------------------------------------------
    # FBOs
    # ------------------------------------------------------------------
    def _add_bloom_target(self, fbo: FrameBufferObject, name: str) -> None:
        """Half/full-res single RGBA16F colour attachment, LINEAR filtered
        (bright pass and blur both sample these at a different res)."""
        fbo.add_colour_attachment(
            name,
            GLAttachment._0,
            GLTextureFormat.RGBA,
            GLTextureInternalFormat.RGBA16F,
            GLTextureDataType.FLOAT,
            GLTextureMinFilter.LINEAR,
            GLTextureMagFilter.LINEAR,
            GLTextureWrap.CLAMP_TO_EDGE,
            GLTextureWrap.CLAMP_TO_EDGE,
        )

    def _create_fbos(self, width: int, height: int) -> None:
        """(Re)build all four post-chain FBOs at this size -- called from
        initializeGL and every resize (the OITransparency
        _create_fbos/_delete_fbos pattern)."""
        self._delete_fbos()
        self.half_width = max(1, width // 2)
        self.half_height = max(1, height // 2)

        # pass 1: full-res HDR scene colour + depth. LINEAR filtering
        # because the bright pass samples this texture at half res.
        self.scene_fbo = FrameBufferObject.create(width, height)
        with self.scene_fbo:
            self._add_bloom_target(self.scene_fbo, "scene")
            self.scene_fbo.add_depth_buffer(
                GLTextureDepthFormats.DEPTH_COMPONENT24,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
            )
            if not self.scene_fbo.is_complete():
                logger.error("scene FBO incomplete")
        self.textures["scene"] = self.scene_fbo.get_texture_id("scene")

        # pass 2: half-res bright pass target
        self.bright_fbo = FrameBufferObject.create(self.half_width, self.half_height)
        with self.bright_fbo:
            self._add_bloom_target(self.bright_fbo, "bright")
        self.textures["bright"] = self.bright_fbo.get_texture_id("bright")

        # pass 3: two half-res ping-pong blur targets
        self.blur_a_fbo = FrameBufferObject.create(self.half_width, self.half_height)
        with self.blur_a_fbo:
            self._add_bloom_target(self.blur_a_fbo, "blur_a")
        self.textures["blur_a"] = self.blur_a_fbo.get_texture_id("blur_a")

        self.blur_b_fbo = FrameBufferObject.create(self.half_width, self.half_height)
        with self.blur_b_fbo:
            self._add_bloom_target(self.blur_b_fbo, "blur_b")
        self.textures["blur_b"] = self.blur_b_fbo.get_texture_id("blur_b")

    def _delete_fbos(self) -> None:
        # Dropping the references lets FrameBufferObject.__del__ clean up
        # the underlying framebuffer + texture GL objects.
        self.scene_fbo = None
        self.bright_fbo = None
        self.blur_a_fbo = None
        self.blur_b_fbo = None
        self.textures.clear()

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

    def _draw_full_screen_triangle(self) -> None:
        gl.glBindVertexArray(self.screen_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        global_tx = self.scene_global_tx()

        self._scene_pass(global_tx)
        bloom_tex = self.textures["scene"]  # fallback if bloom disabled
        if self.bloom_enabled:
            self._bright_pass()
            bloom_tex = self._blur_passes()
        self._tonemap_pass(bloom_tex)

        self._draw_hud()

    def _scene_pass(self, global_tx: Mat4) -> None:
        """Pass 1: grid + teapot (lit) + emissive spheres into the HDR FBO."""
        self.scene_fbo.bind()
        self.scene_fbo.set_viewport()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.02, 0.02, 0.03, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.COLOUR)
        grid_tx = Mat4().translate(0.0, -0.5, 0.0)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx @ grid_tx)
        ShaderLib.set_uniform("Colour", 0.35, 0.35, 0.4, 1.0)
        Primitives.draw("grid")

        ShaderLib.use(SCENE_SHADER)
        ShaderLib.set_uniform("lightDir", Vec3(0.4, 0.8, 0.6))

        def load_matrices(model: Mat4) -> None:
            mv = self.view @ global_tx @ model
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(mv).inverse().transposed()
            )

        teapot_tx = Mat4().translate(0.0, 0.0, 0.0)
        load_matrices(teapot_tx)
        ShaderLib.set_uniform("emissive", False)
        ShaderLib.set_uniform("Colour", 0.7, 0.7, 0.72)
        Primitives.draw("teapot")

        ShaderLib.set_uniform("emissive", True)
        for position, colour in EMISSIVE_SPHERES:
            load_matrices(Mat4().translate(position.x, position.y, position.z))
            ShaderLib.set_uniform("Colour", *colour)
            Primitives.draw("emissiveSphere")

    def _bright_pass(self) -> None:
        """Pass 2: half-res max(colour - threshold, 0)."""
        self.bright_fbo.bind()
        self.bright_fbo.set_viewport()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        ShaderLib.use(BRIGHT_SHADER)
        ShaderLib.set_uniform("threshold", self.threshold)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["scene"])
        self._draw_full_screen_triangle()

    def _blur_passes(self) -> int:
        """Pass 3: separable Gaussian, ping-ponged H/V between blur_a and
        blur_b for self.n_passes iterations. Returns the final bloom
        texture id (always blur_b, since every iteration ends there)."""
        self.blur_a_fbo.set_viewport()
        texel_size = (1.0 / self.half_width, 1.0 / self.half_height)
        ShaderLib.use(BLUR_SHADER)
        ShaderLib.set_uniform("texelSize", *texel_size)

        source_tex = self.textures["bright"]
        for _ in range(self.n_passes):
            # horizontal: source -> blur_a
            self.blur_a_fbo.bind()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            ShaderLib.set_uniform("horizontal", True)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, source_tex)
            self._draw_full_screen_triangle()

            # vertical: blur_a -> blur_b
            self.blur_b_fbo.bind()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            ShaderLib.set_uniform("horizontal", False)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["blur_a"])
            self._draw_full_screen_triangle()

            source_tex = self.textures["blur_b"]
        return self.textures["blur_b"]

    def _tonemap_pass(self, bloom_tex: int) -> None:
        """Pass 4: scene + bloomStrength*bloom, exposure, operator, gamma."""
        self.scene_fbo.unbind()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        ShaderLib.use(TONEMAP_SHADER)
        ShaderLib.set_uniform("bloomEnabled", self.bloom_enabled)
        ShaderLib.set_uniform("bloomStrength", self.bloom_strength)
        ShaderLib.set_uniform("exposure", self.exposure)
        ShaderLib.set_uniform("operatorMode", self.operator_mode)
        ShaderLib.set_uniform("splitScreen", self.split_screen)
        ShaderLib.set_uniform("screenWidth", float(self.window_width))

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["scene"])
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, bloom_tex)

        self._draw_full_screen_triangle()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_hud(self) -> None:
        state = (
            f"[B] bloom {'ON ' if self.bloom_enabled else 'OFF'}  "
            f"[T] tonemap {TONEMAP_NAMES[self.operator_mode]}  "
            f"[E/Shift+E] exposure {self.exposure:.2f}  "
            f"[+/-] blur passes {self.n_passes}  "
            f"[H] split-screen {'ON' if self.split_screen else 'OFF'}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 60.0)
        if self.scene_fbo is not None:
            self.makeCurrent()
            self._create_fbos(self.window_width, self.window_height)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_B:
            self.bloom_enabled = not self.bloom_enabled
        elif key == Qt.Key_T:
            self.operator_mode = self.operator_mode % 3 + 1
        elif key == Qt.Key_E and shift:
            self.exposure = max(0.1, self.exposure - 0.1)
        elif key == Qt.Key_E:
            self.exposure = min(4.0, self.exposure + 0.1)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.n_passes = min(8, self.n_passes + 1)
        elif key == Qt.Key_Minus:
            self.n_passes = max(1, self.n_passes - 1)
        elif key == Qt.Key_H:
            self.split_screen = not self.split_screen
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
