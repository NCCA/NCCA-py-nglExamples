#!/usr/bin/env -S uv run --script
"""
Classic two-pass shadow mapping (OpenGL).

This is the deliberate GL mirror of the ``WebGPUShadows`` demo -- same
scene, same PCF option, same artefact toggles -- so the two can be diffed
API-for-API to see what each backend makes you do explicitly.

    P            toggle PCF (3x3 percentage-closer filtering) vs a single tap
    B / Shift+B  increase / decrease the depth bias (HUD shows the value --
                 too small -> shadow acne, too large -> peter-panning)
    C            toggle front-face culling during the depth pass
    V            toggle the depth-map debug inset (top-right corner)
    L            pause / resume the light orbit
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit

Teaching points:
    1. Shadow mapping is just "render depth from the light, then compare":
       pass 1 renders the scene into a depth-only FBO from the light's
       ortho view-projection; pass 2 re-projects every fragment's world
       position through that same matrix and compares its depth against
       the stored value.
    2. Depth comparison is done *explicitly* in the fragment shader
       (texture(shadowMap, uv).r vs currentDepth - bias) rather than with
       sampler2DShadow/textureProj, which would hide the perspective
       divide and the [-1,1] -> [0,1] remap that make the technique work.
    3. Every visible shadow-mapping artefact traces back to one of: bias
       (acne vs peter-panning), culling in the depth pass (self-shadowing
       vs peter-panning), or the wrap mode at the light frustum edge
       (CLAMP_TO_BORDER with border 1.0 so geometry outside the light's
       view is lit, not shadowed).
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, ortho, perspective
from ncca.ngl.opengl import (
    AbstractVAO,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

DEPTH_SHADER = "ShadowDepth"
SHADE_SHADER = "Shade"
DEBUG_SHADER = "DebugQuad"

SHADOW_MAP_SIZE = 2048
LIGHT_ORTHO_EXTENT = 6.0
LIGHT_NEAR = 1.0
LIGHT_FAR = 20.0
LIGHT_RADIUS = 6.0
LIGHT_HEIGHT = 6.0
LIGHT_ORBIT_SPEED = 0.4  # radians / second, paused with L

BIAS_STEP = 0.0005
DEBUG_INSET_PX = 220


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Two-pass shadow-mapped scene with togglable bias / PCF / culling."""

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
        self.setTitle("Shadow Mapping (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.pcf_enabled: bool = True
        self.bias: float = 0.0025
        self.cull_front: bool = True
        self.show_inset: bool = True
        self.light_paused: bool = False
        self.light_angle: float = 0.7

        self.depth_texture: int = 0
        self.depth_fbo: int = 0
        self.debug_vao: AbstractVAO | None = None
        self.light_space: Mat4 = Mat4()
        self.light_pos: Vec3 = Vec3(0.0, LIGHT_HEIGHT, 0.0)

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.35, 0.38, 0.42, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 3.5, 8.0), Vec3(0.0, 0.5, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            DEPTH_SHADER,
            str(shader_dir / "ShadowDepthVertex.glsl"),
            str(shader_dir / "ShadowDepthFragment.glsl"),
        )
        ok = (
            ShaderLib.load_shader(
                SHADE_SHADER,
                str(shader_dir / "ShadeVertex.glsl"),
                str(shader_dir / "ShadeFragment.glsl"),
            )
            and ok
        )
        ok = (
            ShaderLib.load_shader(
                DEBUG_SHADER,
                str(shader_dir / "DebugQuadVertex.glsl"),
                str(shader_dir / "DebugQuadFragment.glsl"),
            )
            and ok
        )
        if not ok:
            print("error loading shaders")
            self.close()

        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

        self._build_scene()
        self._create_shadow_map()
        self._create_debug_quad()
        self._update_light()

    def _build_scene(self) -> None:
        """Floor + a handful of casters/receivers with varied heights, so
        the demo shows contact shadows, self-shadowing and a floating
        object (good for demonstrating peter-panning)."""
        from ncca.ngl import Prims

        Primitives.create(
            Prims.TRIANGLE_PLANE, "floor", 12.0, 12.0, 20, 20, Vec3(0, 1, 0)
        )
        Primitives.create(Prims.SPHERE, "ball", 0.6, 40)

        self.objects: list[tuple[str, Mat4, tuple[float, float, float, float]]] = [
            ("floor", Mat4(), (0.65, 0.65, 0.68, 1.0)),
            ("teapot", Mat4.translate(0.0, 0.05, -0.5), (0.85, 0.55, 0.25, 1.0)),
            ("cube", Mat4.translate(-2.2, 0.5, 0.8), (0.4, 0.65, 0.85, 1.0)),
            ("ball", Mat4.translate(1.8, 1.4, 1.0), (0.75, 0.3, 0.35, 1.0)),
        ]

    def _create_shadow_map(self) -> None:
        """2048^2 depth-only FBO: a GL_DEPTH_COMPONENT24 *texture* (not a
        renderbuffer) so it can be sampled in the shading pass, with no
        colour attachment at all."""
        self.depth_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT24,
            SHADOW_MAP_SIZE,
            SHADOW_MAP_SIZE,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER
        )
        # border colour 1.0 (== far plane / never in shadow): anything
        # sampled outside the light frustum reads as fully lit.
        gl.glTexParameterfv(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0]
        )
        # We read the depth texture manually with texture(...).r in the
        # fragment shader (not sampler2DShadow), so the comparison mode
        # must stay off, otherwise every read silently becomes a 0/1
        # comparison result instead of a raw depth value.
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_COMPARE_MODE, gl.GL_NONE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.depth_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_TEXTURE_2D,
            self.depth_texture,
            0,
        )
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            logger.error(f"Shadow map FBO incomplete: {status}")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _create_debug_quad(self) -> None:
        """A tiny NDC-space quad used to preview the depth texture in the
        corner of the screen; drawn with its own shader and a shrunk
        viewport, no MVP involved."""
        # x, y, u, v
        verts = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.debug_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self.debug_vao:
            self.debug_vao.set_data(VertexData(data=verts, size=6))
            stride = 4 * verts.itemsize
            self.debug_vao.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, stride, 0)
            self.debug_vao.set_vertex_attribute_pointer(
                1, 2, gl.GL_FLOAT, stride, 2 * verts.itemsize
            )

    # ------------------------------------------------------------------
    # per-frame light + camera update
    # ------------------------------------------------------------------
    def _update_light(self) -> None:
        x = LIGHT_RADIUS * math.cos(self.light_angle)
        z = LIGHT_RADIUS * math.sin(self.light_angle)
        self.light_pos = Vec3(x, LIGHT_HEIGHT, z)
        light_view = look_at(self.light_pos, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        light_project = ortho(
            -LIGHT_ORTHO_EXTENT,
            LIGHT_ORTHO_EXTENT,
            -LIGHT_ORTHO_EXTENT,
            LIGHT_ORTHO_EXTENT,
            LIGHT_NEAR,
            LIGHT_FAR,
        )
        # ncca.ngl.Mat4's __matmul__ makes "project @ view" read like
        # column-major maths (v @ (project @ view) applies view first,
        # then project) -- see shadow_maths.py for the equivalent done in
        # plain numpy, where the operand order is reversed.
        self.light_space = light_project @ light_view

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
        if not self.light_paused:
            self.light_angle += LIGHT_ORBIT_SPEED / 60.0
        self._update_light()

        global_tx = self.scene_global_tx()
        self._depth_pass(global_tx)
        self._shade_pass(global_tx)
        if self.show_inset:
            self._debug_pass()
        self._draw_hud()
        self.update()

    def _depth_pass(self, global_tx: Mat4) -> None:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_fbo)
        gl.glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        # Culling front faces during the depth pass writes the *back*
        # surface's depth into the shadow map, which fixes shadow acne on
        # the lit face at the cost of more peter-panning on thin geometry
        # -- flip [C] to see both artefacts live.
        if self.cull_front:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_FRONT)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        ShaderLib.use(DEPTH_SHADER)
        ShaderLib.set_uniform("lightSpaceMatrix", self.light_space)
        for name, model, _colour in self.objects:
            ShaderLib.set_uniform("M", global_tx @ model)
            Primitives.draw(name)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _shade_pass(self, global_tx: Mat4) -> None:
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(SHADE_SHADER)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        ShaderLib.set_uniform("shadowMap", 0)
        ShaderLib.set_uniform("lightSpaceMatrix", self.light_space)
        ShaderLib.set_uniform("lightDir", self.light_pos.normalized())
        ShaderLib.set_uniform("bias", self.bias)
        ShaderLib.set_uniform("pcfEnabled", self.pcf_enabled)

        for name, model, colour in self.objects:
            m = global_tx @ model
            mv = self.view @ m
            ShaderLib.set_uniform("M", m)
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            ShaderLib.set_uniform("Colour", *colour)
            Primitives.draw(name)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _debug_pass(self) -> None:
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(
            self.window_width - DEBUG_INSET_PX - 10,
            self.window_height - DEBUG_INSET_PX - 10,
            DEBUG_INSET_PX,
            DEBUG_INSET_PX,
        )
        ShaderLib.use(DEBUG_SHADER)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        ShaderLib.set_uniform("shadowMap", 0)
        with self.debug_vao:
            self.debug_vao.draw()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_hud(self) -> None:
        state = (
            f"[P] PCF {'ON ' if self.pcf_enabled else 'OFF'}  "
            f"[B/Shift+B] bias {self.bias:.4f}  "
            f"[C] cull-front(depth pass) {'ON ' if self.cull_front else 'OFF'}  "
            f"[V] inset {'ON ' if self.show_inset else 'OFF'}  "
            f"[L] light orbit {'PAUSED' if self.light_paused else 'running'}"
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
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_P:
            self.pcf_enabled = not self.pcf_enabled
        elif key == Qt.Key_B and shift:
            self.bias = max(0.0, self.bias - BIAS_STEP)
        elif key == Qt.Key_B:
            self.bias = min(0.02, self.bias + BIAS_STEP)
        elif key == Qt.Key_C:
            self.cull_front = not self.cull_front
        elif key == Qt.Key_V:
            self.show_inset = not self.show_inset
        elif key == Qt.Key_L:
            self.light_paused = not self.light_paused
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
