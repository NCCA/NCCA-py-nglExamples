#!/usr/bin/env -S uv run --script
"""
Billboards: camera-facing quads (OpenGL).

Roughly thirty textured sprites are scattered around a teapot (the depth
reference -- it never billboards, so you always have something solid to
judge distance against). Press M to cycle three ways of orienting the
sprites and watch the same 30 quads go from "obviously wrong" to "always
correct":

    M  cycle billboard mode: 1 fixed world-space / 2 cylindrical / 3 spherical
    B  toggle alpha blending (sorted back-to-front) vs. alpha-tested cutout
    Space reset the camera, Esc quit

The three modes:
    1. Fixed world-space -- the quad's right/up vectors never change. Orbit
       the scene and, at some angles, you are looking at the sprite edge-on:
       it thins to a sliver and vanishes. This is what "just drawing a
       textured quad" gets you without billboarding at all.
    2. Cylindrical -- up is locked to world +y (trees, lampposts, anything
       that should stay upright), right is rebuilt from the camera direction
       every frame. Robust to orbiting sideways; drag vertically to pitch
       the view and it visibly tips, because a cylindrical billboard has no
       answer for "which way is sideways" when you're looking down its
       locked axis.
    3. Spherical -- both right and up are rebuilt from the camera every
       frame. Always face-on, from any angle. This is the one particle
       systems and impostors actually use.

See billboard_maths.py for the vector maths behind all three, and the
README for the worked-through derivation.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from billboard_maths import back_to_front, cylindrical_basis, spherical_basis
from ncca.ngl import Mat3, Mat4, Prims, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
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

BILLBOARD_SHADER = "BillboardShader"

# Fixed seed so the scatter is reproducible run to run -- this demo is about
# the billboarding maths, not about randomness, so nothing here should shift
# between two people comparing screenshots.
_SCATTER_SEED = 42
_BILLBOARD_COUNT = 30
_BILLBOARD_SIZE = 0.7
_SPRITE_TEXTURE_SIZE = 64

MODE_NAMES = ("1: fixed world-space", "2: cylindrical (up locked)", "3: spherical")


def _make_scatter_positions(count: int, seed: int) -> np.ndarray:
    """Fixed, reproducible scatter of billboard centres around the origin.

    Kept out of a fixed radius of the teapot so the depth reference stays
    readable, spread across a range of heights so the "breaks when
    orbiting" effect in mode 1 is visible from more than one angle.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = rng.uniform(2.0, 6.0, count)
    height = rng.uniform(0.2, 3.2, count)
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    return np.stack([x, height, z], axis=1).astype(np.float64)


def _make_sprite_texture(size: int) -> np.ndarray:
    """Procedural soft radial-gradient RGBA sprite (a glow dot).

    No binary asset: the brief for this demo asks for the texture to be
    generated in numpy at startup and uploaded with glTexImage2D directly.
    Alpha falls off as a squared cosine-like falloff from the centre so the
    edge fades rather than hard-cuts.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    centre = (size - 1) / 2.0
    dist = np.sqrt((xs - centre) ** 2 + (ys - centre) ** 2) / centre
    alpha = np.clip(1.0 - dist, 0.0, 1.0) ** 2

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # warm glow: full red
    rgba[..., 1] = 220  # slightly less green
    rgba[..., 2] = 140  # much less blue
    rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
    return rgba


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Grid + teapot + a field of billboarded sprites, three modes."""

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
        self.setTitle("Billboards (OpenGL)")

        self.mode: int = 3  # 1=fixed, 2=cylindrical, 3=spherical -- start correct
        self.blend_enabled: bool = False
        self.scatter = _make_scatter_positions(_BILLBOARD_COUNT, _SCATTER_SEED)

        self.billboard_vao = None
        self.sprite_texture: int | None = None

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.35, 0.35, 0.38, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 2.5, 9.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            BILLBOARD_SHADER,
            str(shader_dir / "BillboardVertex.glsl"),
            str(shader_dir / "BillboardFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 16.0, 16.0, 32)

        self.billboard_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        self._create_sprite_texture()

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    def _create_sprite_texture(self) -> None:
        data = _make_sprite_texture(_SPRITE_TEXTURE_SIZE)
        self.sprite_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.sprite_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            _SPRITE_TEXTURE_SIZE,
            _SPRITE_TEXTURE_SIZE,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

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

    def _billboard_basis(
        self, model_view_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Right/up for the current mode, in local (pre-global_tx) space.

        Modes 2 and 3 are handed the *combined* model-view matrix (view
        rotated through the same global_tx that also carries the scatter
        positions and the teapot), not just the raw camera view. That
        combined matrix is exactly what spherical_basis/cylindrical_basis
        need to hand back local-space vectors that -- once the mouse-drag
        rotation is reapplied on the GPU -- land in the true camera-facing
        direction regardless of how far the scene has been orbited. See the
        README for the derivation; it hinges on global_tx being a pure
        rotation (orthonormal), so the correction cancels exactly.
        """
        if self.mode == 1:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        if self.mode == 2:
            return cylindrical_basis(model_view_np)
        return spherical_basis(model_view_np)

    def _build_billboard_vertices(self, global_tx: Mat4) -> np.ndarray:
        """Interleaved (x, y, z, u, v) vertex data for every billboard, two
        triangles each, ordered back-to-front when blending is on."""
        model_view_np = (self.view @ global_tx).to_numpy().astype(np.float64)
        right, up = self._billboard_basis(model_view_np)
        half = _BILLBOARD_SIZE * 0.5

        order = range(len(self.scatter))
        if self.blend_enabled:
            order = back_to_front(self.scatter, model_view_np)

        verts = np.empty((len(self.scatter) * 6, 5), dtype=np.float32)
        row = 0
        for i in order:
            centre = self.scatter[i]
            bl = centre - right * half - up * half
            br = centre + right * half - up * half
            tr = centre + right * half + up * half
            tl = centre - right * half + up * half
            for corner, uv in (
                (bl, (0.0, 0.0)),
                (br, (1.0, 0.0)),
                (tr, (1.0, 1.0)),
                (bl, (0.0, 0.0)),
                (tr, (1.0, 1.0)),
                (tl, (0.0, 1.0)),
            ):
                verts[row, :3] = corner
                verts[row, 3:] = uv
                row += 1
        return verts

    def _load_scene_matrices(self, model: Mat4, global_tx: Mat4) -> None:
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

        # ---- 1. opaque pass: grid + teapot, depth write on ----
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        ShaderLib.set_uniform("Colour", 0.6, 0.6, 0.6, 1.0)
        Primitives.draw("grid")

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 0.85, 0.8, 0.75, 1.0)
        ShaderLib.set_uniform("lightPos", Vec3(2.0, 4.0, 3.0))
        # lightDiffuse is a vec4 in DefaultShader.DIFFUSE -- it must be set with
        # four components (a Vec3 here emits glUniform3f and GL rejects it with
        # GL_INVALID_OPERATION, aborting the whole frame).
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        self._load_scene_matrices(Mat4(), global_tx)
        Primitives.draw("teapot")

        # ---- 2. billboards ----
        ShaderLib.use(BILLBOARD_SHADER)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.sprite_texture)
        ShaderLib.set_uniform("spriteTex", 0)

        if self.blend_enabled:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)

        verts = self._build_billboard_vertices(global_tx)
        stride = verts.shape[1] * 4
        with self.billboard_vao as vao:
            vao.set_data(
                VertexData(
                    data=verts.reshape(-1), size=verts.shape[0], mode=gl.GL_DYNAMIC_DRAW
                )
            )
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)
            vao.set_vertex_attribute_pointer(1, 2, gl.GL_FLOAT, stride, 12)
            vao.draw()

        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        self._draw_hud()

    def _draw_hud(self) -> None:
        state = (
            f"[M]ode {MODE_NAMES[self.mode - 1]}   "
            f"[B]lend {'ON (sorted back-to-front)' if self.blend_enabled else 'OFF (alpha-tested cutout)'}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))

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
        elif key == Qt.Key_M:
            self.mode = self.mode % 3 + 1
        elif key == Qt.Key_B:
            self.blend_enabled = not self.blend_enabled
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
