#!/usr/bin/env -S uv run --script
"""
Stencil buffer selection outlines (OpenGL).

A teapot, two cubes and a sphere sit on a grid; `Tab` cycles which one is
"selected" and a Maya-style orange fringe is drawn around it, entirely via
the stencil buffer -- no post-process pass, no second render target.

    Tab   cycle the selected object
    O     toggle the outline pass (see the extra draw call it costs)
    V     visualise the stencil buffer (tint every pixel where stencil == 1)
    +/-   grow / shrink the outline width (outlineScale)
    Space reset the camera, Esc quit

The two-pass recipe, per frame:
    1. Clear colour, depth *and* stencil.
    2. Draw every object normally. While drawing the selected one, also
       write 1 into the stencil buffer wherever it passes the depth test
       (glStencilFunc(GL_ALWAYS, 1, 0xFF), glStencilOp(..., GL_REPLACE)).
    3. Redraw the selected object fattened along its normals, with the
       depth test off and glStencilFunc(GL_NOTEQUAL, 1, 0xFF). Its own
       footprint just got stamped with 1 in step 2, so GL_NOTEQUAL rejects
       every fattened pixel that lands back on top of the original mesh --
       only the silhouette fringe beyond it survives.

See StencilOutline/README.md for the full state-machine table and the
teapot-vs-cube seam pitfall this technique has by design.
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
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

STENCIL_SHADER = "StencilShader"
OUTLINE_SHADER = "OutlineShader"
TINT_SHADER = "StencilTintShader"

OUTLINE_COLOUR = (1.0, 0.45, 0.05, 1.0)
TINT_COLOUR = (1.0, 0.45, 0.05, 0.35)

DEFAULT_OUTLINE_SCALE = 0.03
MIN_OUTLINE_SCALE = 0.005
MAX_OUTLINE_SCALE = 0.15


@dataclass(frozen=True)
class SceneObject:
    """One selectable object: which mesh, where, and its own colour."""

    label: str
    prim: str
    position: Vec3
    scale: float
    colour: tuple[float, float, float, float]


# teapot demonstrates a clean outline (smooth normals); the two cubes
# demonstrate the seam artefact scaling-along-normals produces on hard
# edges -- see the README, this is intentional, not a bug.
OBJECTS: tuple[SceneObject, ...] = (
    SceneObject("teapot", "teapot", Vec3(-2.6, 0.5, 0.0), 1.0, (0.75, 0.68, 0.55, 1.0)),
    SceneObject("cube A", "cube", Vec3(0.0, 0.5, 0.0), 0.7, (0.55, 0.62, 0.78, 1.0)),
    SceneObject("cube B", "cube", Vec3(2.4, 0.5, 1.2), 0.5, (0.55, 0.78, 0.6, 1.0)),
    SceneObject("sphere", "sphere", Vec3(0.6, 0.9, -2.2), 0.9, (0.78, 0.55, 0.62, 1.0)),
)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Grid scene with a keyboard-cycled selection and a stencil outline."""

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
        self.setTitle("Stencil Outline Selection (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.selected: int = 0
        self.outline_enabled: bool = True
        self.stencil_visualise: bool = False
        self.outline_scale: float = DEFAULT_OUTLINE_SCALE

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.18, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        # Stencil test stays enabled for the whole demo; func/op/mask are
        # switched per draw call below. Left at GL_ALWAYS/GL_KEEP with a
        # 0x00 write mask it is a no-op, which is the resting state we
        # return to after every object and every frame.
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
        gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_KEEP)
        gl.glStencilMask(0x00)

        self.view = look_at(
            Vec3(0.0, 3.0, 9.0), Vec3(0.0, 0.5, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            STENCIL_SHADER,
            str(shader_dir / "StencilVertex.glsl"),
            str(shader_dir / "StencilFragment.glsl"),
        )
        ok = ok and ShaderLib.load_shader(
            OUTLINE_SHADER,
            str(shader_dir / "OutlineVertex.glsl"),
            str(shader_dir / "FlatColourFragment.glsl"),
        )
        ok = ok and ShaderLib.load_shader(
            TINT_SHADER,
            str(shader_dir / "ScreenTintVertex.glsl"),
            str(shader_dir / "FlatColourFragment.glsl"),
        )
        if not ok:
            print("error loading shaders")
            self.close()
        ShaderLib.use(STENCIL_SHADER)
        ShaderLib.set_uniform("lightDir", Vec3(0.5, 1.0, 0.8))

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 12.0, 12.0, 24)
        # sphere isn't in the pre-baked default set load_default_primitives()
        # pulls from (unlike cube/teapot), so it must be created explicitly
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)
        # empty VAO for the full-screen tint triangle (core profile
        # requires a VAO to be bound even with no vertex attributes)
        self.screen_vao = gl.glGenVertexArrays(1)

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

    def _object_matrix(self, obj: SceneObject) -> Mat4:
        t = Transform()
        t.set_position(obj.position.x, obj.position.y, obj.position.z)
        t.set_scale(obj.scale, obj.scale, obj.scale)
        return t.matrix()

    def _load_matrices(self, shader: str, model: Mat4, global_tx: Mat4) -> None:
        MV = self.view @ global_tx @ model
        ShaderLib.use(shader)
        ShaderLib.set_uniform("MVP", self.project @ MV)
        if shader == STENCIL_SHADER:
            normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
            ShaderLib.set_uniform("normalMatrix", normal_matrix)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT
        )
        global_tx = self.scene_global_tx()

        # floor grid, unaffected by stencil: op is still GL_KEEP/GL_KEEP/GL_KEEP
        # from the last frame's restore, so nothing gets written regardless
        # of the write mask (which is 0xFF at this point, from that same restore)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        ShaderLib.set_uniform("Colour", 0.55, 0.55, 0.55, 1.0)
        Primitives.draw("grid")

        # ---- pass 1: draw every object; the selected one also stamps the
        # stencil buffer with 1 wherever it passes the depth test ----
        for i, obj in enumerate(OBJECTS):
            model = self._object_matrix(obj)
            is_selected = i == self.selected
            if is_selected and self.outline_enabled:
                gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
                gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
                gl.glStencilMask(0xFF)
            else:
                gl.glStencilMask(0x00)

            self._load_matrices(STENCIL_SHADER, model, global_tx)
            ShaderLib.set_uniform("Colour", *obj.colour)
            Primitives.draw(obj.prim)

            if is_selected and self.outline_enabled:
                # ---- pass 2: outline -- fattened copy, stencil-rejected
                # everywhere the object above just wrote 1 into ----
                gl.glStencilFunc(gl.GL_NOTEQUAL, 1, 0xFF)
                gl.glStencilMask(0x00)
                gl.glDisable(gl.GL_DEPTH_TEST)

                self._load_matrices(OUTLINE_SHADER, model, global_tx)
                ShaderLib.set_uniform("outlineScale", self.outline_scale)
                ShaderLib.set_uniform("Colour", *OUTLINE_COLOUR)
                Primitives.draw(obj.prim)

                gl.glEnable(gl.GL_DEPTH_TEST)

            # restore the resting state before the next object / frame
            gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_KEEP)
            gl.glStencilMask(0x00)

        # ---- optional: visualise the stencil buffer as a translucent
        # tint wherever it reads 1 ----
        if self.stencil_visualise:
            gl.glStencilFunc(gl.GL_EQUAL, 1, 0xFF)
            gl.glStencilMask(0x00)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            ShaderLib.use(TINT_SHADER)
            ShaderLib.set_uniform("Colour", *TINT_COLOUR)
            gl.glBindVertexArray(self.screen_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)

            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_KEEP)

        # stencil write mask must land back on 0xFF before the *next*
        # frame's glClear, otherwise GL_STENCIL_BUFFER_BIT is masked off
        # and the clear silently stops touching the stencil buffer.
        gl.glStencilMask(0xFF)

        self._draw_hud()

    def _draw_hud(self) -> None:
        selected_obj = OBJECTS[self.selected]
        state = (
            f"[Tab] selected: {selected_obj.label}  "
            f"[O]utline {'ON ' if self.outline_enabled else 'OFF'}  "
            f"[V]isualise stencil {'ON ' if self.stencil_visualise else 'OFF'}  "
            f"width [+/-] {self.outline_scale:.3f}"
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
        elif key == Qt.Key_Tab:
            self.selected = (self.selected + 1) % len(OBJECTS)
        elif key == Qt.Key_O:
            self.outline_enabled = not self.outline_enabled
        elif key == Qt.Key_V:
            self.stencil_visualise = not self.stencil_visualise
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.outline_scale = min(MAX_OUTLINE_SCALE, self.outline_scale + 0.005)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.outline_scale = max(MIN_OUTLINE_SCALE, self.outline_scale - 0.005)
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
    # LOUD COMMENT: without this the stencil buffer does not exist and every
    # glStencilFunc/glStencilOp call above is silently a no-op -- the whole
    # demo runs, nothing crashes, and no outline ever appears. This one line
    # is the entire "setup" for stencil rendering.
    format.setStencilBufferSize(8)
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
