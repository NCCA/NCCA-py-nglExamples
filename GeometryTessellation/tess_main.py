#!/usr/bin/env -S uv run --script
"""
Tessellated, noise-displaced plane with distance-based LOD (OpenGL).

A flat 16x16 grid of 4-vertex `GL_PATCHES` is uploaded once; every frame the
tessellation control shader (TCS) decides *how much* to subdivide each
patch based on its distance from the camera, and the tessellation
evaluation shader (TES) bilinearly interpolates the generated vertex across
the patch and displaces it with 4-octave fbm noise computed entirely in
GLSL (no textures).

    L            toggle distance-based LOD vs. a fixed tessellation level
    +/-          (fixed-level mode only) raise/lower the fixed level
    W            toggle wireframe -- this is the whole point of the demo:
                 watch the triangle density grow near the camera and
                 shrink far away, and see fractional_even spacing keep it
                 from popping as the LOD boundary sweeps past
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit

Teaching points:
    1. Tessellation is a *fixed* pipeline stage pair sitting between the
       vertex shader and the geometry/fragment stages: TCS runs once per
       *output* control point (here 4, since GL_PATCH_VERTICES=4) and sets
       `gl_TessLevelOuter/Inner`; the fixed-function tessellator then
       generates new vertices inside the patch's parametric domain; TES
       runs once per *generated* vertex and decides its final position.
    2. `glPatchParameteri(GL_PATCH_VERTICES, 4)` and drawing with
       `GL_PATCHES` are both required -- without them the draw call
       silently produces nothing, there is no error.
    3. `gl_TessLevelOuter/Inner` must only be written by TCS invocation 0
       (`if (gl_InvocationID == 0)`) -- they are per-patch state, and every
       invocation racing to write them is undefined behaviour.
    4. `fractional_even_spacing` (vs. `equal_spacing`) grows/shrinks the
       outermost ring of triangles continuously as the tessellation level
       changes, instead of new triangles snapping into existence -- this
       is what keeps the LOD transition from visibly "popping".
    5. There is no `Primitives` helper for patches; the control-point grid
       is built by hand in numpy (`tess_grid.build_patch_grid`) and
       uploaded as a flat, non-indexed vertex buffer.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    PySideEventHandlingMixin,
    ShaderLib,
    ShaderType,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from tess_grid import build_patch_grid, patch_count

TESS_SHADER = "TessPlane"

GRID_RESOLUTION = 16
GRID_SIZE = 10.0
EYE_POSITION = Vec3(0.0, 5.0, 9.0)

NEAR_DISTANCE = 3.0
FAR_DISTANCE = 16.0
HEIGHT_SCALE = 1.2
NOISE_SCALE = 0.35
LIGHT_DIR_WORLD = Vec3(0.4, 1.0, 0.3).normalized()

MIN_FIXED_LEVEL = 1.0
MAX_FIXED_LEVEL = 64.0
FIXED_LEVEL_STEP = 1.0


def load_tess_program(program_name: str, shader_dir: Path) -> bool:
    """Build a vert/TCS/TES/frag program via ShaderLib's *lower-level* API.

    ``ShaderLib.load_shader(name, vert, frag, geo=...)`` -- the convenience
    entry point used everywhere else in this repo -- only wires up vertex,
    fragment and (optionally) geometry stages; it has no parameters for
    tessellation control/evaluation shaders. The underlying registry it is
    built on top of is *not* similarly limited: ``ShaderType`` already
    defines ``TESSCONTROL``/``TESSEVAL`` (they map straight to
    ``GL_TESS_CONTROL_SHADER``/``GL_TESS_EVALUATION_SHADER``), and
    ``ShaderLib`` exposes the per-stage building blocks
    (``create_shader_program``, ``attach_shader``, ``load_shader_source``,
    ``compile_shader``, ``attach_shader_to_program``,
    ``link_program_object``) used internally by ``load_shader`` itself.
    Driving those directly assembles a 4-stage program through ShaderLib
    -- so it stays registered exactly like any other shader (``use``,
    ``set_uniform`` etc. all work unmodified) -- with no library changes
    and no need to duplicate compilation/linking in raw PyOpenGL.
    """
    ShaderLib.create_shader_program(program_name)

    stages = (
        (f"{program_name}Vert", ShaderType.VERTEX, "TessPlaneVertex.glsl"),
        (f"{program_name}TC", ShaderType.TESSCONTROL, "TessPlaneControl.glsl"),
        (f"{program_name}TE", ShaderType.TESSEVAL, "TessPlaneEval.glsl"),
        (f"{program_name}Frag", ShaderType.FRAGMENT, "TessPlaneFragment.glsl"),
    )

    ok = True
    for shader_name, shader_type, filename in stages:
        ShaderLib.attach_shader(shader_name, shader_type)
        ShaderLib.load_shader_source(shader_name, str(shader_dir / filename))
        ok = ShaderLib.compile_shader(shader_name) and ok
        ShaderLib.attach_shader_to_program(program_name, shader_name)

    return ShaderLib.link_program_object(program_name) and ok


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Distance-tessellated, noise-displaced GL_PATCHES plane."""

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
        self.setTitle("Tessellated Displaced Plane (OpenGL)")

        self.fixed_level_mode: bool = False
        self.fixed_level_value: float = 8.0
        self.wireframe: bool = True
        self.patch_vao = None

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.08, 0.08, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(EYE_POSITION, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))

        shader_dir = Path(__file__).parent / "shaders"
        if not load_tess_program(TESS_SHADER, shader_dir):
            print("error loading tessellation shader stages")
            self.close()

        # Required or GL_PATCHES draws nothing -- silently, no GL error.
        gl.glPatchParameteri(gl.GL_PATCH_VERTICES, 4)

        self._build_patch_grid()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    def _build_patch_grid(self) -> None:
        verts = build_patch_grid(GRID_RESOLUTION, GRID_SIZE)
        self.patch_count = patch_count(GRID_RESOLUTION)

        self.patch_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_PATCHES)
        with self.patch_vao:
            self.patch_vao.set_data(VertexData(data=verts, size=verts.shape[0]))
            stride = 3 * np.dtype(np.float32).itemsize
            self.patch_vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)

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

        M = self.scene_global_tx()
        VP = self.project @ self.view

        ShaderLib.use(TESS_SHADER)
        ShaderLib.set_uniform("M", M)
        ShaderLib.set_uniform("VP", VP)
        ShaderLib.set_uniform("cameraPosWorld", EYE_POSITION)
        ShaderLib.set_uniform("nearDistance", NEAR_DISTANCE)
        ShaderLib.set_uniform("farDistance", FAR_DISTANCE)
        ShaderLib.set_uniform("fixedLevel", self.fixed_level_mode)
        ShaderLib.set_uniform("fixedLevelValue", self.fixed_level_value)
        ShaderLib.set_uniform("heightScale", HEIGHT_SCALE)
        ShaderLib.set_uniform("noiseScale", NOISE_SCALE)
        ShaderLib.set_uniform("lightDirWorld", LIGHT_DIR_WORLD)

        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        with self.patch_vao:
            self.patch_vao.draw()

        self._draw_hud()

    def _draw_hud(self) -> None:
        level_text = (
            f"fixed={self.fixed_level_value:.0f}"
            if self.fixed_level_mode
            else f"distance-based [{NEAR_DISTANCE:.0f}..{FAR_DISTANCE:.0f}]"
        )
        state = (
            f"[L] LOD mode: {level_text}   "
            f"[+/-] fixed level (when L is fixed)   "
            f"[W] wireframe {'ON ' if self.wireframe else 'OFF'}"
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
        elif key == Qt.Key_L:
            self.fixed_level_mode = not self.fixed_level_mode
        elif key == Qt.Key_W:
            self.wireframe = not self.wireframe
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.fixed_level_value = min(
                MAX_FIXED_LEVEL, self.fixed_level_value + FIXED_LEVEL_STEP
            )
        elif key == Qt.Key_Minus:
            self.fixed_level_value = max(
                MIN_FIXED_LEVEL, self.fixed_level_value - FIXED_LEVEL_STEP
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
