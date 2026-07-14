#!/usr/bin/env -S uv run --script
"""
Marching Cubes: metaballs polygonised on the CPU, every frame (OpenGL).

Four metaballs drift along lissajous paths. Each frame their scalar field
is re-sampled and re-polygonised in numpy (see marching_cubes.py), and the
resulting triangle soup is pushed straight into the GPU with
`glBufferData(..., GL_DYNAMIC_DRAW)` -- there is no persistent mesh here,
just a VAO that gets completely re-specified every paint. See
VertexArrayObject/ChangingVAO for the same lesson in miniature.

    +/-        grid resolution (16 / 32 / 48 / 64)
    I          raise iso level
    Shift+I    lower iso level
    W          toggle wireframe
    Space      pause/resume the metaball animation
    Esc        quit

See the README for the algorithm walk-through and why 48^3 is the sweet
spot between blobbiness and frame time.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from marching_cubes import polygonise, sample_metaballs
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

# grid resolutions cycled with +/-
GRID_SIZES = (16, 32, 48, 64)
DEFAULT_GRID_INDEX = 2  # 48^3 -- measured well under 50 ms/frame, see README

EXTENT = 2.0  # metaball field spans [-EXTENT, EXTENT]^3
DEFAULT_ISO = 1.0
ISO_STEP = 0.1
ISO_MIN, ISO_MAX = 0.2, 4.0

# four metaballs on lissajous paths: (amplitude, (fx, fy, fz), (px, py, pz), radius)
METABALL_PARAMS = (
    (1.1, (0.7, 0.9, 0.5), (0.0, 0.0, 0.0), 0.85),
    (1.0, (0.5, 0.6, 0.8), (1.3, 0.4, 2.1), 0.75),
    (0.9, (0.9, 0.4, 0.6), (2.6, 1.9, 0.7), 0.7),
    (1.0, (0.6, 0.8, 0.9), (0.5, 2.4, 1.5), 0.8),
)


def _metaball_centres(t: float) -> np.ndarray:
    """World-space centres of the four metaballs at time `t` (seconds)."""
    centres = np.empty((len(METABALL_PARAMS), 3), dtype=np.float32)
    for i, (amp, freq, phase, _radius) in enumerate(METABALL_PARAMS):
        fx, fy, fz = freq
        px, py, pz = phase
        centres[i] = (
            amp * np.sin(fx * t + px),
            amp * np.sin(fy * t + py),
            amp * np.sin(fz * t + pz),
        )
    return centres


_METABALL_RADII = np.array([p[3] for p in METABALL_PARAMS], dtype=np.float32)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Re-polygonises a 4-metaball scalar field into a mesh every frame."""

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
        self.setTitle("Marching Cubes: metaballs (OpenGL)")

        self.grid_index: int = DEFAULT_GRID_INDEX
        self.iso: float = DEFAULT_ISO
        self.wireframe: bool = False
        self.paused: bool = False

        self._clock = QElapsedTimer()
        self._sim_time: float = 0.0  # accumulated, frozen while paused
        self._last_wall_time: float = 0.0

        # HUD stats, refreshed each time polygonise() runs
        self.triangle_count: int = 0
        self.polygonise_ms: float = 0.0

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.18, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 1.5, 8.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 4.0, 5.0, 6.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        self.vao = VAOFactory.create_vao(VAOType.MULTI_BUFFER, gl.GL_TRIANGLES)

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

        self._clock.start()
        # ~60 fps repaint loop -- this is a continuously-animating scene,
        # not a mouse-driven one, so we drive it with our own timer rather
        # than waiting for events.
        self.startTimer(16)

    # ------------------------------------------------------------------
    # per-frame mesh rebuild
    # ------------------------------------------------------------------
    def _rebuild_mesh(self) -> None:
        grid_n = GRID_SIZES[self.grid_index]
        centres = _metaball_centres(self._sim_time)

        t0 = time.perf_counter()
        field = sample_metaballs(centres, _METABALL_RADII, grid_n, EXTENT)
        verts, normals = polygonise(field, self.iso, EXTENT)
        self.polygonise_ms = (time.perf_counter() - t0) * 1000.0
        self.triangle_count = len(verts) // 3

        with self.vao:
            self.vao.set_data(VertexData(verts, len(verts), gl.GL_DYNAMIC_DRAW), 0)
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
            self.vao.set_data(VertexData(normals, len(normals), gl.GL_DYNAMIC_DRAW), 1)
            self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 0, 0)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def paintGL(self) -> None:
        self.makeCurrent()

        if not self.paused:
            now = self._clock.elapsed() * 0.001
            self._sim_time += now - self._last_wall_time
            self._last_wall_time = now
        else:
            self._last_wall_time = self._clock.elapsed() * 0.001

        self._rebuild_mesh()

        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPolygonMode(
            gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL
        )

        global_tx = self.scene_global_tx()
        MV = self.view @ global_tx
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("MVP", self.project @ MV)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(MV).inverse().transposed())
        ShaderLib.set_uniform("Colour", 0.85, 0.55, 0.25, 1.0)

        if self.triangle_count > 0:
            with self.vao:
                self.vao.draw()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        self._draw_hud()

    def _draw_hud(self) -> None:
        grid_n = GRID_SIZES[self.grid_index]
        Text.render_text(
            "Arial",
            10,
            20,
            f"grid {grid_n}^3  [+/-]   iso {self.iso:.2f}  [I/Shift+I]   "
            f"[W]ireframe {'ON' if self.wireframe else 'off'}   "
            f"{'PAUSED [Space]' if self.paused else '[Space] pause'}",
            Vec3(1.0, 1.0, 1.0),
        )
        Text.render_text(
            "Arial",
            10,
            45,
            f"triangles {self.triangle_count}   polygonise {self.polygonise_ms:.2f} ms",
            Vec3(1.0, 1.0, 0.0),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)
        Text.set_screen_size(w, h)

    def timerEvent(self, event) -> None:
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.grid_index = min(self.grid_index + 1, len(GRID_SIZES) - 1)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.grid_index = max(self.grid_index - 1, 0)
        elif key == Qt.Key_I and shift:
            self.iso = max(ISO_MIN, self.iso - ISO_STEP)
        elif key == Qt.Key_I:
            self.iso = min(ISO_MAX, self.iso + ISO_STEP)
        elif key == Qt.Key_W:
            self.wireframe = not self.wireframe
        elif key == Qt.Key_Space:
            self.paused = not self.paused
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
