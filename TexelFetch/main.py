#!/usr/bin/env -S uv run --script
"""TexelFetch: a 200x200 grid of points whose height is read from a Texture
Buffer Object with texelFetch in the vertex shader, indexed by gl_VertexID.

Ported from NGL9Demos/TexelFetch -- the XZ positions live in an ordinary
vertex buffer, but the Y (height) values live in a separate TBO that gets
re-uploaded every tick, so the vertex shader never touches a fixed-height
attribute at all.
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
    VAOFactory,
    VAOType,
)
from ncca.ngl.opengl.abstract_vao import VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

# 200 samples per axis in [-10, 10) -- 40,000 points total, matching the
# C++ source's m_gridSize.
_GRID_DIM = 10.0
_GRID_STEP = 0.1
_GRID_N = int(round(2 * _GRID_DIM / _GRID_STEP))  # 200
_COORDS = (np.arange(_GRID_N, dtype=np.float32) * _GRID_STEP) - _GRID_DIM


def _build_xz() -> np.ndarray:
    """Build the (x, z) grid positions in the same order as the C++'s outer-z/inner-x loop.

    Returns
    -------
        np.ndarray
            (_GRID_N * _GRID_N, 2) array of (x, z) pairs, Z varying slowest.
    """
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    return np.stack([x.ravel(), z.ravel()], axis=1).astype(np.float32)


def _build_y(offset: float | None) -> np.ndarray:
    """Build the height (Y) values for the TBO, in the same order as `_build_xz`.

    Parameters
    ----------
        offset : float | None
            Animation offset. None reproduces the source's `buildTextureBuffer()`,
            which uses `sin(x)` alone before the first tick -- everything after
            that first frame goes through `updateTextureBuffer()`'s
            `sin(x + offset) + cos(x - offset)` instead. That asymmetry is in
            the original C++, not a bug here.

    Returns
    -------
        np.ndarray
            (_GRID_N * _GRID_N,) array of heights.
    """
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    if offset is None:
        return np.sin(x).ravel().astype(np.float32)
    return (np.sin(x + offset) + np.cos(x - offset)).ravel().astype(np.float32)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Renders the grid as GL_POINTS, texelFetch-ing height from a TBO each vertex."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("TexelFetch")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.vertex_count: int = _GRID_N * _GRID_N
        self.offset: float = 0.0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.window_width / self.window_height, 0.01, 150.0
        )

        # The C++ source also does ShaderLib::use("nglColourShader") and sets
        # a Colour uniform here, but that shader is never bound again --
        # paintGL only ever uses "TexelShader", whose fragment shader
        # hardcodes vec4(1.0) regardless. Dead code from the original,
        # deliberately left out rather than carried over.

        xz = _build_xz()
        self.xz_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_POINTS)
        self.xz_vao.bind()
        self.xz_vao.set_data(VertexData(xz.ravel(), self.vertex_count))
        self.xz_vao.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, 2 * 4, 0)
        self.xz_vao.set_num_indices(self.vertex_count)
        self.xz_vao.unbind()

        # Height data lives in a Texture Buffer Object rather than a second
        # vertex attribute, so the vertex shader can texelFetch it by
        # gl_VertexID. No ncca.ngl wrapper exists for TBOs, hence raw GL here.
        y = _build_y(None)
        self.ypos_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.ypos_buffer)
        gl.glBufferData(gl.GL_TEXTURE_BUFFER, y.nbytes, y, gl.GL_STATIC_DRAW)
        self.tbo_texture = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.tbo_texture)
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R32F, self.ypos_buffer)

        shader_dir = Path(__file__).parent
        ShaderLib.load_shader(
            "TexelShader",
            str(shader_dir / "VertexShader.glsl"),
            str(shader_dir / "FragmentShader.glsl"),
        )
        ShaderLib.use("TexelShader")
        ShaderLib.set_uniform("yPosSampler", 0)

        self.animation_timer.start(20)

    def _on_tick(self) -> None:
        """Recompute the height data and re-upload it to the TBO's buffer."""
        y = _build_y(self.offset)
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.ypos_buffer)
        gl.glBufferData(gl.GL_TEXTURE_BUFFER, y.nbytes, y, gl.GL_STATIC_DRAW)
        self.offset += 0.01
        self.update()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use("TexelShader")
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.tbo_texture)
        gl.glPointSize(4)
        with self.xz_vao as vao:
            vao.draw()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 150.0)

    def keyPressEvent(self, event) -> None:
        # F/N (fullscreen/windowed) on top of the mixin's own Escape/W/S/Space
        # handling -- W and S are visually inert here since GL_POINTS has no
        # edges to outline, but they're ported anyway for fidelity with the
        # source, which has the same vestigial behaviour.
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        else:
            super().keyPressEvent(event)
            return
        self.update()

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions from Qt event handlers instead of swallowing them."""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
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

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
