#!/usr/bin/env -S uv run --script
"""TextureCompressor: a viewer for DXT1/S3TC block-compressed textures.

Ported from NGL9Demos/TextureCompressor/DXTViewer -- a full-screen NDC quad
(ScreenQuad.cpp) samples a compressed texture loaded from an `ngl::cmptx`
file (DXTTexture::load()). There's no camera here: the quad already covers
clip space, so paintGL never touches an MVP uniform, and the mixin's mouse
rotate/pan/zoom -- wired up for consistency with the rest of this repo --
has nothing visible to drive, exactly like the source's empty mouse handlers.

Unlike the source, which links `libsquish` to do the actual DXT1 encoding,
the `.cmptx` files this demo reads are written by this folder's own
`compress_texture.py`, a from-scratch numpy encoder -- see dxt_texture.py.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from dxt_texture import read_cmptx
from ncca.ngl import Vec3, logger
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QFileDialog

_DEFAULT_TEXTURE = Path(__file__).parent / "textures" / "texture.cmptx"

# ScreenQuad.cpp's two triangles, verbatim: position in NDC, UV with V
# flipped (0.0 at the top) to match the source's winding.
_QUAD_VERTS = (
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    -1.0, 1.0, 0.0,
)  # fmt: skip
_QUAD_UVS = (
    0.0, 1.0,
    1.0, 1.0,
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 0.0,
)  # fmt: skip


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Draws one compressed texture on a full-screen quad; `O` swaps it for another file."""

    def __init__(self, initial_file: Path | None = None, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.initial_file: Path = initial_file or _DEFAULT_TEXTURE
        self.tex_id: int = 0
        self.filename: Path = self.initial_file
        self.setTitle(f"DXT Viewer use o to load new file current {self.filename}")

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self._build_screen_quad()

        shader_dir = Path(__file__).parent
        ShaderLib.load_shader(
            "Texture",
            str(shader_dir / "TextureVertex.glsl"),
            str(shader_dir / "TextureFragment.glsl"),
        )
        ShaderLib.use("Texture")
        ShaderLib.set_uniform("tex", 0)

        self._load_cmptx(self.initial_file)

    def _build_screen_quad(self) -> None:
        """A two-triangle NDC quad, position + UV, same layout as ScreenQuad.cpp."""
        verts = np.array(_QUAD_VERTS, dtype=np.float32)
        uvs = np.array(_QUAD_UVS, dtype=np.float32)

        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        vbo_id = gl.glGenBuffers(2)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id[1])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, uvs.nbytes, uvs, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    def _load_cmptx(self, path: Path) -> None:
        """Read an ngl::cmptx file and upload it as a compressed GL texture.

        Mirrors DXTTexture::load(): delete whatever texture is currently bound
        (reset()) then upload the new one via glCompressedTexImage2D.

        Note: don't pass an explicit imageSize here, despite the C++ 8-arg
        signature suggesting you should. This PyOpenGL build's wrapper
        computes imageSize itself from `data`'s buffer length (see
        OpenGL.GL.images.CompressedImageConverter) and has *already* dropped
        imageSize from the Python-facing argument list -- passing it anyway
        shifts every later positional argument along by one, so `data` lands
        where PyOpenGL expects `imageSize` and raises `'NumberHandler' object
        has no attribute 'arrayByteCount'`. Confirmed against the installed
        OpenGL.GL.VERSION.GL_1_3 wrapper, not just inferred from the error.
        """
        try:
            width, height, internal_format, data = read_cmptx(path)
        except (ValueError, OSError) as exc:
            logger.error(f"Could not load {path}: {exc}")
            return

        if self.tex_id:
            gl.glDeleteTextures(1, [self.tex_id])

        self.tex_id = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glCompressedTexImage2D(
            gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, data
        )

        self.filename = path
        self.setTitle(f"DXT Viewer use o to load new file current {path}")
        # DXTViewer resizes the window to the loaded texture's dimensions in
        # initializeGL (setWidth/setHeight) -- reproduced here for fidelity.
        self.resize(width, height)

    def _reload(self) -> None:
        """QFileDialog file-open, matching DXTViewer::reload()'s `O`-key handler."""
        filename, _ = QFileDialog.getOpenFileName(
            None,
            "load texture",
            str(Path.cwd()),
            "Compressed Texture (*.cmptx);;All Files (*)",
        )
        if filename:
            self._load_cmptx(Path(filename))

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use("Texture")
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
        gl.glBindVertexArray(self.vao_id)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())

    def keyPressEvent(self, event) -> None:
        # F/N (fullscreen/windowed) and O (load a new .cmptx file) on top of
        # the mixin's own Escape/W/S/Space handling.
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_O:
            self._reload()
        else:
            super().keyPressEvent(event)
            return
        self.update()


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
        "--file",
        type=Path,
        default=None,
        help="a .cmptx file to preload instead of the bundled sample texture",
    )
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
    window = MainWindow(initial_file=args.file)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
