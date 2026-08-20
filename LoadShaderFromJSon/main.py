#!/usr/bin/env -S uv run --script
"""LoadShaderFromJSon: assemble a shader program at runtime from a JSON manifest.

Ported from NGL9Demos/LoadShaderFromJSon -- rather than pointing ShaderLib at
one vertex file and one fragment file, `shaders/shaders.json` lists each
stage as an *ordered list of files to concatenate*. That's what lets
`common.glsl` (the `Materials`/`Lights` structs and the `light`/`material`/
`time`/`repeat` uniforms) and `noise3D.glsl` (Ashima Arts' `snoise`) be
shared source fragments pulled into the vertex and fragment stages without
duplicating them by hand. `load_shader_from_json()` below reads that
manifest and drives ShaderLib's low-level per-stage API directly, since
`ShaderLib.load_shader()` only takes a single file per stage.

The teapot itself is shaded by a Phong lighting pass whose surface colour
comes from six octaves of 3D simplex noise (a fractal sum), giving the gold
material a mottled, marble-like look that shifts as `time` advances.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    ShaderType,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

_TYPE_TO_SHADERTYPE = {"Vertex": ShaderType.VERTEX, "Fragment": ShaderType.FRAGMENT}


def load_shader_from_json(json_path: Path) -> str:
    """Build a ShaderLib program from a JSON manifest of per-stage source files.

    Each entry under "Shaders" names one pipeline stage and a list of GLSL
    files to concatenate, in order, into that stage's source -- this is how
    `common.glsl`'s struct/uniform declarations and `noise3D.glsl`'s
    `snoise()` end up shared between the vertex and fragment stages without
    copy-pasting them.

    Parameters
    ----------
        json_path : Path
            Path to the manifest, e.g. `shaders/shaders.json`.

    Returns
    -------
        str
            The registered shader program name, ready for `ShaderLib.use()`.
    """
    data = json.loads(json_path.read_text())
    program = data["ShaderProgram"]
    program_name = program["name"]
    ShaderLib.create_shader_program(program_name)

    base_dir = (
        json_path.parent.parent
    )  # JSON's "shaders/..." paths are relative to the demo folder
    ok = True
    for stage in program["Shaders"]:
        shader_name = stage["name"]
        shader_type = _TYPE_TO_SHADERTYPE[stage["type"]]
        source = "\n".join((base_dir / p).read_text() for p in stage["path"])
        ShaderLib.attach_shader(shader_name, shader_type)
        ShaderLib.load_shader_source_from_string(shader_name, source)
        ok = ShaderLib.compile_shader(shader_name) and ok
        ShaderLib.attach_shader_to_program(program_name, shader_name)

    if not (ShaderLib.link_program_object(program_name) and ok):
        logger.error(
            f"Failed to build shader program {program_name!r} from {json_path}"
        )
    return program_name


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Renders a Phong-lit teapot whose colour is perturbed by layered simplex noise."""

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
        self.setTitle("LoadShaderFromJSon")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.program_name: str = ""
        self.time: float = 0.0
        # Faithfully-preserved source quirk: the C++ key handler's `repeat`
        # is a `static float` local to keyPressEvent, seeded at 0.1f --
        # entirely separate from the 0.01f the shader's `repeat` uniform is
        # given in initializeGL. The first `1`/`2` press therefore jumps the
        # uniform from 0.01 straight to 0.09/0.11, a visible discontinuity
        # in the noise UV scale. This attribute mirrors the key handler's
        # static, not the init value -- don't "fix" the mismatch.
        self.repeat: float = 0.1
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.window_width / self.window_height, 0.05, 350.0
        )

        self.program_name = load_shader_from_json(
            Path(__file__).parent / "shaders" / "shaders.json"
        )
        ShaderLib.use(self.program_name)

        ShaderLib.set_uniform("light.position", -2.0, 5.0, 2.0, 0.0)
        ShaderLib.set_uniform("light.ambient", 0.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("light.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light.specular", 0.8, 0.8, 0.8, 1.0)
        # gold-like NoiseShader material
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 0.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 0.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 0.0)
        ShaderLib.set_uniform("material.shininess", 51.2)
        ShaderLib.set_uniform("viewerPos", 0.0, 1.0, 2.0)

        ShaderLib.set_uniform("time", 0.0)
        ShaderLib.set_uniform("repeat", 0.01)

        Primitives.load_default_primitives()

        self.animation_timer.start(20)

    def _on_tick(self) -> None:
        self.time += 0.01
        ShaderLib.use(self.program_name)
        ShaderLib.set_uniform("time", self.time)
        self.update()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(self.program_name)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        M = global_tx
        MV = self.view @ M
        MVP = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("M", M)

        Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        # F/N (fullscreen/windowed) and 1/2 (repeat -/+ 0.01) on top of the
        # mixin's own Escape/W/S/Space handling.
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_1:
            self.repeat -= 0.01
            ShaderLib.use(self.program_name)
            ShaderLib.set_uniform("repeat", self.repeat)
        elif key == Qt.Key_2:
            self.repeat += 0.01
            ShaderLib.use(self.program_name)
            ShaderLib.set_uniform("repeat", self.repeat)
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
