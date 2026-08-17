#!/usr/bin/env -S uv run --script
"""BvhViewer: play back a .bvh motion-capture file (PyNGL / PySide6).

A port of the C++ NGL BvhViewer demo -- see ../README.md for what changed
and why. Loads one skeleton and plays its animation on a loop.

Controls:
    left mouse    rotate
    right mouse   pan
    wheel         zoom
    R             replay from frame 0
    P             pause / continue
    Left / Right  step backward / forward one frame (works while paused)
    Space         clear the character from the scene
    T             toggle trace mode (stop clearing the framebuffer, for a motion-trail look)
    W / S         wireframe / filled
    F             fullscreen
    Escape        quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from bvh import Bvh
from bvh_scene import BvhScene
from ncca.ngl import Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import PySideEventHandlingMixin, Text
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_BVH = DEMO_DIR / "bvh" / "Male1_B10_WalkTurnLeft45.bvh"
HUD_FONT = DEMO_DIR.parent / "font" / "Arial.ttf"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """The BvhViewer window: camera, playback controls and the scene."""

    def __init__(self, bvh_path: Path = DEFAULT_BVH, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self._bvh_path = bvh_path
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.scene: BvhScene = BvhScene()
        self._trace: bool = False
        self.setTitle("BvhViewer (PyNGL)")

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 25, 100), Vec3(0, 0, 0), Vec3(0, 1, 0))

        self.scene.initialize_gl()
        character = Bvh(self._bvh_path)
        self.scene.add_character(character)

        Text.add_font("Arial", str(HUD_FONT), 14)

        self.startTimer(max(1, int(character.frame_time * 1000)))
        self._update_title()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        if not self._trace:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        mouse_global_tx = rot_y @ rot_x
        mouse_global_tx[3, 0] = self.model_position.x
        mouse_global_tx[3, 1] = self.model_position.y
        mouse_global_tx[3, 2] = self.model_position.z

        self.scene.draw(self.view, self.project, mouse_global_tx)
        self._draw_hud()

    def _draw_hud(self) -> None:
        Text.render_text("Arial", 10, 20, f"characters: {len(self.scene.characters)}")
        Text.render_text(
            "Arial",
            10,
            40,
            f"frame: {self.scene.current_frame_number()} {'(paused)' if self.scene.paused else ''}",
        )
        Text.render_text(
            "Arial",
            10,
            60,
            "r replay | p pause | arrows step | space clear | t trace | w/s wire/fill",
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / max(h, 1), 0.05, 1500.0)
        Text.set_screen_size(w, h)

    def timerEvent(self, event) -> None:
        self.scene.advance()
        self._update_title()
        self.update()

    def _update_title(self) -> None:
        state = "paused" if self.scene.paused else "playing"
        self.setTitle(
            f"BvhViewer (PyNGL) - {self._bvh_path.name} - frame {self.scene.current_frame_number()} - {state}"
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_R:
            self.scene.replay()
        elif key == Qt.Key_P:
            self.scene.toggle_pause()
        elif key == Qt.Key_Right:
            self.scene.step_forward()
        elif key == Qt.Key_Left:
            self.scene.step_backward()
        elif key == Qt.Key_Space:
            self.scene.clear_characters()
        elif key == Qt.Key_T:
            self._trace = not self._trace
        elif key == Qt.Key_F:
            self.showFullScreen()
        else:
            super().keyPressEvent(event)
            return
        self._update_title()
        self.update()


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
        "--bvh",
        type=Path,
        default=DEFAULT_BVH,
        help="path to the .bvh file to play (default: %(default)s)",
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow(args.bvh)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
