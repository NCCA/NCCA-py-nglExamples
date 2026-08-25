#!/usr/bin/env -S uv run --script
"""Game-style held-key spaceship control (OpenGL).

A direct port of NGL9Demos/AdvancedGameKeyControl: a spaceship nudged around
a 2D plane by whichever of Up/Down/Left/Right/R are held down *right now*,
looked up each tick in `game_controls.MOTION_TABLE` -- a flat 32-entry table
indexed by the held-key bitmask, so every simultaneous combination (e.g.
Up+Left) resolves to one pre-computed diagonal offset rather than a chain of
if/elif branches. Space toggles recording of every frame's bitmask to a
`KeyRecorder`; P replays it back over the spaceship's original starting
position; S/L save and load a recording to/from a `.kp` file.

Unlike almost every other demo in this repo there is no mouse camera control
at all -- the camera is a single fixed `look_at`, matching the C++ source,
which never overrides a mouse handler. There are also two independent
QTimers rather than the usual one: a 15ms timer that samples the held keys
and steps the ship (`_on_ship_tick`), and a separate 30ms timer that only
calls `update()` -- preserving the source's `timerEvent` split between input
sampling and redraw rate.
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from game_controls import GameControls, KeyRecorder, move_ship, ship_transform
from ncca.ngl import Mat4, Obj, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import DefaultShader, ShaderLib, Text
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QFileDialog

MODELS_DIR = Path(__file__).parent / "models"
FONT_PATH = Path(__file__).parent.parent / "font" / "Arial.ttf"


class MainWindow(QOpenGLWindow):
    """Held-key spaceship demo window.

    Deliberately plain `QOpenGLWindow` -- no `PySideEventHandlingMixin`, no
    mouse handlers of any kind. The C++ source has none, so neither does
    this port.
    """

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Game style key control in Qt")

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        # A static camera -- set once, never touched by mouse input.
        self.view = look_at(
            Vec3(0.0, 0.0, 80.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(45.0, self.width() / self.height(), 0.05, 350.0)

        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)

        self.ship = Obj.from_file(str(MODELS_DIR / "SpaceShip.obj"))
        self.ship.create_vao()
        self.ship_pos = Vec3(0.0, 0.0, 0.0)
        self.ship_rotation = 0.0

        Text.add_font("Arial", str(FONT_PATH), 12)
        Text.set_screen_size(self.width(), self.height())

        self.keys_pressed = 0
        self.recording = False
        self.playback_active = False
        self.current_playback_frame = 0
        self.key_recorder = KeyRecorder()

        # Two independent timers, matching the source's timerEvent split:
        # one drives ship movement/recording/playback at 15ms, the other
        # only triggers a redraw at 30ms.
        self.ship_timer = QTimer(self)
        self.ship_timer.timeout.connect(self._on_ship_tick)
        self.ship_timer.start(15)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.timeout.connect(self.update)
        self.redraw_timer.start(30)

    def _on_ship_tick(self) -> None:
        if self.playback_active:
            if self.current_playback_frame <= 0:
                self.ship_pos = self.key_recorder.get_start_position()
            if self.current_playback_frame < self.key_recorder.size():
                self.keys_pressed = self.key_recorder[self.current_playback_frame]
                self.current_playback_frame += 1
            else:
                self.playback_active = False
                self.current_playback_frame = 0
        elif self.recording:
            self.key_recorder.add_frame(self.keys_pressed)
        self.ship_pos, self.ship_rotation = move_ship(
            self.ship_pos, self.ship_rotation, self.keys_pressed
        )

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.window_width, self.window_height)

        ShaderLib.use(DefaultShader.COLOUR)
        mvp = (
            self.project @ self.view @ ship_transform(self.ship_pos, self.ship_rotation)
        )
        ShaderLib.set_uniform("MVP", mvp)
        self.ship.draw()

        if self.recording:
            Text.render_text("Arial", 10, 18, "Recording", Vec3(1.0, 0.0, 0.0))
        if self.playback_active:
            Text.render_text(
                "Arial",
                10,
                18,
                f"Playback doing frame {self.current_playback_frame}",
                Vec3(1.0, 1.0, 0.0),
            )

    def resizeGL(self, w: int, h: int) -> None:
        self.project = perspective(45.0, w / h, 0.05, 350.0)
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        Text.set_screen_size(self.window_width, self.window_height)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.keys_pressed |= int(GameControls.UP)
        elif key == Qt.Key_Down:
            self.keys_pressed |= int(GameControls.DOWN)
        elif key == Qt.Key_Left:
            self.keys_pressed |= int(GameControls.LEFT)
        elif key == Qt.Key_Right:
            self.keys_pressed |= int(GameControls.RIGHT)
        elif key == Qt.Key_R:
            self.keys_pressed |= int(GameControls.ROTATE)
        elif key == Qt.Key_Space:
            self.key_recorder.set_start_position(self.ship_pos)
            self.recording = not self.recording
        elif key == Qt.Key_P:
            self.playback_active = not self.playback_active
        elif key == Qt.Key_S:
            path, _ = QFileDialog.getSaveFileName(None, "Save Keypresses", ".", "*.kp")
            if path:
                self.key_recorder.save(path)
        elif key == Qt.Key_L:
            path, _ = QFileDialog.getOpenFileName(None, "Load Keypresses", ".", "*.kp")
            if path:
                self.key_recorder.load(path)
        elif key == Qt.Key_Escape:
            self.close()

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.keys_pressed &= ~int(GameControls.UP)
        elif key == Qt.Key_Down:
            self.keys_pressed &= ~int(GameControls.DOWN)
        elif key == Qt.Key_Left:
            self.keys_pressed &= ~int(GameControls.LEFT)
        elif key == Qt.Key_Right:
            self.keys_pressed &= ~int(GameControls.RIGHT)
        elif key == Qt.Key_R:
            self.keys_pressed &= ~int(GameControls.ROTATE)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.ship_timer.stop()
        self.redraw_timer.stop()
        super().closeEvent(event)


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
