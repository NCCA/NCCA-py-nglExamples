#!/usr/bin/env -S uv run --script
"""
Gimbal lock: Euler vs quaternion orientation, side by side (OpenGL).

LEFT half is the classic three-ring Euler rig: an outer ring spun about
world Z (yaw), carrying a ring spun about its own Y (pitch), carrying a
ring spun about its own X (roll), carrying a little cube aeroplane. RIGHT
half drives an identical aeroplane straight from a quaternion built from
the same three angles.

    X/Y/Z   nudge roll/pitch/yaw by +5 degrees (Shift for -5)
    G       scripted "watch the DOF vanish": pitch ramps to 90, then yaw
            wobbles while roll sits still -- on the left the aeroplane
            barely reacts to yaw any more, because roll already owns that
            axis; on the right the quaternion just keeps working
    Space   reset all three angles to zero
    Esc     quit

The teaching point: nest three single-axis rotations and at pitch = +/-90
the outer and inner rings' axes line up, so one whole degree of freedom
disappears -- turning the yaw ring now does exactly what turning the roll
ring does. That is gimbal lock. It is a property of the three-angle
*representation*, not of the underlying rotation, which is why the
quaternion on the right never notices: it never decomposes anything into
three sequential single-axis turns in the first place.

See GimbalLock/rotation_maths.py for the maths this rests on (and its
docstring for the exact composition order -- get that order out of step
with how the rings are nested here and the picture stops matching the
argument).
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Vec3, logger, look_at, perspective
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
from rotation_maths import euler_to_quat, is_gimbal_locked, lost_dof_axis

ANGLE_STEP = 5.0
RING_TUBE_RADIUS = 0.035
RING_RADII = {"z": 1.0, "y": 0.75, "x": 0.5}  # outer to inner
RING_COLOUR = {
    "x": Vec3(0.85, 0.2, 0.2),
    "y": Vec3(0.2, 0.75, 0.25),
    "z": Vec3(0.25, 0.4, 0.9),
}
ARROW_COLOUR = Vec3(0.9, 0.85, 0.2)

# scripted "G" lock animation
ANIM_TICK_MS = 16
ANIM_PITCH_STEP = 3.0
ANIM_YAW_AMPLITUDE = 35.0
ANIM_YAW_PERIOD_S = 2.0
ANIM_YAW_DURATION_S = 4.0


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Split-screen Euler-rig vs quaternion gimbal-lock demo."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, -6),
        )
        self.view: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Gimbal Lock: Euler vs Quaternion (OpenGL)")

        # the three driven Euler angles, in degrees
        self.rx: float = 0.0
        self.ry: float = 0.0
        self.rz: float = 0.0

        # scripted "G" animation state
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_step)
        self._anim_phase: str | None = None
        self._anim_elapsed: float = 0.0

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.18, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1.5, 6), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        Primitives.load_default_primitives()
        # one ring mesh, reused at three radii/orientations/colours
        Primitives.create(Prims.TORUS, "ring", RING_TUBE_RADIUS, 1.0, 24, 48)

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 18
        )

    # ------------------------------------------------------------------
    # matrix helpers
    # ------------------------------------------------------------------
    def _camera_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _load_matrices(self, model: Mat4, global_tx: Mat4, project: Mat4) -> None:
        mv = self.view @ global_tx @ model
        mvp = project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def _draw_ring(self, axis: str, tx: Mat4, project: Mat4, global_tx: Mat4) -> None:
        """Draw one ring, scaled to its radius and oriented so its hole
        points down the axis it represents.

        The torus mesh lies flat with its hole facing +Y, so the X and Z
        rings need a *static* extra spin to point their hole the right
        way -- this is a fixed visual constant, nothing to do with rx/ry/rz.
        """
        orient = {
            "x": Mat4().rotate_z(90.0),
            "y": Mat4(),
            "z": Mat4().rotate_x(90.0),
        }[axis]
        scale = Mat4.scale(RING_RADII[axis], RING_RADII[axis], RING_RADII[axis])
        model = tx @ scale @ orient
        ShaderLib.set_uniform("Colour", *RING_COLOUR[axis], 1.0)
        self._load_matrices(model, global_tx, project)
        Primitives.draw("ring")

    def _draw_arrow(self, tx: Mat4, project: Mat4, global_tx: Mat4) -> None:
        """A small cube aeroplane: fuselage, wings, tailfin, all cubes."""
        parts = (
            Mat4.scale(0.12, 0.12, 0.55),  # fuselage, nose along +Z
            Mat4.translate(0.0, 0.0, 0.35) @ Mat4.scale(0.03, 0.2, 0.03),
            Mat4.translate(0.0, 0.03, -0.35) @ Mat4.scale(0.03, 0.16, 0.12),  # tailfin
            Mat4.translate(0.4, 0.0, -0.2) @ Mat4.scale(0.4, 0.02, 0.14),  # wings
        )
        ShaderLib.set_uniform("Colour", *ARROW_COLOUR, 1.0)
        for part in parts:
            model = tx @ part
            self._load_matrices(model, global_tx, project)
            Primitives.draw("cube")

    # ------------------------------------------------------------------
    # scenes
    # ------------------------------------------------------------------
    def _draw_euler_scene(self, project: Mat4, global_tx: Mat4) -> None:
        """Nested-ring rig: each ring's matrix carries its parent ring's,
        matching rotation_maths.euler_to_mat's X-then-Y-then-Z compose
        order (Z outermost, carrying Y, carrying X, carrying the arrow)."""
        outer_tx = Mat4().rotate_z(self.rz)
        mid_tx = outer_tx @ Mat4().rotate_y(self.ry)
        inner_tx = mid_tx @ Mat4().rotate_x(self.rx)

        ShaderLib.use(DefaultShader.DIFFUSE)
        self._draw_ring("z", outer_tx, project, global_tx)
        self._draw_ring("y", mid_tx, project, global_tx)
        self._draw_ring("x", inner_tx, project, global_tx)
        self._draw_arrow(inner_tx, project, global_tx)

    def _draw_quat_scene(self, project: Mat4, global_tx: Mat4) -> None:
        """The same arrow, driven directly by a quaternion built from the
        same (rx, ry, rz) -- no rings, no gimbal, no lock."""
        quat = euler_to_quat(self.rx, self.ry, self.rz)
        model = quat.to_mat4()
        ShaderLib.use(DefaultShader.DIFFUSE)
        self._draw_arrow(model, project, global_tx)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        half_w = max(1, self.window_width // 2)
        right_w = max(1, self.window_width - half_w)
        global_tx = self._camera_tx()

        gl.glEnable(gl.GL_SCISSOR_TEST)

        gl.glViewport(0, 0, half_w, self.window_height)
        gl.glScissor(0, 0, half_w, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        left_project = perspective(45.0, half_w / self.window_height, 0.01, 350.0)
        self._draw_euler_scene(left_project, global_tx)

        gl.glViewport(half_w, 0, right_w, self.window_height)
        gl.glScissor(half_w, 0, right_w, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        right_project = perspective(45.0, right_w / self.window_height, 0.01, 350.0)
        self._draw_quat_scene(right_project, global_tx)

        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, self.window_width, self.window_height)
        self._draw_hud(half_w)

    def _draw_hud(self, half_w: int) -> None:
        locked = is_gimbal_locked(self.ry)
        pose = f"roll(X) {self.rx:6.1f}  pitch(Y) {self.ry:6.1f}  yaw(Z) {self.rz:6.1f}"
        Text.render_text("Arial", 10, 20, "EULER RIG (nested rings)", Vec3(1, 1, 1))
        Text.render_text("Arial", 10, 42, pose, Vec3(1, 1, 1))
        if locked:
            axis = lost_dof_axis(self.rx, self.ry, self.rz)
            Text.render_text(
                "Arial", 10, 64, f"GIMBAL LOCKED -- lost DOF: {axis}", Vec3(1, 0.3, 0.3)
            )
        Text.render_text(
            "Arial",
            half_w + 10,
            20,
            "QUATERNION (same angles)",
            Vec3(1, 1, 1),
        )
        Text.render_text(
            "Arial",
            half_w + 10,
            42,
            "no rings, no lock, ever",
            Vec3(0.6, 1.0, 0.6),
        )
        Text.render_text(
            "Arial",
            10,
            self.window_height - 20,
            "[X/Y/Z] +5 deg  [Shift+X/Y/Z] -5 deg  [G] watch the DOF vanish  [Space] reset",
            Vec3(0.8, 0.8, 0.8),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # scripted "G" lock animation
    # ------------------------------------------------------------------
    def _start_lock_animation(self) -> None:
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self._anim_phase = "pitch"
        self._anim_elapsed = 0.0
        self._anim_timer.start(ANIM_TICK_MS)

    def _animate_step(self) -> None:
        dt = ANIM_TICK_MS / 1000.0
        if self._anim_phase == "pitch":
            self.ry = min(90.0, self.ry + ANIM_PITCH_STEP)
            if self.ry >= 90.0:
                self._anim_phase = "yaw"
                self._anim_elapsed = 0.0
        elif self._anim_phase == "yaw":
            self._anim_elapsed += dt
            phase = 2.0 * math.pi * self._anim_elapsed / ANIM_YAW_PERIOD_S
            self.rz = ANIM_YAW_AMPLITUDE * math.sin(phase)
            if self._anim_elapsed >= ANIM_YAW_DURATION_S:
                self._anim_phase = None
                self._anim_timer.stop()
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        negative = bool(event.modifiers() & Qt.ShiftModifier)
        step = -ANGLE_STEP if negative else ANGLE_STEP
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_X:
            self.rx += step
        elif key == Qt.Key_Y:
            self.ry += step
        elif key == Qt.Key_Z:
            self.rz += step
        elif key == Qt.Key_G:
            self._start_lock_animation()
        elif key == Qt.Key_Space:
            self.rx = self.ry = self.rz = 0.0
            self._anim_timer.stop()
            self._anim_phase = None
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
