#!/usr/bin/env -S uv run --script
"""
Transform hierarchies: a robot arm built from a scene graph (OpenGL).

Every segment of the arm is a ``scene_graph.Node`` holding nothing but a
local matrix, a mesh name and a colour. The whole scene is one depth-first
``root.walk()`` call: each node's world matrix is its parent's world matrix
combined with its own local one, so rotating the shoulder swings the elbow,
wrist and both fingers with it for free -- nobody hand-propagates anything.

    base -> upper_arm -> lower_arm -> claw_left
                                    -> claw_right

Controls:
    1..5   select joint (base, upper arm, lower arm, left claw, right claw)
    Left/Right  rotate the selected joint +/-5 degrees
    P      toggle a canned sine-driven waving animation
    Space  reset to the resting pose, Esc quit

The point of the demo is in scene_graph.py, not here: this file just turns
joint angles into local matrices and draws whatever ``root.walk()`` hands
back.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
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
from scene_graph import Node

# ----------------------------------------------------------------------
# Arm dimensions and joint set. Angles live in a flat dict keyed by node
# name -- the "single source of truth" the brief asks for -- and the graph
# is rebuilt from it every frame in _build_scene().
# ----------------------------------------------------------------------
BASE_HEIGHT = 0.4
UPPER_LEN = 1.2
LOWER_LEN = 1.0
FINGER_LEN = 0.4
FINGER_SPREAD = 0.18

JOINT_NAMES = ["base", "upper_arm", "lower_arm", "claw_left", "claw_right"]
JOINT_LABELS = {
    "base": "1 base (yaw)",
    "upper_arm": "2 upper arm (shoulder)",
    "lower_arm": "3 lower arm (elbow)",
    "claw_left": "4 left claw",
    "claw_right": "5 right claw",
}
DEFAULT_ANGLES = {
    "base": 0.0,
    "upper_arm": -25.0,
    "lower_arm": 40.0,
    "claw_left": 15.0,
    "claw_right": -15.0,
}
JOINT_STEP = 5.0

# Per-node (offset, scale) applied only for drawing -- these are NOT part
# of a node's local matrix, so they never leak into a child's placement.
# A node's local matrix is purely "where is this joint's pivot", the mesh
# offset is purely "how do I draw a box hanging off that pivot".
MESH_TRANSFORM = {
    "base": ((0.0, BASE_HEIGHT * 0.5, 0.0), (1.2, BASE_HEIGHT, 1.2)),
    "upper_arm": ((0.0, UPPER_LEN * 0.5, 0.0), (0.3, UPPER_LEN, 0.3)),
    "lower_arm": ((0.0, LOWER_LEN * 0.5, 0.0), (0.25, LOWER_LEN, 0.25)),
    "claw_left": ((0.0, FINGER_LEN * 0.5, 0.0), (0.12, FINGER_LEN, 0.12)),
    "claw_right": ((0.0, FINGER_LEN * 0.5, 0.0), (0.12, FINGER_LEN, 0.12)),
}
COLOURS = {
    "base": (0.55, 0.55, 0.6),
    "upper_arm": (0.85, 0.55, 0.15),
    "lower_arm": (0.85, 0.7, 0.15),
    "claw_left": (0.7, 0.15, 0.15),
    "claw_right": (0.15, 0.6, 0.7),
}


def _build_scene(angles: dict[str, float]) -> Node:
    """Rebuild the whole tree from the current joint angles.

    Nothing is cached: this runs every frame, so a rotated joint is simply
    a different local matrix next time this function is called. That is
    the entire animation system -- there is no separate "pose" object.
    """
    base = Node(
        "base", local=Mat4.rotate_y(angles["base"]), mesh="cube", colour=COLOURS["base"]
    )
    upper_arm = base.add(
        Node(
            "upper_arm",
            local=Mat4.translate(0.0, BASE_HEIGHT, 0.0)
            @ Mat4.rotate_z(angles["upper_arm"]),
            mesh="cube",
            colour=COLOURS["upper_arm"],
        )
    )
    lower_arm = upper_arm.add(
        Node(
            "lower_arm",
            local=Mat4.translate(0.0, UPPER_LEN, 0.0)
            @ Mat4.rotate_z(angles["lower_arm"]),
            mesh="cube",
            colour=COLOURS["lower_arm"],
        )
    )
    lower_arm.add(
        Node(
            "claw_left",
            local=Mat4.translate(-FINGER_SPREAD, LOWER_LEN, 0.0)
            @ Mat4.rotate_z(angles["claw_left"]),
            mesh="cube",
            colour=COLOURS["claw_left"],
        )
    )
    lower_arm.add(
        Node(
            "claw_right",
            local=Mat4.translate(FINGER_SPREAD, LOWER_LEN, 0.0)
            @ Mat4.rotate_z(angles["claw_right"]),
            mesh="cube",
            colour=COLOURS["claw_right"],
        )
    )
    return base


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Robot arm driven entirely by scene_graph.Node.walk()."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, -1, -6),
        )
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("SceneGraph: transform hierarchies (OpenGL)")

        # --- demo state driven by the keyboard ---
        self.angles: dict[str, float] = dict(DEFAULT_ANGLES)
        self.selected: int = 0  # index into JOINT_NAMES
        self.animating: bool = False
        self.anim_t: float = 0.0

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.14, 0.14, 0.16, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 2.0, 6.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 2.0, 3.0, 2.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )
        self.startTimer(16)

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
        global_tx = self.scene_global_tx()

        ShaderLib.use(DefaultShader.DIFFUSE)
        root = _build_scene(self.angles)
        for node, world in root.walk():
            offset, scale = MESH_TRANSFORM[node.name]
            # mesh-only placement: scale the unit primitive then shift it
            # out from the joint pivot -- deliberately NOT folded into
            # node.local, so it never affects where children attach.
            mesh_tx = Mat4.translate(*offset) @ Mat4.scale(*scale)
            model = world @ mesh_tx
            MV = self.view @ global_tx @ model
            ShaderLib.set_uniform("MVP", self.project @ MV)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(MV).inverse().transposed()
            )
            ShaderLib.set_uniform("Colour", *node.colour, 1.0)
            Primitives.draw(node.mesh)

        self._draw_hud()

    def _draw_hud(self) -> None:
        selected_name = JOINT_NAMES[self.selected]
        Text.render_text(
            "Arial",
            10,
            20,
            f"joint {self.selected + 1}/5: {JOINT_LABELS[selected_name]}"
            f"   angle {self.angles[selected_name]:.0f} deg",
            Vec3(1.0, 1.0, 1.0),
        )
        Text.render_text(
            "Arial",
            10,
            45,
            "[1-5] select joint  [Left/Right] rotate +/-5deg  "
            f"[P] animate {'ON' if self.animating else 'OFF'}  [Space] reset",
            Vec3(1.0, 1.0, 1.0),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # animation
    # ------------------------------------------------------------------
    def timerEvent(self, event) -> None:
        if self.animating:
            self.anim_t += 0.05
            # a canned wave: shoulder and elbow bob, both claws open/close
            # in sync -- purely for show, driven by sin() not user input.
            self.angles["base"] = 25.0 * math.sin(self.anim_t * 0.5)
            self.angles["upper_arm"] = -25.0 + 15.0 * math.sin(self.anim_t)
            self.angles["lower_arm"] = 40.0 + 20.0 * math.sin(self.anim_t * 1.3)
            claw = 15.0 + 10.0 * math.sin(self.anim_t * 2.0)
            self.angles["claw_left"] = claw
            self.angles["claw_right"] = -claw
            self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif Qt.Key_1 <= key <= Qt.Key_5:
            self.selected = key - Qt.Key_1
        elif key == Qt.Key_Left:
            name = JOINT_NAMES[self.selected]
            self.angles[name] -= JOINT_STEP
        elif key == Qt.Key_Right:
            name = JOINT_NAMES[self.selected]
            self.angles[name] += JOINT_STEP
        elif key == Qt.Key_P:
            self.animating = not self.animating
            self.anim_t = 0.0
        elif key == Qt.Key_Space:
            self.angles = dict(DEFAULT_ANGLES)
            self.animating = False
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
