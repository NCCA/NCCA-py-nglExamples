#!/usr/bin/env -S uv run --script
"""
Skeletal animation: linear blend skinning vs dual-quaternion skinning (OpenGL).

A 4-bone chain along +y wrapped in a procedurally-built tube. Every bone
twists about its own long axis, which is the one deformation mode where LBS
and DQS visibly disagree: press D to see it.

    D  toggle LBS / DQS
    T  toggle the candy-wrapper twist pose (a fixed 180 degree twist of the
       end bone, the same pose the maths tests pin numerically)
    S  toggle the skeleton overlay
    Space reset the camera, Esc quit

The teaching point (Kavan et al. 2007, "Skinning with Dual Quaternions"):
LBS blends each influence's *matrix* linearly. Averaging two rigid
transforms related by a large twist averages each vertex with a point
closer to the opposite side of the joint, so the cross-section pinches
towards the bone axis -- the classic "candy wrapper" collapse. DQS instead
blends the *rotations* (as unit quaternions, taking the shortest path), so
the cross-section stays rigid through the same twist. Both meshes come from
the same bind-pose geometry and the same per-vertex weights; only the blend
rule differs. See skinning_maths.py for the numpy reference both the CPU
tests and this GLSL shader are transcriptions of.
"""

import argparse
import ctypes
import sys
import time
import traceback
from pathlib import Path

import numpy as np
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
from skinning_maths import (
    bind_pose,
    build_tube_mesh,
    compute_dq_palette,
    compute_palette,
    compute_weights,
    pose_matrices,
)

SKIN_SHADER = "SkinShader"

N_BONES = 4
BONE_LENGTH = 0.5
TUBE_RADIUS = 0.15
TUBE_SEGMENTS = 16
RING_SPACING = 0.25

TWIST_AMPLITUDE_DEG = 70.0
TWIST_SPEED = 0.6
CANDY_WRAPPER_ANGLES = [0.0, 0.0, 0.0, 180.0]  # T key: twist the end bone 180 degrees


def _vertex_buffer(verts, normals, idx, wts) -> np.ndarray:
    """Interleave position/normal/bone-index/bone-weight into one VBO.

    Layout per vertex (10 floats): pos.xyz, normal.xyz, boneIdx.xy, boneWt.xy
    """
    n = len(verts)
    data = np.empty((n, 10), dtype=np.float32)
    data[:, 0:3] = verts
    data[:, 3:6] = normals
    data[:, 6:8] = idx
    data[:, 8:10] = wts
    return data


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Tube-on-a-chain scene with a D/T/S-toggled LBS vs DQS comparison."""

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
        self.setTitle("Skeletal Animation: LBS vs DQS (OpenGL)")

        # --- skeleton / mesh state built once in initializeGL ---
        self.bind_matrices: list = []
        self.inverse_bind: list = []
        self.n_indices: int = 0

        # --- demo state driven by the keyboard ---
        self.use_dqs: bool = False
        self.twist_pose: bool = False
        self.show_skeleton: bool = True
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        chain_length = (N_BONES - 1) * BONE_LENGTH
        self.view = look_at(
            Vec3(0.0, chain_length * 0.5, 2.5),
            Vec3(0.0, chain_length * 0.5, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            SKIN_SHADER,
            str(shader_dir / "SkinVertex.glsl"),
            str(shader_dir / "SkinFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()

        self.bind_matrices, self.inverse_bind = bind_pose(N_BONES, BONE_LENGTH)
        self._build_tube_vao()
        self._build_skeleton_vao()

        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

    def _build_tube_vao(self) -> None:
        verts, normals, faces, bone_origins = build_tube_mesh(
            N_BONES, BONE_LENGTH, TUBE_RADIUS, TUBE_SEGMENTS, RING_SPACING
        )
        idx, wts = compute_weights(verts, bone_origins)
        interleaved = _vertex_buffer(verts, normals, idx, wts)
        self.n_indices = faces.size

        self.tube_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.tube_vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW
        )
        stride = interleaved.shape[1] * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4)
        )
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(
            3, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4)
        )
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(
            4, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8 * 4)
        )

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, gl.GL_STATIC_DRAW
        )
        gl.glBindVertexArray(0)

    def _build_skeleton_vao(self) -> None:
        """A dynamic line-list VBO re-filled every frame with the current
        pose's joint positions (skeleton overlay, drawn with the COLOUR
        shader on GL_LINES, S to toggle)."""
        self.skeleton_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.skeleton_vao)
        self.skeleton_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.skeleton_vbo)
        max_bytes = N_BONES * 3 * 4
        gl.glBufferData(gl.GL_ARRAY_BUFFER, max_bytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # per-frame pose
    # ------------------------------------------------------------------
    def _current_angles(self) -> list:
        if self.twist_pose:
            return CANDY_WRAPPER_ANGLES
        elapsed = time.time() - self._start_time
        return [
            TWIST_AMPLITUDE_DEG * np.sin(elapsed * TWIST_SPEED + 0.6 * i)
            for i in range(N_BONES)
        ]

    def _upload_bone_uniforms(self) -> None:
        pose = pose_matrices(self._current_angles(), BONE_LENGTH)
        palette = compute_palette(pose, self.inverse_bind)

        program = ShaderLib.get_program_id(SKIN_SHADER)

        bones_loc = gl.glGetUniformLocation(program, "bones")
        # Our numpy matrices are row-vector/row-major (translation in row
        # 3); transpose=GL_TRUE tells GL to transpose them into its own
        # column-major storage so the shader's `bones[i] * vec4(v, 1.0)`
        # reproduces the same transform skinning_maths applies as v @ M.
        gl.glUniformMatrix4fv(
            bones_loc, len(palette), gl.GL_TRUE, palette.astype(np.float32)
        )

        dq_palette = compute_dq_palette(palette)
        # Reorder (s, x, y, z) -> (x, y, z, s) [w-last] for the shader.
        dq_flat = np.empty((len(dq_palette) * 2, 4), dtype=np.float32)
        for i, (qr, qd) in enumerate(dq_palette):
            dq_flat[2 * i] = [qr[1], qr[2], qr[3], qr[0]]
            dq_flat[2 * i + 1] = [qd[1], qd[2], qd[3], qd[0]]
        dqs_loc = gl.glGetUniformLocation(program, "boneDQs")
        gl.glUniform4fv(dqs_loc, len(dq_flat), dq_flat)

        return pose

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

    def _load_matrices(self, model: Mat4, global_tx: Mat4) -> None:
        mv = self.view @ global_tx @ model
        ShaderLib.set_uniform("MVP", self.project @ mv)
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        global_tx = self.scene_global_tx()
        model = Mat4()

        ShaderLib.use(SKIN_SHADER)
        pose = self._upload_bone_uniforms()
        self._load_matrices(model, global_tx)
        ShaderLib.set_uniform("lightDirView", 0.4, 0.6, 1.0)
        ShaderLib.set_uniform("colour", 0.85, 0.55, 0.2, 1.0)
        ShaderLib.set_uniform("useDQS", 1 if self.use_dqs else 0)

        gl.glBindVertexArray(self.tube_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        if self.show_skeleton:
            self._draw_skeleton(pose, model, global_tx)

        self._draw_hud()

    def _draw_skeleton(self, pose, model: Mat4, global_tx: Mat4) -> None:
        joints = np.array([m[3, :3] for m in pose], dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.skeleton_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, joints.nbytes, joints)

        gl.glDisable(gl.GL_DEPTH_TEST)
        ShaderLib.use(DefaultShader.COLOUR)
        mv = self.view @ global_tx @ model
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        gl.glBindVertexArray(self.skeleton_vao)
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(joints))
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_hud(self) -> None:
        state = (
            f"[D] mode: {'DQS' if self.use_dqs else 'LBS'}   "
            f"[T] twist pose: {'ON' if self.twist_pose else 'OFF (animating)'}   "
            f"[S] skeleton: {'ON' if self.show_skeleton else 'OFF'}"
        )
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))
        Text.render_text(
            "Arial",
            10,
            45,
            "Twist the end bone 180 degrees: LBS pinches the tube, DQS keeps its radius",
            Vec3(1.0, 1.0, 1.0),
        )

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
        elif key == Qt.Key_D:
            self.use_dqs = not self.use_dqs
        elif key == Qt.Key_T:
            self.twist_pose = not self.twist_pose
        elif key == Qt.Key_S:
            self.show_skeleton = not self.show_skeleton
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

    timer = QTimer()
    timer.timeout.connect(window.update)
    timer.start(16)

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
