#!/usr/bin/env -S uv run --script
"""
Cubemaps: skybox rendering plus reflection / refraction on a teapot (OpenGL).

A procedural sky (see cubemap_gen.py -- no image assets needed) is uploaded
as a GL_TEXTURE_CUBE_MAP and drawn as a skybox behind an env-mapped teapot.

    M    cycle teapot shading mode: reflect / refract / fresnel mix / diffuse
    +/-  increase / decrease the index of refraction (refract & fresnel mix)
    LMB  rotate    RMB pan    wheel zoom    Space reset camera    Esc quit

Teaching points:
    1. A skybox is a unit cube sampled with a *direction*, not a UV: the
       vertex shader strips translation from the view matrix
       (P * mat4(mat3(V))) and forces depth to the far plane
       (gl_Position = pos.xyww) so the skybox is drawn as if infinitely far
       away and never occludes real geometry.
    2. Despite most tutorials drawing the skybox *first* "to save fill
       rate", drawing opaque geometry first and the skybox *last* lets
       early-z reject the (expensive, full-screen) skybox fragments that
       sit behind already-drawn opaque pixels -- the classic tutorial order
       is actually the slower one on modern hardware.
    3. reflect()/refract() need world-space position, normal and camera
       position -- not the usual view-space lighting setup -- because the
       cubemap itself is sampled in world space.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from cubemap_gen import FACE_ORDER, generate_cubemap_faces
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

SKYBOX_SHADER = "SkyboxShader"
ENVMAP_SHADER = "EnvMapShader"

_GL_FACE_TARGET = {
    "+x": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    "-x": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    "+y": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    "-y": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    "+z": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    "-z": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
}

# (label, mode index) cycled with M
MODES = (
    ("reflect", 0),
    ("refract", 1),
    ("fresnel mix", 2),
    ("plain diffuse", 3),
)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Skybox + reflective/refractive teapot scene."""

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
        self.setTitle("SkyBox / EnvMap (OpenGL)")

        self.eye = Vec3(0.0, 1.5, 6.0)
        self.mode_index: int = 0
        self.ior: float = 1.52  # glass
        self.cube_texture: int = 0

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(self.eye, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0))

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            SKYBOX_SHADER,
            str(shader_dir / "SkyboxVertex.glsl"),
            str(shader_dir / "SkyboxFragment.glsl"),
        ) or not ShaderLib.load_shader(
            ENVMAP_SHADER,
            str(shader_dir / "EnvMapVertex.glsl"),
            str(shader_dir / "EnvMapFragment.glsl"),
        ):
            print("error loading shaders")
            self.close()

        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

        self._create_cube_map()

    def _create_cube_map(self) -> None:
        """Generate and upload the procedural sky as a GL_TEXTURE_CUBE_MAP."""
        faces = generate_cubemap_faces()
        self.cube_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.cube_texture)
        for face_name, face_pixels in zip(FACE_ORDER, faces):
            target = _GL_FACE_TARGET[face_name]
            size = face_pixels.shape[0]
            gl.glTexImage2D(
                target,
                0,
                gl.GL_RGBA8,
                size,
                size,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                np.ascontiguousarray(face_pixels),
            )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE
        )
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

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

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.cube_texture)

        # ---- 1. opaque pass: env-mapped teapot (depth write on) ----
        ShaderLib.use(ENVMAP_SHADER)
        ShaderLib.set_uniform("skybox", 0)
        ShaderLib.set_uniform("camPos", self.eye)
        ShaderLib.set_uniform("lightDir", Vec3(0.4, 1.0, 0.6))
        ShaderLib.set_uniform("mode", MODES[self.mode_index][1])
        ShaderLib.set_uniform("ior", self.ior)
        model = global_tx
        MV = self.view @ model
        ShaderLib.set_uniform("M", model)
        ShaderLib.set_uniform("MVP", self.project @ MV)
        normal_matrix = Mat3.from_mat4(model).inverse().transposed()
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        Primitives.draw("teapot")

        # ---- 2. skybox pass: drawn LAST so early-z rejects fragments
        # behind already-shaded opaque geometry (see module docstring) ----
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)
        ShaderLib.use(SKYBOX_SHADER)
        ShaderLib.set_uniform("skybox", 0)
        view_rotation_only = self.view.copy()
        view_rotation_only[3, 0] = 0.0
        view_rotation_only[3, 1] = 0.0
        view_rotation_only[3, 2] = 0.0
        ShaderLib.set_uniform("MVP", self.project @ view_rotation_only)
        Primitives.draw("cube")
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        self._draw_hud()

    def _draw_hud(self) -> None:
        label, _ = MODES[self.mode_index]
        state = f"[M] mode: {label}   [+/-] IOR: {self.ior:.2f}"
        Text.render_text("Arial", 10, 20, state, Vec3(1.0, 1.0, 1.0))

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
        elif key == Qt.Key_M:
            self.mode_index = (self.mode_index + 1) % len(MODES)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.ior = min(2.5, self.ior + 0.05)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.ior = max(1.01, self.ior - 0.05)
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
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv
    if "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
