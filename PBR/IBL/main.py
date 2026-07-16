#!/usr/bin/env -S uv run --script
"""
Image-Based Lighting (IBL) -- ambient environment lighting for PBR (OpenGL).

Completes the PBR story started in ``PBR/SimplePBR`` (direct-only Cook-Torrance
lighting) by replacing its flat ``vec3(0.03) * albedo * ao`` ambient hack with
real environment lighting: a diffuse irradiance map and a Karis split-sum BRDF
LUT, both precomputed once in numpy at startup and cached to ``.npy`` files
next to this script (see ``ibl_precompute.py`` -- that module is pure numpy,
no GL/Qt, so it's the one part of this demo that's unit-tested headlessly).

The environment itself is the same procedural sky as ``SkyBoxEnvMap``
(``cubemap_gen.py``, copied here -- demo folders stay standalone).

**Cut stage**: the brief's four precompute stages are (1) irradiance map,
(2) roughness-mipped GGX-prefiltered specular chain, (3) BRDF LUT, (4) the
shader ambient term. Stage 2 was cut for time -- see ``ibl_precompute.py``'s
module docstring and ``shaders/IBLFragment.glsl``'s "CUT STAGE" comment for
the exact consequence: specular reflections stay sharp at all roughness
values (a single unblurred cubemap sample) instead of blurring out with mip
level. Diffuse IBL and the BRDF LUT are real.

    I    toggle IBL ambient vs SimplePBR's flat direct-only ambient (the money shot)
    E    cycle debug view: off / irradiance-as-skybox / BRDF LUT (corner overlay)
    LMB  rotate    RMB pan    wheel zoom    Space reset camera    Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from cubemap_gen import FACE_ORDER, generate_cubemap_faces
from ibl_precompute import load_or_compute_brdf_lut, load_or_compute_irradiance
from ncca.ngl import Mat3, Mat4, Prims, Vec3, Vec3Array, logger, look_at, perspective
from ncca.ngl.opengl import Primitives, PySideEventHandlingMixin, ShaderLib, Text
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

PBR_SHADER = "IBLPbr"
SKYBOX_SHADER = "SkyboxShader"
LUT_QUAD_SHADER = "LUTQuad"

_GL_FACE_TARGET = {
    "+x": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    "-x": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    "+y": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    "-y": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    "+z": gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    "-z": gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
}

DEBUG_VIEWS = ("off", "irradiance skybox", "BRDF LUT overlay")


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """PBR sphere grid lit by direct lights + (toggleable) image-based ambient."""

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
        self.setTitle("IBL -- Image Based Lighting (OpenGL)")

        self.eye = Vec3(0, 5, 13)
        self.use_ibl: bool = True
        self.debug_view: int = 0

        self.env_cubemap: int = 0
        self.irradiance_cubemap: int = 0
        self.brdf_lut_texture: int = 0
        self._empty_vao: int = 0

    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.05, 0.05, 0.07, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            PBR_SHADER,
            vert=str(shader_dir / "IBLVertex.glsl"),
            frag=str(shader_dir / "IBLFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            SKYBOX_SHADER,
            str(shader_dir / "SkyboxVertex.glsl"),
            str(shader_dir / "SkyboxFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            LUT_QUAD_SHADER,
            str(shader_dir / "LUTQuadVertex.glsl"),
            str(shader_dir / "LUTQuadFragment.glsl"),
        )
        if not ok:
            logger.error("Error loading shaders")
            self.close()

        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 0.5, 40)
        Primitives.create(Prims.TRIANGLE_PLANE, "floor", 20, 20, 10, 10, Vec3(0, 1, 0))
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent.parent / "font" / "Arial.ttf"), 20
        )

        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))

        self.light_positions = Vec3Array(
            [
                Vec3(-10.0, 4.0, -10.0),
                Vec3(10.0, 4.0, -10.0),
                Vec3(-10.0, 4.0, 10.0),
                Vec3(10.0, 4.0, 10.0),
            ]
        )
        self.light_colors = Vec3Array([Vec3(300.0, 300.0, 300.0) for _ in range(4)])

        self._build_environment()

        # a bound-but-empty VAO: core profile requires *some* VAO bound for
        # glDrawArrays even when the vertex shader has no attribute inputs
        # (the LUT debug quad is generated entirely from gl_VertexID).
        self._empty_vao = gl.glGenVertexArrays(1)

    # ------------------------------------------------------------------
    def _build_environment(self) -> None:
        """Generate the procedural sky, then precompute (or load cached)
        the irradiance map and BRDF LUT from it, uploading all three as GL
        textures."""
        faces_u8 = generate_cubemap_faces()
        self.env_cubemap = self._upload_cubemap_u8(faces_u8)

        faces_f = [f[..., :3].astype(np.float64) / 255.0 for f in faces_u8]
        irradiance_faces = load_or_compute_irradiance(faces_f)  # (6, s, s, 3)
        self.irradiance_cubemap = self._upload_cubemap_f32(irradiance_faces)

        lut = load_or_compute_brdf_lut().astype(np.float32)  # (S, S, 2)
        self.brdf_lut_texture = self._upload_lut(lut)

    @staticmethod
    def _upload_cubemap_u8(faces: list[np.ndarray]) -> int:
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, tex)
        for face_name, pixels in zip(FACE_ORDER, faces, strict=False):
            size = pixels.shape[0]
            gl.glTexImage2D(
                _GL_FACE_TARGET[face_name],
                0,
                gl.GL_RGBA8,
                size,
                size,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                np.ascontiguousarray(pixels),
            )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
        )
        for wrap in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T, gl.GL_TEXTURE_WRAP_R):
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, wrap, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)
        return tex

    @staticmethod
    def _upload_cubemap_f32(faces: np.ndarray) -> int:
        """``faces``: (6, size, size, 3) float64 in [0, 1], FACE_ORDER order."""
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, tex)
        for face_name, i in zip(FACE_ORDER, range(6), strict=False):
            pixels = np.ascontiguousarray(faces[i].astype(np.float32))
            size = pixels.shape[0]
            gl.glTexImage2D(
                _GL_FACE_TARGET[face_name],
                0,
                gl.GL_RGB16F,
                size,
                size,
                0,
                gl.GL_RGB,
                gl.GL_FLOAT,
                pixels,
            )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
        )
        for wrap in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T, gl.GL_TEXTURE_WRAP_R):
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, wrap, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)
        return tex

    @staticmethod
    def _upload_lut(lut: np.ndarray) -> int:
        """``lut``: (size, size, 2) float32 RG split-sum A/B table."""
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        size = lut.shape[0]
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RG16F,
            size,
            size,
            0,
            gl.GL_RG,
            gl.GL_FLOAT,
            np.ascontiguousarray(lut),
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _draw_sphere_grid(self, global_tx: Mat4) -> None:
        """The 7x7 sphere grid: rows sweep metallic 0..1, columns sweep
        roughness 0.05..1, all under the arcball rotation `global_tx`."""
        for row in range(7):
            for col in range(7):
                metallic = row / 6.0
                roughness = max(0.05, col / 6.0)
                model = Mat4()
                model[3, 0] = (col - 3) * 1.3
                model[3, 1] = (row - 3) * 1.3
                model = global_tx @ model
                MV = self.view @ model
                MVP = self.project @ MV
                # IBLFragment.glsl lights in world space (V = camPos -
                # WorldPos, N indexes the irradiance cube directly), so the
                # normal matrix comes from the model alone -- MV would drag
                # the normals into view space and the lighting would rotate
                # with the camera.
                normal_matrix = Mat3.from_mat4(model).inverse().transposed()

                ShaderLib.set_uniform("albedo", 0.5, 0.0, 0.0)
                ShaderLib.set_uniform("metallic", metallic)
                ShaderLib.set_uniform("roughness", roughness)
                ShaderLib.set_uniform("MVP", MVP)
                ShaderLib.set_uniform("normalMatrix", normal_matrix)
                ShaderLib.set_uniform("M", model)
                Primitives.draw("sphere")

    # ------------------------------------------------------------------
    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_tx = self.scene_global_tx()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.irradiance_cubemap)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.brdf_lut_texture)

        ShaderLib.use(PBR_SHADER)
        ShaderLib.set_uniform("irradianceMap", 0)
        ShaderLib.set_uniform("envMap", 1)
        ShaderLib.set_uniform("brdfLUT", 2)
        ShaderLib.set_uniform("useIBL", self.use_ibl)
        ShaderLib.set_uniform("ao", 1.0)
        ShaderLib.set_uniform("camPos", self.eye)
        for i in range(4):
            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_positions[i])
            ShaderLib.set_uniform(f"lightColors[{i}]", self.light_colors[i])

        self._draw_sphere_grid(global_tx)

        self._draw_skybox()
        self._draw_debug_overlay()
        self._draw_hud()

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

    def _draw_skybox(self) -> None:
        """The background sky, or (debug view 1) the low-res irradiance map
        wrapped over the same cube -- makes the diffuse convolution visible
        as a blurred version of the sky."""
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)
        ShaderLib.use(SKYBOX_SHADER)
        texture = self.irradiance_cubemap if self.debug_view == 1 else self.env_cubemap
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture)
        ShaderLib.set_uniform("skybox", 0)
        view_rotation_only = self.view.copy()
        view_rotation_only[3, 0] = 0.0
        view_rotation_only[3, 1] = 0.0
        view_rotation_only[3, 2] = 0.0
        ShaderLib.set_uniform("MVP", self.project @ view_rotation_only)
        Primitives.draw("cube")
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

    def _draw_debug_overlay(self) -> None:
        """Debug view 2: the BRDF LUT drawn as a small quad, bottom-right
        corner, red = A (scale), green = B (bias)."""
        if self.debug_view != 2:
            return
        gl.glDisable(gl.GL_DEPTH_TEST)
        ShaderLib.use(LUT_QUAD_SHADER)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.brdf_lut_texture)
        ShaderLib.set_uniform("lut", 0)
        gl.glBindVertexArray(self._empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_hud(self) -> None:
        ibl_state = "ON" if self.use_ibl else "OFF (direct-only)"
        debug_state = DEBUG_VIEWS[self.debug_view]
        Text.render_text(
            "Arial",
            10,
            20,
            f"[I] IBL: {ibl_state}   [E] debug: {debug_state}",
            Vec3(1.0, 1.0, 1.0),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_I:
            self.use_ibl = not self.use_ibl
        elif key == Qt.Key_E:
            self.debug_view = (self.debug_view + 1) % len(DEBUG_VIEWS)
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

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
