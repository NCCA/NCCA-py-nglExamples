#!/usr/bin/env -S uv run --script
"""HDRI image-based lighting (OpenGL). See README.md."""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from exr_loader import load_equirect_hdr
from ncca.ngl import (
    Mat3,
    Mat4,
    Transform,
    Vec3,
    Vec3Array,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.opengl import (
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

SHADER_DIR = Path(__file__).resolve().parent / "shaders"
HDRI_PATH = (
    Path(__file__).resolve().parent / "images" / "historic_cloister_passage_1k.exr"
)

ENV_SIZE, IRRADIANCE_SIZE, PREFILTER_SIZE, PREFILTER_MIPS, LUT_SIZE = (
    512,
    32,
    128,
    5,
    512,
)

# Cycled by the `E` key to prove out each bake stage visually; see
# _draw_skybox and _draw_debug_overlay.
DEBUG_VIEWS = ("off", "irradiance", "prefilter mip 2", "brdf lut")

# The six view matrices that look out the six cube faces from the origin
# (LearnOpenGL "Diffuse irradiance"). Order matches GL_TEXTURE_CUBE_MAP_POSITIVE_X..+5.
_CAPTURE_PROJECTION = perspective(90.0, 1.0, 0.1, 10.0)
_CAPTURE_VIEWS = [
    look_at(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
    look_at(Vec3(0, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, -1)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, -1, 0)),
]


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """
    The main window for the HDRI image-based lighting demo.

    Inherits from QOpenGLWindow to provide a canvas for OpenGL rendering within a PySide6 GUI.
    It handles user input (mouse, keyboard) for camera control and manages the OpenGL context.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initializes the main window and sets up default scene parameters.
        """
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )  # --- Camera and Transformation Attributes ---
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height
        self.setTitle("HDRI Image-Based Lighting (OpenGL)")
        self.transform = Transform()

        self.env_cubemap: int = 0
        self.irradiance_map: int = 0
        self.prefilter_map: int = 0
        self.brdf_lut: int = 0
        self._capture_fbo: int = 0
        self._capture_rbo: int = 0
        self._capture_rbo_size: int = 0
        self._empty_vao: int = 0
        self.debug_view: int = 0
        self.use_ibl: bool = True
        self.eye: Vec3 = Vec3(0, 0, 30)

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and bake the
        environment cubemap out of the source HDRI.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        gl.glClearColor(0.05, 0.05, 0.07, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.view = look_at(self.eye, to, up)
        self.project = perspective(45.0, 1024.0 / 720.0, 0.05, 350.0)

        ok = ShaderLib.load_shader(
            "equirect2cube",
            vert=str(SHADER_DIR / "CubeVertex.glsl"),
            frag=str(SHADER_DIR / "Equirect2CubeFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "skybox",
            vert=str(SHADER_DIR / "SkyboxVertex.glsl"),
            frag=str(SHADER_DIR / "SkyboxFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "irradiance",
            vert=str(SHADER_DIR / "CubeVertex.glsl"),
            frag=str(SHADER_DIR / "IrradianceFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "prefilter",
            vert=str(SHADER_DIR / "CubeVertex.glsl"),
            frag=str(SHADER_DIR / "PrefilterFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "brdf",
            vert=str(SHADER_DIR / "BRDFVertex.glsl"),
            frag=str(SHADER_DIR / "BRDFFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "lutquad",
            vert=str(SHADER_DIR / "LUTQuadVertex.glsl"),
            frag=str(SHADER_DIR / "LUTQuadFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            "pbr",
            vert=str(SHADER_DIR / "PBRVertex.glsl"),
            frag=str(SHADER_DIR / "PBRFragment.glsl"),
        )
        if not ok:
            logger.error("Error loading shaders")
            self.close()

        # "cube" and "teapot" are both mesh defaults -- Prims.CUBE is not a
        # parametric Primitives.create() type (that raises ValueError and would
        # abort initializeGL), so load them from the bundled default meshes.
        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent.parent / "font" / "Arial.ttf"), 20
        )

        # Empty VAO for the gl_VertexID-generated fullscreen triangle/quad
        # draws (BRDF LUT bake, LUT debug overlay) -- core profile still
        # requires *some* VAO bound even when no attributes are pulled.
        self._empty_vao = gl.glGenVertexArrays(1)

        # Capture FBO used by every bake stage (equirect->cube here; irradiance,
        # prefilter and LUT below reuse this same FBO, resized per stage by
        # _resize_capture_depth).
        self._capture_fbo = gl.glGenFramebuffers(1)
        self._capture_rbo = gl.glGenRenderbuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._capture_fbo)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self._capture_rbo,
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self._resize_capture_depth(ENV_SIZE)

        self.equirect_tex = self._upload_equirect()
        self.env_cubemap = self._new_cubemap(ENV_SIZE, mip=True)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.equirect_tex)
        ShaderLib.use("equirect2cube")
        ShaderLib.set_uniform("equirectangularMap", 0)
        self._render_cube_faces("equirect2cube", self.env_cubemap, ENV_SIZE)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)  # for prefilter sampling later

        # --- irradiance ---
        self.irradiance_map = self._new_cubemap(IRRADIANCE_SIZE, mip=False)
        self._resize_capture_depth(IRRADIANCE_SIZE)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        ShaderLib.use("irradiance")
        ShaderLib.set_uniform("environmentMap", 0)
        self._render_cube_faces("irradiance", self.irradiance_map, IRRADIANCE_SIZE)

        # --- prefilter chain: one bake per mip, roughness = mip / (mips-1) ---
        self.prefilter_map = self._new_cubemap(PREFILTER_SIZE, mip=True)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        ShaderLib.use("prefilter")
        ShaderLib.set_uniform("environmentMap", 0)
        for mip in range(PREFILTER_MIPS):
            size = PREFILTER_SIZE >> mip
            self._resize_capture_depth(size)
            ShaderLib.set_uniform("roughness", mip / (PREFILTER_MIPS - 1))
            self._render_cube_faces("prefilter", self.prefilter_map, size, mip=mip)

        # --- BRDF LUT (fullscreen triangle into RG16F 2D) ---
        self.brdf_lut = self._bake_brdf_lut()

        # --- PBR teapot-grid shader: constant uniforms (lights, material,
        # sampler units) set once here; per-teapot metallic/roughness/MVP
        # are set per-draw in paintGL ---
        ShaderLib.use("pbr")
        ShaderLib.set_uniform("albedo", 0.5, 0.0, 0.0)
        ShaderLib.set_uniform("ao", 1.0)
        ShaderLib.set_uniform("camPos", self.eye)
        ShaderLib.set_uniform("irradianceMap", 0)
        ShaderLib.set_uniform("prefilterMap", 1)
        ShaderLib.set_uniform("brdfLUT", 2)
        light_colors = Vec3Array([Vec3(300.0, 300.0, 300.0) for _ in range(4)])
        self.light_positions = Vec3Array(
            [
                Vec3(-10.0, 4.0, -10.0),
                Vec3(10.0, 4.0, -10.0),
                Vec3(-10.0, 4.0, 10.0),
                Vec3(10.0, 4.0, 10.0),
            ]
        )
        for i in range(4):
            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_positions[i])
            ShaderLib.set_uniform(f"lightColors[{i}]", light_colors[i])

    def _upload_equirect(self) -> int:
        """Upload the HDRI as a float32 2D texture (source for the bake)."""
        img = load_equirect_hdr(HDRI_PATH)  # (H, W, 3) float32
        h, w = img.shape[:2]
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB16F,
            w,
            h,
            0,
            gl.GL_RGB,
            gl.GL_FLOAT,
            img,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        return tex

    def _new_cubemap(self, size: int, mip: bool) -> int:
        """Allocate an empty RGB16F cube texture (optionally with a mip chain)."""
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, tex)
        for face in range(6):
            gl.glTexImage2D(
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                0,
                gl.GL_RGB16F,
                size,
                size,
                0,
                gl.GL_RGB,
                gl.GL_FLOAT,
                None,
            )
        wrap = gl.GL_CLAMP_TO_EDGE
        for axis in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T, gl.GL_TEXTURE_WRAP_R):
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, axis, wrap)
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR if mip else gl.GL_LINEAR
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, min_filter)
        gl.glTexParameteri(
            gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
        )
        if mip:
            gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)
        return tex

    def _resize_capture_depth(self, size: int) -> None:
        """Resize the shared capture FBO's depth renderbuffer to `size`x`size`.

        Every bake stage (env cubemap, irradiance, each prefilter mip) renders
        at a different resolution, so the depth buffer backing the capture FBO
        is resized to match immediately before that stage's draws.
        """
        if size == self._capture_rbo_size:
            return
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._capture_rbo)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, size, size
        )
        self._capture_rbo_size = size

    def _bake_brdf_lut(self) -> int:
        """Bake the split-sum BRDF LUT into a 512^2 RG16F 2D texture, drawn
        from a `gl_VertexID`-generated fullscreen triangle (no VBO needed)."""
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RG16F,
            LUT_SIZE,
            LUT_SIZE,
            0,
            gl.GL_RG,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._capture_fbo)
        self._resize_capture_depth(LUT_SIZE)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0
        )
        assert (
            gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        )
        gl.glViewport(0, 0, LUT_SIZE, LUT_SIZE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use("brdf")
        gl.glBindVertexArray(self._empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return tex

    def _render_cube_faces(
        self, shader: str, target: int, size: int, mip: int = 0
    ) -> None:
        """Render the bound shader into all six faces of `target` at mip level `mip`.

        Assumes the shader takes `projection`/`view` uniforms and samples via the
        cube's interpolated local position (see CubeVertex.glsl). The caller binds
        any source textures/uniforms (e.g. roughness) before calling.
        """
        ShaderLib.use(shader)
        ShaderLib.set_uniform("projection", _CAPTURE_PROJECTION)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._capture_fbo)
        gl.glViewport(0, 0, size, size)
        for face in range(6):
            ShaderLib.set_uniform("view", _CAPTURE_VIEWS[face])
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                target,
                mip,
            )
            if face == 0:
                # Cheap sanity check: catch a mis-sized depth buffer or a bad
                # attachment loudly instead of silently baking garbage.
                assert (
                    gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                    == gl.GL_FRAMEBUFFER_COMPLETE
                )
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            Primitives.draw("cube")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _draw_skybox(self) -> None:
        """The background sky: the baked environment cubemap, always drawn last
        behind everything else -- translation is stripped from the view so it
        stays centred on the camera no matter where we pan.

        Doubles as the `E` debug view for the cube bakes: "irradiance" swaps
        in the 32^2 convolved cubemap, "prefilter mip 2" keeps the env cubemap
        but samples it through the prefiltered chain at a fixed LOD.
        """
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)
        ShaderLib.use("skybox")
        gl.glActiveTexture(gl.GL_TEXTURE0)
        debug = DEBUG_VIEWS[self.debug_view]
        if debug == "irradiance":
            gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.irradiance_map)
            lod = 0.0
        elif debug == "prefilter mip 2":
            gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.prefilter_map)
            lod = 2.0
        else:
            gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
            lod = 0.0
        ShaderLib.set_uniform("skybox", 0)
        ShaderLib.set_uniform("lod", lod)
        view_rotation_only = self.view.copy()
        view_rotation_only[3, 0] = 0.0
        view_rotation_only[3, 1] = 0.0
        view_rotation_only[3, 2] = 0.0
        ShaderLib.set_uniform("MVP", self.project @ view_rotation_only)
        Primitives.draw("cube")
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

    def load_matrices_to_shader(self) -> None:
        """Copy the transform's M/MVP/normalMatrix to the currently bound
        shader (copied from PBR/SimplePBR:133-142)."""
        M = self.mouse_global_tx @ self.transform.matrix()
        MV = self.view @ M
        MVP = self.project @ MV

        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("M", M)

    def _draw_teapot_grid(self) -> None:
        """The 7x7 PBR teapot grid: rows sweep metallic 0..1, columns sweep
        roughness 0.05..1, all lit by the split-sum IBL baked in
        initializeGL (or the flat SimplePBR ambient hack when `I` toggles
        IBL off)."""
        ShaderLib.use("pbr")
        ShaderLib.set_uniform("useIBL", self.use_ibl)
        ShaderLib.set_uniform("camPos", self.eye)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.irradiance_map)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.prefilter_map)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.brdf_lut)
        for row in range(7):
            for col in range(7):
                self.transform.reset()
                self.transform.set_position((col - 3) * 3.0, (row - 3) * 3.0, 0.0)
                ShaderLib.set_uniform("metallic", row / 6.0)
                ShaderLib.set_uniform("roughness", max(0.05, col / 6.0))
                self.load_matrices_to_shader()
                Primitives.draw("teapot")

    def _draw_hud(self) -> None:
        ibl_state = "on" if self.use_ibl else "off"
        Text.render_text(
            "Arial",
            10,
            20,
            f"IBL: {ibl_state}  |  view: {DEBUG_VIEWS[self.debug_view]}",
            Vec3(1.0, 1.0, 1.0),
        )

    def _draw_debug_overlay(self) -> None:
        """`E` debug view "brdf lut": the BRDF LUT drawn as a small quad in
        the bottom-right corner, red = A (scale), green = B (bias)."""
        if DEBUG_VIEWS[self.debug_view] != "brdf lut":
            return
        gl.glDisable(gl.GL_DEPTH_TEST)
        ShaderLib.use("lutquad")
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.brdf_lut)
        ShaderLib.set_uniform("lut", 0)
        gl.glBindVertexArray(self._empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """`E` cycles the debug view, `I` toggles IBL; everything else falls
        through to the mixin's defaults (Escape/W/S/Space)."""
        if event.key() == Qt.Key_E:
            self.debug_view = (self.debug_view + 1) % len(DEBUG_VIEWS)
            logger.info(f"debug view: {DEBUG_VIEWS[self.debug_view]}")
            self.update()
            return
        if event.key() == Qt.Key_I:
            self.use_ibl = not self.use_ibl
            logger.info(f"IBL: {'on' if self.use_ibl else 'off'}")
            self.update()
            return
        super().keyPressEvent(event)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        self._draw_teapot_grid()
        self._draw_skybox()
        self._draw_debug_overlay()
        self._draw_hud()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        # Update the projection matrix to match the new aspect ratio.
        # This creates a perspective projection with a 45-degree field of view.
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        Text.set_screen_size(w, h)


class DebugApplication(QApplication):
    """
    A custom QApplication subclass for improved debugging.

    By default, Qt's event loop can suppress exceptions that occur within event handlers
    (like paintGL or mouseMoveEvent), making it very difficult to debug as the application
    may simply crash or freeze without any error message. This class overrides the `notify`
    method to catch these exceptions, print a full traceback to the console, and then
    re-raise the exception to halt the program, making the error immediately visible.
    """

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            # Attempt to process the event as usual
            return super().notify(receiver, event)
        except Exception:
            # If an exception occurs, print the full traceback
            traceback.print_exc()
            # Re-raise the exception to stop the application
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

    # --- Application Entry Point ---
    print("starting")
    # Create a QSurfaceFormat object to request a specific OpenGL context
    format: QSurfaceFormat = QSurfaceFormat()
    # Request 4x multisampling for anti-aliasing
    format.setSamples(4)
    # Request OpenGL version 4.1 as this is the highest supported on macOS
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    # Request a Core Profile context, which removes deprecated, fixed-function pipeline features
    format.setProfile(QSurfaceFormat.CoreProfile)
    # Request a 24-bit depth buffer for proper 3D sorting
    format.setDepthBufferSize(24)
    # Set default format for all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    # Apply this format to all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    # Start the application's event loop
    sys.exit(app.exec())
