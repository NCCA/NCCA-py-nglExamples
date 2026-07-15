#!/usr/bin/env -S uv run --script
"""HDRI image-based lighting (OpenGL). See README.md."""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from exr_loader import load_equirect_hdr
from ncca.ngl import (
    Mat4,
    Prims,
    Transform,
    Vec3,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.opengl import (
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
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
        self._capture_fbo: int = 0

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

        eye = Vec3(0, 0, 4)
        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.view = look_at(eye, to, up)
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
        if not ok:
            logger.error("Error loading shaders")
            self.close()

        Primitives.create(Prims.CUBE, "cube", 2.0)
        Primitives.load_default_primitives()  # for "teapot", used in later tasks

        # Capture FBO used by every bake stage (equirect->cube here; irradiance,
        # prefilter and LUT in Tasks 3-4 reuse this same FBO).
        self._capture_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._capture_fbo)
        rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, ENV_SIZE, ENV_SIZE
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, rbo
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        self.equirect_tex = self._upload_equirect()
        self.env_cubemap = self._new_cubemap(ENV_SIZE, mip=True)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.equirect_tex)
        ShaderLib.use("equirect2cube")
        ShaderLib.set_uniform("equirectangularMap", 0)
        self._render_cube_faces("equirect2cube", self.env_cubemap, ENV_SIZE)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)  # for prefilter sampling later
        # (Tasks 3-4 append irradiance/prefilter/LUT bakes and the PBR draw here.)

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
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            Primitives.draw("cube")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _draw_skybox(self) -> None:
        """The background sky: the baked environment cubemap, always drawn last
        behind everything else -- translation is stripped from the view so it
        stays centred on the camera no matter where we pan."""
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)
        ShaderLib.use("skybox")
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_cubemap)
        ShaderLib.set_uniform("skybox", 0)
        ShaderLib.set_uniform("lod", 0.0)
        view_rotation_only = self.view.copy()
        view_rotation_only[3, 0] = 0.0
        view_rotation_only[3, 1] = 0.0
        view_rotation_only[3, 2] = 0.0
        ShaderLib.set_uniform("MVP", self.project @ view_rotation_only)
        Primitives.draw("cube")
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

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

        self._draw_skybox()

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
