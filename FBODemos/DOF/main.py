#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import math
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from FrameBufferObject import FrameBufferObject
from ncca.ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    Prims,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    Transform,
    VAOFactory,
    VAOType,
    Vec2,
    Vec3,
    Vec4,
    VertexData,
    logger,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from TextureTypes import (
    GLAttachment,
    GLTextureDataType,
    GLTextureDepthFormats,
    GLTextureFormat,
    GLTextureInternalFormat,
    GLTextureMagFilter,
    GLTextureMinFilter,
    GLTextureWrap,
)

DOF_SHADER = "DOFShader"
PHONG_SHADER = "PhongShader"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """
    The main window for the OpenGL application.

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
            translation_sensitivity=0.1,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )  # --- Camera and Transformation Attributes ---
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width¦
        self.window_height: int = 720  # Window height
        self.setTitle("DOF Shader")
        self.fstop = 2.8
        self.av = int(3)  # used in f-stop calc where fstop=sqrtf(2^av)
        self.focal_length = 1.0
        self.focal_distance = 2.0
        self.focus_distance = 5.0
        self.transform = Transform()
        self.rot = 0.1

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        # Set the background color to a dark grey
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        # Enable depth testing, which ensures that objects closer to the camera obscure those further away
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Enable multisampling for anti-aliasing, which smooths jagged edges
        gl.glEnable(gl.GL_MULTISAMPLE)
        # Set up the camera's view matrix.
        # It looks from (0, 1, 4) towards (0, 0, 0) with the 'up' direction along the Y-axis.
        self.eye = Vec3(0, 2, 10)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.light_pos = Vec4(-2.0, 5.0, 2.0, 0.0)

        self._load_phong_shader()
        self._setup_checker_shader()
        self._setup_dof_shader()
        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "floor", 25, 25, 1, 1, Vec3(0, 1, 0))
        Primitives.create(Prims.SPHERE, "sphere", 0.4, 80)
        self._create_screen_quad()
        self._create_fbos()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        print(self.render_fbo.get_texture_id("renderTarget"))
        gl.glBindTexture(
            gl.GL_TEXTURE_2D, self.render_fbo.get_texture_id("renderTarget")
        )
        print(f"{self.render_fbo.depth_texture_id=}")
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_fbo.depth_texture_id)
        print(f"{self.blur_fbo.get_texture_id("blurTarget")=}")
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.blur_fbo.get_texture_id("blurTarget"))
        Text.add_font("Arial", "../../font/Arial.ttf", 40)

        self.startTimer(10)

    def _create_fbos(self):
        self.render_fbo = FrameBufferObject.create(
            1024 * self.devicePixelRatio(), 720 * self.devicePixelRatio()
        )
        with self.render_fbo:
            self.render_fbo.add_colour_attachment(
                "renderTarget",
                GLAttachment._0,
                GLTextureFormat.RGBA,
                GLTextureInternalFormat.RGBA8,
                GLTextureDataType.UNSIGNED_BYTE,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )

            self.render_fbo.add_depth_buffer(
                GLTextureDepthFormats.DEPTH_COMPONENT16,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )

            logger.info(f"RenderTargetID {self.render_fbo.id}")
            self.render_fbo.print()
            logger.info(f"Is complete {self.render_fbo.is_complete()}")
            self.blur_fbo = FrameBufferObject.create(
                self.window_width * self.devicePixelRatio(),
                self.window_height * self.devicePixelRatio(),
            )
            self.blur_fbo.bind()
            self.blur_fbo.add_colour_attachment(
                "blurTarget",
                GLAttachment._0,
                GLTextureFormat.RGBA,
                GLTextureInternalFormat.RGBA8,
                GLTextureDataType.UNSIGNED_BYTE,
                GLTextureMinFilter.LINEAR,
                GLTextureMagFilter.LINEAR,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )

            logger.info(f"blurTargetID {self.blur_fbo.id}")
            self.blur_fbo.print()
            logger.info(f"Is complete {self.blur_fbo.is_complete()}")

    def _load_dof_uniforms(self):
        ShaderLib.use(DOF_SHADER)
        try:
            magnification = self.focal_length / abs(
                self.focal_distance - self.focal_length
            )
        except ZeroDivisionError:
            magnification = 0.0
        blur = self.focal_length * magnification / self.fstop
        ppm = (
            math.sqrt(
                self.window_width * self.window_width
                + self.window_height * self.window_height
            )
            / 35
        )
        ShaderLib.set_uniform("depthRange", Vec2(0.1, 50.0))
        ShaderLib.set_uniform("blurCoefficient", blur)
        ShaderLib.set_uniform("PPM", ppm)
        ShaderLib.set_uniform(
            "screenResolution", Vec2(self.window_width, self.window_height)
        )
        ShaderLib.set_uniform("focusDistance", self.focal_distance)

    def _setup_dof_shader(self):
        if not ShaderLib.load_shader(
            DOF_SHADER, "shaders/DOFVertex.glsl", "shaders/DOFFragment.glsl"
        ):
            logger.error("Failed to load DOF shader")
        self._load_dof_uniforms()

    def _setup_checker_shader(self):
        ShaderLib.use(DefaultShader.CHECKER)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("checkOn", 1)
        ShaderLib.set_uniform(
            "lightPos", self.light_pos.x, self.light_pos.y, self.light_pos.z
        )  # vec3 here
        ShaderLib.set_uniform("colour1", 0.9, 0.9, 0.9, 1.0)
        ShaderLib.set_uniform("colour2", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("checkSize", 60.0)

    def _load_phong_shader(self):
        if not ShaderLib.load_shader(
            PHONG_SHADER, "shaders/PhongVert.glsl", "shaders/PhongFrag.glsl"
        ):
            logger.error("Failed to load Phong shader")
        ShaderLib.use(PHONG_SHADER)
        ShaderLib.set_uniform("light.position", self.light_pos)
        ShaderLib.set_uniform("light.ambient", 0.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("light.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light.specular", 0.8, 0.8, 0.8, 1.0)
        # gold like Phong material
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 0.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 0.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 0.0)
        ShaderLib.set_uniform("material.shininess", 51.2)
        ShaderLib.set_uniform("viewerPos", self.eye)

    def _create_screen_quad(self):
        self.screen_quad = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)

        quad = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
        with self.screen_quad:
            self.screen_quad.set_data(VertexData(quad, len(quad) // 2))
            self.screen_quad.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, 0, 0)
            self.screen_quad.set_num_indices(6)

    def _load_matrices_to_shader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        ShaderLib.use(PHONG_SHADER)
        M = self.mouse_global_tx @ self.transform.matrix()
        MV = self.view @ M
        mvp = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("M", M)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT or gl.GL_DEPTH_BUFFER_BIT)
        # Pass one draw to FBO
        ShaderLib.use(PHONG_SHADER)
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(
            gl.GL_TEXTURE_2D, self.render_fbo.get_texture_id("renderTarget")
        )
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_fbo.depth_texture_id)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.blur_fbo.get_texture_id("blurTarget"))

        with self.render_fbo:
            gl.glViewport(0, 0, self.window_width, self.window_height)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            self.transform.reset()

            for x in np.arange(-10, 10, 1.8):
                for z in np.arange(-10, 10, 1.8):
                    for y in np.arange(-10, 10, 1.8):
                        self.transform.set_rotation(0, self.rot, 0)
                        self.transform.set_position(x, 0.0, z)
                        self._load_matrices_to_shader()
                        Primitives.draw("teapot")
            self.rot += 1.0
            ShaderLib.use(DefaultShader.CHECKER)
            self.transform.reset()
            self.transform.set_position(0.0, -0.45, 0.0)
            MVP = (
                self.project
                @ self.view
                @ self.mouse_global_tx
                @ self.transform.matrix()
            )
            normal_matrix = Mat3.from_mat4(self.view @ self.mouse_global_tx)
            normal_matrix = normal_matrix.inverse().transposed()
            ShaderLib.set_uniform("MVP", MVP)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            Primitives.draw("floor")

        ShaderLib.use(DOF_SHADER)
        self.blur_fbo.bind()  # only bind for the first blur pass ()
        with self.screen_quad:
            self._load_dof_uniforms()
            self.blur_fbo.set_viewport()
            # HORIZONTAL BLUR
            ShaderLib.set_uniform("colourSampler", 0)
            ShaderLib.set_uniform("depthSampler", 1)
            ShaderLib.set_uniform("uTexelOffset", 1.0, 0.0)
            self.screen_quad.draw()
            self.blur_fbo.unbind()
            # Vertical Blur to default FB
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            ShaderLib.set_uniform("uTexelOffset", 0.0, 1.0)
            ShaderLib.set_uniform("colourSampler", 2)
            gl.glViewport(0, 0, self.window_width, self.window_height)
            ShaderLib.set_uniform(
                "screenResolution", Vec2(self.window_width, self.window_height)
            )
            self.screen_quad.draw()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        self._render_info()

    def _render_info(self):
        Text.render_text(
            "Arial",
            10,
            40,
            "Depth of Field Demo Use Arrows and OP / KL to change params",
            Vec3(1.0, 1.0, 0.0),
        )
        Text.render_text(
            "Arial",
            10,
            40 * 2,
            f"Focal distance: {self.focal_distance:.2f}",
            Vec3(1.0, 1.0, 0.0),
        )
        Text.render_text(
            "Arial", 10, 40 * 3, f"F-Stop : {self.fstop:.2f}", Vec3(1.0, 1.0, 0.0)
        )
        Text.render_text(
            "Arial",
            10,
            40 * 4,
            f"Focal Length: {self.focal_length:.2f}",
            Vec3(1.0, 1.0, 0.0),
        )
        Text.render_text("Arial", 10, 40 * 5, f"AV: {self.av:.2f}", Vec3(1.0, 1.0, 0.0))
        Text.render_text(
            "Arial",
            10,
            40 * 6,
            f"Focus Distance: {self.focus_distance:.2f}",
            Vec3(1.0, 1.0, 0.0),
        )

        # --- All drawing code should go here ---

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

        Text.set_screen_size(self.window_width, self.window_height)

    def timerEvent(self, event):
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.av -= 1
            self.fstop = math.sqrt(math.pow(2, self.av))
        elif event.key() == Qt.Key_Right:
            self.av += 1
            self.fstop = math.sqrt(math.pow(2, self.av))
        elif event.key() == Qt.Key_Up:
            self.focal_distance += 0.1
        elif event.key() == Qt.Key_Down:
            self.focal_distance -= 0.1
        elif event.key() == Qt.Key_I:
            self.focal_length += 0.1
        elif event.key() == Qt.Key_O:
            self.focal_length -= 0.1
        elif event.key() == Qt.Key_K:
            self.focus_distance += 0.1
        elif event.key() == Qt.Key_L:
            self.focus_distance -= 0.1

        else:
            super().keyPressEvent(event)


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
    # --- Application Entry Point ---

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

    # Check for a "--debug" command-line argument to run the DebugApplication
    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()
    # Start the application's event loop
    sys.exit(app.exec())
