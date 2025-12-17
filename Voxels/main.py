#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import sys
import traceback

import OpenGL.GL as gl
from FrameBufferObject import FrameBufferObject
from ncca.ngl import (
    FirstPersonCamera,
    Mat4,
    PySideEventHandlingMixin,
    ShaderLib,
    Texture,
    Vec3,
    logger,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from Terrain import Terrain
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

VOXEL_SHADER = "VoxelShader"


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
            translation_sensitivity=0.01,
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
        self.setTitle("Texture Buffer Voxel Rendering on the GPU")
        self._setup_camera()
        self.debug = True

    def _setup_camera(self):
        eye = Vec3(0, 10, 260)
        look = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.camera = FirstPersonCamera(eye, look, up, 45.0)
        self.camera.set_projection(
            45.0, self.window_width / self.window_height, 0.05, 350.0
        )

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
        if not ShaderLib.load_shader(
            VOXEL_SHADER,
            "shaders/VoxelVertex.glsl",
            "shaders/VoxelFragment.glsl",
            geo="shaders/VoxelGeometry.glsl",
        ):
            logger.error("Failed to load shader")

        ShaderLib.use(VOXEL_SHADER)
        ShaderLib.set_uniform("textureAtlasDims", 16, 16)
        texture = Texture("textures/minecrafttextures.jpg")
        self.texture_id = texture.set_texture_gl()
        ShaderLib.set_uniform("textureSampler", 0)
        ShaderLib.set_uniform("posSampler", 1)
        ShaderLib.set_uniform("texIndexSampler", 2)
        ShaderLib.set_uniform("isActiveSampler", 3)
        # Generate a VAO so we can trigger drawing There is no
        # geometry on this
        self.vao = gl.glGenVertexArrays(1)
        self._create_framebuffer()
        self.terrain = Terrain(250, 30, 100, 16 * 16)
        self.terrain.gen_texture_buffer()

    def _create_framebuffer(self):
        FrameBufferObject.set_default_fbo(self.defaultFramebufferObject())
        self.render_fbo = FrameBufferObject.create(
            int(self.window_width * self.devicePixelRatio()),
            int(self.window_height * self.devicePixelRatio()),
        )
        with self.render_fbo:
            # self.render_fbo.bind()
            self.render_fbo.add_colour_attachment(
                "colour",
                GLAttachment._0,
                GLTextureFormat.RGBA,
                GLTextureInternalFormat.RGBA16F,
                GLTextureDataType.FLOAT,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )
            self.render_fbo.add_colour_attachment(
                "id",
                GLAttachment._1,
                GLTextureFormat.RGBA,
                GLTextureInternalFormat.RGBA16F,
                GLTextureDataType.FLOAT,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )
            self.render_fbo.add_depth_buffer(
                GLTextureDepthFormats.DEPTH_COMPONENT24,
                GLTextureMinFilter.NEAREST,
                GLTextureMagFilter.NEAREST,
                GLTextureWrap.CLAMP_TO_EDGE,
                GLTextureWrap.CLAMP_TO_EDGE,
                True,
            )
            attachments = [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1]

            gl.glDrawBuffers(len(attachments), attachments)
            if not self.render_fbo.is_complete():
                logger.error("FrameBuffer incomplete")
            else:
                self.render_fbo.print()

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Render the scene to our custom framebuffer
        with self.render_fbo as fbo:
            fbo.set_viewport()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            ShaderLib.use(VOXEL_SHADER)
            ShaderLib.set_uniform("MVP", self.camera.get_vp())

            # Bind the texture atlas to texture unit 0
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

            # Bind the terrain data (TBOs) to texture units 1, 2, and 3
            self.terrain.activate_texture_buffer(
                gl.GL_TEXTURE1, gl.GL_TEXTURE2, gl.GL_TEXTURE3
            )

            # Draw the points that will be turned into voxels by the geometry shader
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.terrain.voxel_positions.size)
            gl.glBindVertexArray(0)

        # Now, copy the result from our FBO to the screen

        # Bind the FBO for reading
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.render_fbo.get_id())
        # Bind the default window framebuffer for drawing
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())

        # Select which color attachment to read from based on the debug flag
        if self.debug:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)  # Use the ID buffer for debugging
        else:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)  # Use the main color buffer

        # Get the FBO dimensions for the blit operation
        w = self.render_fbo.width
        h = self.render_fbo.height

        # Perform the blit. This copies the FBO content to the screen.
        gl.glBlitFramebuffer(
            0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST
        )

        # It's good practice to re-bind the default FBO for both read and draw
        # and reset the read buffer to the default when done.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glReadBuffer(gl.GL_BACK)

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
