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
from ngl import ShaderLib, VAOFactory, VAOType, VertexData, logger, perspective
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

SCREEN_QUAD = "ScreenQuad"


class MainWindow(QOpenGLWindow):
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

        # --- Window and UI Attributes ---
        self.window_width: int = int(1024 * self.devicePixelRatio())  # Window width¦
        self.window_height: int = int(720 * self.devicePixelRatio())  # Window height
        self.setTitle("Blit Textures")
        self.buffer_index = 0

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        # Set the background color to a dark grey
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        if not ShaderLib.load_shader(
            SCREEN_QUAD,
            "shaders/ScreenQuadVertex.glsl",
            "shaders/ScreenQuadFragment.glsl",
        ):
            print("error loading shaders")
            self.close()

        # get some setup stuff
        max_attach = gl.glGetIntegerv(gl.GL_MAX_COLOR_ATTACHMENTS)
        max_draw_buff = gl.glGetIntegerv(
            gl.GL_MAX_DRAW_BUFFERS,
        )
        logger.info(f"{max_attach=} {max_draw_buff=}")
        self._create_texture_objects(max_attach)
        self._create_frambuffer_object()
        self._create_screen_quad()

    def _create_texture_objects(self, number: int) -> None:
        self.textures = gl.glGenTextures(number)
        gl.glGenTextures(number, self.textures)
        # // create a texture object
        for t in self.textures:
            gl.glBindTexture(gl.GL_TEXTURE_2D, t)
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST
            )
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST
            )
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
            )
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                self.window_width,
                self.window_height,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                None,
            )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _create_frambuffer_object(self):
        # create a framebuffer object
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        attachment = 0
        for t in self.textures:
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0 + attachment,
                gl.GL_TEXTURE_2D,
                t,
                0,
            )
            attachment += 1

        # create a renderbuffer object for depth and stencil attachment
        # It is possible to do both at the same time in C / C++ python is not
        # liking it tho.
        self.depth_stencil_rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth_stencil_rbo)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER,
            gl.GL_DEPTH24_STENCIL8,
            self.window_width,
            self.window_height,
        )
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_STENCIL_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.depth_stencil_rbo,
        )

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            logger.error("Error Frambuffer not complete")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _create_screen_quad(self):
        self.screen_quad = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)

        quad = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
        with self.screen_quad:
            self.screen_quad.set_data(VertexData(quad, len(quad) // 2))
            self.screen_quad.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, 0, 0)
            self.screen_quad.set_num_indices(6)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # draw to our FBO first
        ShaderLib.use(SCREEN_QUAD)
        ShaderLib.set_uniform("checkSize", 40.0)
        ShaderLib.set_uniform("width", self.window_width)
        ShaderLib.set_uniform("height", self.window_height)
        gl.glViewport(0, 0, self.window_width, self.window_height)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Pass 1 render to framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        draw_buffers = [
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_COLOR_ATTACHMENT1,
            gl.GL_COLOR_ATTACHMENT2,
            gl.GL_COLOR_ATTACHMENT3,
            gl.GL_COLOR_ATTACHMENT4,
            gl.GL_COLOR_ATTACHMENT5,
            gl.GL_COLOR_ATTACHMENT6,
            gl.GL_COLOR_ATTACHMENT7,
        ]
        gl.glDrawBuffers(8, draw_buffers)
        with self.screen_quad:
            self.screen_quad.draw()

        # pass 2 bind default fbo

        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_id)
        if self.buffer_index != 8:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + self.buffer_index)
            gl.glBlitFramebuffer(
                0,
                0,
                self.window_width,
                self.window_height,
                0,
                0,
                self.window_width,
                self.window_height,
                gl.GL_COLOR_BUFFER_BIT,
                gl.GL_NEAREST,
            )

        else:
            w4 = self.window_width // 4
            h2 = self.window_height // 2
            for i in range(0, 8):
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                if i < 4:
                    gl.glBlitFramebuffer(
                        w4 * i,
                        0,
                        w4 * i + w4,
                        h2,
                        w4 * i,
                        0,
                        w4 * i + w4,
                        h2,
                        gl.GL_COLOR_BUFFER_BIT,
                        gl.GL_NEAREST,
                    )

                else:
                    gl.glBlitFramebuffer(
                        w4 * (i - 4),
                        h2,
                        w4 * (i - 4) + w4,
                        self.window_height,
                        w4 * (i - 4),
                        h2,
                        w4 * (i - 4) + w4,
                        self.window_height,
                        gl.GL_COLOR_BUFFER_BIT,
                        gl.GL_NEAREST,
                    )

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

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if key == Qt.Key_Escape:
            self.close()
        mapping = {
            Qt.Key_1: 0,
            Qt.Key_2: 1,
            Qt.Key_3: 2,
            Qt.Key_4: 3,
            Qt.Key_5: 4,
            Qt.Key_6: 5,
            Qt.Key_7: 6,
            Qt.Key_8: 7,
            Qt.Key_A: 8,
        }

        if key in mapping:
            self.buffer_index = mapping[key]
        self.update()
        # Call the base class implementation for any unhandled events
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
