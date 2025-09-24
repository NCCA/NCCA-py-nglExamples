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
from ngl import (
    Mat3,
    Mat4,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Transform,
    Vec3,
    Vec4,
    logger,
    look_at,
    perspective,
)
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

PHONG_SHADER = "phong"
TEXTURE_SHADER = "texture"
TEXTURE_WIDTH = 1024
TEXTURE_HEIGHT = 1024


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
        self.setTitle("FBO Render to Texture")
        self.transform: Transform = Transform()
        self.rotation: float = 0.0

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
        self.view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.projection = perspective(
            45.0, self.window_width / self.window_height, 0.1, 100.0
        )
        self._load_shaders()
        self._create_texture_object()
        self._create_fbo()
        Primitives.load_default_primitives()
        Primitives.create_triangle_plane("plane", 2, 2, 20, 20, Vec3(0, 1, 0))
        Primitives.create_sphere("sphere", 0.4, 80)
        self.startTimer(60)

    def _load_shaders(self):
        if not ShaderLib.load_shader(
            PHONG_SHADER, "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use(PHONG_SHADER)
        light_pos = Vec4(-2.0, 5.0, 2.0, 0.0)
        ShaderLib.set_uniform("light.position", light_pos)
        ShaderLib.set_uniform("light.ambient", 0.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("light.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light.specular", 0.8, 0.8, 0.8, 1.0)
        # gold like phong material
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 0.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 0.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 0.0)
        ShaderLib.set_uniform("material.shininess", 51.2)
        ShaderLib.set_uniform("viewerPos", Vec3(2.0, 2.0, 2.0))
        if not ShaderLib.load_shader(
            TEXTURE_SHADER, "shaders/TextureVertex.glsl", "shaders/TextureFragment.glsl"
        ):
            print("error loading shaders")
            self.close()

    def _create_texture_object(self):
        # create a texture object
        self.texture_id = gl.glGenTextures(1)
        # bind it to make it active
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        # set params
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            TEXTURE_WIDTH,
            TEXTURE_HEIGHT,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        # now turn the texture off for now
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _create_fbo(self):
        # create a framebuffer
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)

        # create a renderbuffer object to store depth info
        rbo_id = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo_id)

        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, TEXTURE_WIDTH, TEXTURE_HEIGHT
        )
        # bind
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        # attatch the texture we created earlier to the FBO
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.texture_id,
            0,
        )

        # now attach a renderbuffer to depth attachment point
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, rbo_id
        )
        # now got back to the default render context
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        # were finished as we have an attached RB so delete it
        gl.glDeleteRenderbuffers(1, rbo_id)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Apply rotation based on user input
        rot_x: Mat4 = Mat4().rotate_x(self.spin_x_face)
        rot_y: Mat4 = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Apply translation
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        self._pass_one()
        self._pass_two()

    def _pass_one(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        # set the background colour (using blue to show it up)
        gl.glClearColor(0, 0.4, 0.5, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # set our viewport to the size of the texture
        # if we want a different camera we would set this here
        gl.glViewport(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT)
        # rotate the teapot
        self.transform.reset()

        self.transform.set_rotation(self.rotation, self.rotation, self.rotation)
        self.load_matrices_to_shader()
        Primitives.draw("teapot")

    def _pass_two(self):
        #  now we are going to draw to the normal GL buffer and use the texture created
        #  in the previous render to draw to our objects
        #  first bind the normal render buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        # now enable the texture we just rendered to
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Grey Background
        # clear this screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # get the new shader and set the new viewport size
        ShaderLib.use(TEXTURE_SHADER)
        # this takes into account retina displays etc
        # gl.glViewport(0, 0, int(self.window_width * self.devicePixelRatio()), int(self.window_height * self.devicePixelRatio()))
        gl.glViewport(0, 0, self.window_width, self.window_height)

        self.transform.reset()
        MVP = self.projection @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", MVP)
        Primitives.draw("plane")
        self.transform.set_position(0, 1, 0)
        MVP = (
            self.projection
            @ self.view
            @ self.mouse_global_tx
            @ self.transform.get_matrix()
        )
        ShaderLib.set_uniform("MVP", MVP)
        Primitives.draw("sphere")

    def load_matrices_to_shader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        ShaderLib.use(PHONG_SHADER)
        M = self.transform.get_matrix()
        MV = self.view @ M
        mvp = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("M", M)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

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

    def timerEvent(self, event):
        self.rotation += 1
        self.update()


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
