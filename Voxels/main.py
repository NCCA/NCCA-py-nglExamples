#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import argparse
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from FrameBufferObject import FrameBufferObject
from ncca.ngl import FirstPersonCamera, Mat4, Vec3, logger
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib, Texture
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
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
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )  # --- Camera and Transformation Attributes ---
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width¦
        self.window_height: int = 720  # Window height
        self.setTitle("Texture Buffer Voxel Rendering on the GPU")
        self._setup_camera()
        self.debug = False

        # --- Keyboard movement state ---
        self.keys_pressed: set = set()  # currently held keys
        self.move_timer = QElapsedTimer()  # used to compute frame delta time
        self.last_frame: float = 0.0

        # --- Mouse picking / voxel editing state ---
        self.screen_click_x: float = 0.0  # last mouse position (logical pixels)
        self.screen_click_y: float = 0.0
        self.voxel_index: int = 0  # voxel currently under the cursor
        self.current_depth: float = 1.0  # linear depth under the cursor
        self.orig_x: float = 0.0  # left-drag anchor for camera rotation
        self.orig_y: float = 0.0
        # near/far used for both the projection and depth linearisation
        self.near: float = 0.05
        self.far: float = 350.0

    def _setup_camera(self):
        eye = Vec3(0, 10, 60)
        look = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.camera = FirstPersonCamera(eye, look, up, 45.0)
        # set_projection returns the matrix but does not store it, so assign it back
        # (mirrors what the FirstPersonCamera constructor does internally).
        self.camera._projection = self.camera.set_projection(
            45.0, self.window_width / self.window_height, 0.05, 350.0
        )
        # set_projection does not store these, but process_mouse_scroll rebuilds
        # the projection from them, so seed them to keep wheel-zoom correct.
        self.camera.aspect = self.window_width / self.window_height
        self.camera.near = 0.05
        self.camera.far = 350.0
        self.camera.zoom = 45.0

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
        self.move_timer.start()

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
        # Update the camera from any held movement keys before drawing.
        self._process_movement()
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
            gl.glDrawArrays(gl.GL_POINTS, 0, self.terrain.get_num_voxels())
            gl.glBindVertexArray(0)

        # Read the voxel index and depth under the cursor. Done every frame
        # (like NGLScene::paintGL) because the depth also gates forward
        # movement in _process_movement. Then apply any held edit keys:
        # S removes, Z/X change the texture index.
        self._update_voxel_index()
        if Qt.Key_S in self.keys_pressed:
            self.terrain.remove_index(self.voxel_index)
        if Qt.Key_Z in self.keys_pressed:
            self.terrain.change_texture_id(self.voxel_index, -1)
        if Qt.Key_X in self.keys_pressed:
            self.terrain.change_texture_id(self.voxel_index, 1)

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

        # Keep redrawing while movement or edit keys are held so motion and
        # continuous voxel editing work while a key is down.
        if self.keys_pressed & (self._MOVE_KEYS | self._EDIT_KEYS):
            self.update()

    # Arrow keys drive the first-person camera (see _process_movement).
    _MOVE_KEYS = {Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down}
    # S removes the voxel under the cursor, Z/X change its texture index.
    _EDIT_KEYS = {Qt.Key_S, Qt.Key_Z, Qt.Key_X}

    def _process_movement(self) -> None:
        """
        Move the camera based on the currently held arrow keys, scaled by the
        time since the last frame so motion is frame-rate independent.

        Mirrors the C++ NGLScene::paintGL movement handling:
        Up/Down move forward/back, Left/Right strafe.
        """
        current_frame = self.move_timer.elapsed() * 0.001  # ms -> seconds
        # Clamp the delta so the first press after an idle gap (when no frames
        # are being drawn) doesn't span the whole gap and jump the camera.
        delta_time = min(current_frame - self.last_frame, 0.05)
        self.last_frame = current_frame

        if not (self.keys_pressed & self._MOVE_KEYS):
            return

        x_direction = 0.0  # forward / back
        y_direction = 0.0  # strafe left / right
        inc = 1.0
        # Only move forward if nothing is close in front of the cursor,
        # so we stop before flying into a block (matches NGLScene::paintGL).
        if Qt.Key_Up in self.keys_pressed and self.current_depth >= 0.005:
            x_direction = inc
        if Qt.Key_Down in self.keys_pressed:
            x_direction = -inc
        if Qt.Key_Right in self.keys_pressed:
            y_direction = inc
        if Qt.Key_Left in self.keys_pressed:
            y_direction = -inc

        self.camera.move(x_direction, y_direction, delta_time)

    def keyPressEvent(self, event) -> None:
        """
        Track held keys for camera movement and toggle debug view with 'D',
        delegating all other keys to the event-handling mixin.
        """
        key = event.key()
        if key in self._MOVE_KEYS or key in self._EDIT_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.add(key)
                self.update()
        elif key == Qt.Key_D:
            self.debug = not self.debug
            self.update()
        else:
            # Let the mixin handle W/S/Space/Escape and anything else.
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """Stop tracking a key once it is really released (ignore auto-repeat)."""
        if not event.isAutoRepeat():
            self.keys_pressed.discard(event.key())
        super().keyReleaseEvent(event)

    def _update_voxel_index(self) -> None:
        """
        Read the ID (picking) buffer at the current mouse position and decode
        the voxel index stored there, matching NGLScene::updateVoxelIndex.

        The fragment shader writes each voxel's vertex ID into colour
        attachment 1 as an RGBA-encoded integer; here we read that single
        pixel back and reconstruct the index.
        """
        dpr = self.devicePixelRatio()
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.render_fbo.get_id())
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)

        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        x = int(self.screen_click_x * dpr)
        # OpenGL's origin is bottom-left, Qt's is top-left, so flip y.
        y = int(viewport[3] - self.screen_click_y * dpr)

        data = np.frombuffer(
            bytes(gl.glReadPixels(x, y, 1, 1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)),
            dtype=np.uint8,
        )
        r, g, b, a = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
        self.voxel_index = (r << 24) | (g << 16) | (b << 8) | a

        # Read the depth at the same pixel and linearise it. This gates forward
        # movement so the camera stops in front of a block.
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        depth = np.frombuffer(
            bytes(gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)),
            dtype=np.float32,
        )[0]
        self.current_depth = (2.0 * self.near) / (
            self.far + self.near - depth * (self.far - self.near)
        )

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)

    def mousePressEvent(self, event) -> None:
        """Store the anchor position for left-button camera rotation."""
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.orig_x = position.x()
            self.orig_y = position.y()

    def mouseMoveEvent(self, event) -> None:
        """
        Track the cursor position for voxel picking and, while the left button
        is held, rotate the first-person camera (mirrors NGLSceneMouseControls).
        """
        position = event.position()
        # Always record the cursor position so picking uses the latest location.
        self.screen_click_x = position.x()
        self.screen_click_y = position.y()

        if event.buttons() == Qt.LeftButton:
            diff_x = position.x() - self.orig_x
            diff_y = position.y() - self.orig_y
            self.orig_x = position.x()
            self.orig_y = position.y()
            self.camera.process_mouse_movement(diff_x, diff_y)
            self.update()

    def wheelEvent(self, event) -> None:
        """Zoom the camera with the mouse wheel (matches NGLScene::wheelEvent)."""
        if event.angleDelta().y() > 0:
            self.camera.process_mouse_scroll(1.0)
        elif event.angleDelta().y() < 0:
            self.camera.process_mouse_scroll(-1.0)
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        # Update the camera's projection matrix to match the new aspect ratio.
        # set_projection returns the matrix, so assign it back onto the camera.
        self.camera._projection = self.camera.set_projection(
            45.0, float(w) / h, 0.05, 350.0
        )
        # keep aspect in sync so wheel-zoom rebuilds the projection correctly
        self.camera.aspect = float(w) / h


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

    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    # Start the application's event loop
    sys.exit(app.exec())
