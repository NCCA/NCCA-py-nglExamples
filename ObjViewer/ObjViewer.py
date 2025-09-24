#!/usr/bin/env -S uv run --script
"""
BoidShaded Example

This script demonstrates how to use OpenGL and Qt to render a simple 3D object (a "boid") with Phong shading.
It sets up a window, handles user input for rotation/translation/zoom, and manages OpenGL resources.
"""

import sys
import traceback

import OpenGL.GL as gl
from ngl import Mat4, Obj, ShaderLib, Vec3, logger, look_at, perspective
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

TEXTURE_SHADER = "TextureShader"


class MainWindow(QOpenGLWindow):
    """

    Handles OpenGL initialization, rendering, and user input for interactive control.
    """

    def __init__(
        self, mesh_name: str = "", texture_name: str = "", parent: object = None
    ) -> None:
        """
        Initialize the window and set up default parameters.
        """
        super().__init__()
        self.mouse_global_tx: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("OBJ Viewer")
        self.view: Mat4 = Mat4()  # View matrix
        self.project: Mat4 = Mat4()  # Projection matrix
        self.model_position: Vec3 = Vec3()  # Position of the model in world space

        self.mesh_name = mesh_name
        self.texture_name = texture_name

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.original_x_rotation: int = (
            0  # Initial X position of the mouse when a rotation starts
        )
        self.original_y_rotation: int = (
            0  # Initial Y position of the mouse when a rotation starts
        )
        self.original_x_pos: int = (
            0  # Initial X position of the mouse when a translation starts
        )
        self.original_y_pos: int = (
            0  # Initial Y position of the mouse when a translation starts
        )
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming

    def initializeGL(self) -> None:
        """
        Set up OpenGL context, load shaders, and initialize scene.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)  # Set background color
        gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
        gl.glEnable(gl.GL_MULTISAMPLE)  # Enable anti-aliasing

        # Set up camera/view matrix
        eye = Vec3(0, 1, 10)
        to = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.view = look_at(eye, to, up)
        # Load and use Texture shader
        if not ShaderLib.load_shader(
            TEXTURE_SHADER, "shaders/TextureVertex.glsl", "shaders/TextureFragment.glsl"
        ):
            print("error loading shaders")
            self.close()
        ShaderLib.use(TEXTURE_SHADER)
        # Mesh needs to be created when we have OpenGL context as using VAO
        self.mesh = Obj.obj_with_vao(self.mesh_name, self.texture_name)

    def loadMatricesToShader(self) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

    def paintGL(self) -> None:
        """
        Render the scene. Called automatically by Qt.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(TEXTURE_SHADER)

        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        self.loadMatricesToShader()
        # Draw geometry
        self.mesh.draw()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle window resizing and update the projection matrix.

        Args:
            w: New window width.
            h: New window height.
        """

        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.1, 350.0)

    def keyPressEvent(self, event) -> None:
        """
        Handle keyboard input for controlling the scene.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)  # Wireframe mode
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)  # Solid mode
        elif key == Qt.Key_Space:
            # Reset rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        else:
            # Call base implementation for unhandled keys
            super().keyPressEvent(event)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement events for camera control.

        Args:
            event: The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        # Translate (pan) the scene if the right mouse button is pressed
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mousePressEvent(self, event) -> None:
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        position = event.position()
        # Left button initiates rotation
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        # Right button initiates translation
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handles mouse button release events to stop rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        # Stop rotating when the left button is released
        if event.button() == Qt.LeftButton:
            self.rotate = False
        # Stop translating when the right button is released
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming.

        Args:
            event: The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
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
    # Set up Qt application and OpenGL format
    # Check for a "--debug" command-line argument to run the DebugApplication
    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    format = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    window = MainWindow("models/Helix.obj", "textures/helix_base.tif")
    window.setFormat(format)
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
"""
std::string oname("models/Helix.obj");
  std::string tname("textures/helix_base.tif");
  if(argc ==2)
  {
    oname=argv[1];
    tname="textures/ratGrid.png";
  }
  else if(argc == 3)
  {
    oname=argv[1];
    tname=argv[2];
  }
  """
