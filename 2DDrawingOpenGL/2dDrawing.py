#!/usr/bin/env -S uv run --script
"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

"""

import argparse
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat4, Vec3, logger, ortho
from ncca.ngl.opengl import ShaderLib, Text, VAOFactory, VAOType, VertexData
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

PBR_SHADER = "pbr"

SIM_WIDTH = 200
SIM_HEIGHT = 200


class MainWindow(QOpenGLWindow):
    """
    The main window for the OpenGL application.

    Inherits from QOpenGLWindow to provide a canvas for OpenGL rendering within a PySide6 GUI.
    It handles user input (mouse, keyboard) for camera control and manages the OpenGL context.
    """

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.project: Mat4 = ortho(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, -SIM_HEIGHT / 2, SIM_HEIGHT / 2, 0, 100
        )
        self.ratio = self.devicePixelRatio()
        # A constant vector to simulate wind, pushing all particles.
        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height
        self.setTitle("2d Points")
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.zoom = 1.0

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
        self.gen_points(1000)
        self.vao = VAOFactory.create_vao(VAOType.MULTI_BUFFER, gl.GL_POINTS)
        with self.vao as vao:
            # setup the points data
            data = VertexData(data=self.positions.flatten(), size=self.positions.nbytes)
            vao.set_data(data, index=0)  # index 0 is positions
            vao.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, 0, 0)  # vec2
            # setup the colour data
            data = VertexData(data=self.colours.flatten(), size=self.colours.nbytes)
            vao.set_data(data, index=1)  # index 1 is colours
            vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 0, 0)  # r,g,b
            vao.set_num_indices(len(self.positions))

        ShaderLib.load_shader(
            "ColourShader", "ColourVertex.glsl", "ColourFragment.glsl"
        )
        ShaderLib.use("ColourShader")
        ShaderLib.set_uniform("projection_matrix", self.project)
        Text.add_font("Arial", "../font/Arial.ttf", 30)
        self.startTimer(16)

    def gen_points(self, num_points: int) -> None:
        """
        Generates random 2D points with associated positions, directions, and colours.

        This function initializes three numpy arrays:
        - self.positions: A 2D array storing the (x, y) coordinates of each point.
        - self.directions: A 2D array storing the (dx, dy) velocity vector for each point.
        - self.colours: A 2D array storing the (r, g, b) colour of each point.

        Parameters
        ----------
            num_points : int
                The number of points to generate.

        """
        # generate positions in 2D space the size of the simulation with 0,0 the center
        self.positions = np.zeros((num_points, 2), dtype=np.float32)
        self.positions[:, 0] = np.random.uniform(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, num_points
        )
        self.positions[:, 1] = np.random.uniform(
            -SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_points
        )

        # generate directions in 2D space with random velocities
        self.directions = np.zeros((num_points, 2), dtype=np.float32)
        self.directions[:, 0] = np.random.uniform(-1, 1, num_points)
        self.directions[:, 1] = np.random.uniform(-1, 1, num_points)

        # Normalize directions to unit length, so they only represent direction
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)

        # Give each particle a random speed
        min_speed = 0.5
        max_speed = 2.0
        # Create a (num_points, 1) array of random speeds
        speeds = np.random.uniform(min_speed, max_speed, (num_points, 1))
        # Multiply the unit direction vectors by the speeds to get final velocity vectors
        self.directions *= speeds

        self.colours = np.random.random((num_points, 3)).astype(np.float32)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # The shader is rendering the points as circles so set the point size
        gl.glPointSize(10)
        ShaderLib.use("ColourShader")
        proj = ortho(
            -SIM_WIDTH / 2 * self.zoom,
            SIM_WIDTH / 2 * self.zoom,
            -SIM_HEIGHT / 2 * self.zoom,
            SIM_HEIGHT / 2 * self.zoom,
            0,
            100,
        )
        ShaderLib.set_uniform("projection_matrix", proj)
        with self.vao as vao:
            data = VertexData(data=self.positions.flatten(), size=self.positions.nbytes)
            vao.set_data(data, index=0)
            vao.set_vertex_attribute_pointer(0, 2, gl.GL_FLOAT, 0, 0)  # vec2

            vao.set_num_indices(len(self.positions))
            vao.draw()

        Text.render_text(
            "Arial",
            10,
            30,
            f"Wind Value use Arrow Keys to Change, Space Reset [{self.wind[0]:.02f}, {self.wind[1]:.02f}]",
            Vec3(1.0, 1.0, 0.0),
        )

    def timerEvent(self, event: QTimerEvent) -> None:
        """
        This event is called at a regular interval (set by startTimer).
        It's used to update the animation of the scene.

        Here, it updates the positions of the points and makes them bounce
        off the edges of the simulation area.

        Parameters
        ----------
            event : QTimerEvent
                The QTimerEvent object, not used in this method but required by the API.


        """
        # Add the wind factor to the particle's own direction to get the final velocity
        velocities = self.directions + self.wind
        # Update positions using the final velocity
        self.positions += velocities

        # Define the boundaries of the simulation area
        x_min = -SIM_WIDTH / 2
        x_max = SIM_WIDTH / 2
        y_min = -SIM_HEIGHT / 2
        y_max = SIM_HEIGHT / 2

        # Create boolean masks to find points that are out of bounds
        hit_left = self.positions[:, 0] < x_min
        hit_right = self.positions[:, 0] > x_max
        hit_bottom = self.positions[:, 1] < y_min
        hit_top = self.positions[:, 1] > y_max

        # Reflect the particle's intrinsic direction for points that have hit a wall.
        self.directions[hit_left | hit_right, 0] *= -1
        self.directions[hit_bottom | hit_top, 1] *= -1

        # Clamp positions to the boundaries to prevent particles from getting stuck.
        # If a particle hit the left wall, its x position is set to the left boundary.
        self.positions[hit_left, 0] = x_min
        # If a particle hit the right wall, its x position is set to the right boundary.
        self.positions[hit_right, 0] = x_max
        # If a particle hit the bottom wall, its y position is set to the bottom boundary.
        self.positions[hit_bottom, 1] = y_min
        # If a particle hit the top wall, its y position is set to the top boundary.
        self.positions[hit_top, 1] = y_max

        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Parameters
        ----------
            w : int
                The new width of the window.
            h : int
                The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.ratio)
        self.window_height = int(h * self.ratio)
        Text.set_screen_size(self.window_width, self.window_height)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles keyboard press events.

        Parameters
        ----------
            event : QKeyEvent
                The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        if key == Qt.Key_Space:
            self.wind[0] = 0
            self.wind[1] = 0
            self.zoom = 1.0
        if key == Qt.Key_Up:
            self.wind[1] += 0.1
        if key == Qt.Key_Down:
            self.wind[1] -= 0.1
        if key == Qt.Key_Left:
            self.wind[0] -= 0.1
        if key == Qt.Key_Right:
            self.wind[0] += 0.1
        # Trigger a redraw to apply changes
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse movement events for camera control.

        Parameters
        ----------
            event : QMouseEvent
                The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if event.buttons() == Qt.LeftButton or event.buttons() == Qt.RightButton:
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse button press events to initiate rotation or translation.

        Parameters
        ----------
            event : QMouseEvent
                The QMouseEvent object.
        """
        event.position()
        # Left button initiates rotation

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events for zooming.

        Parameters
        ----------
            event : QWheelEvent
                The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.zoom += 0.01
        elif num_pixels.x() < 0:
            self.zoom -= 0.01
        self.zoom = max(0.05, min(10.0, self.zoom))
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

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
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
