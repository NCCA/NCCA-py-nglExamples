#!/usr/bin/env -S uv run --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
from FloorPipeline import FloorPipeline
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Prims,
    Vec3,
    look_at,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from TeapotPipeline import TeapotPipeline

# from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of NumpyBufferWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple WebGPU ?")
        self.device = None
        self.mouse_global_tx: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()  # Position of the model in world space
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

        self.pipelines = []
        self.vertex_buffer = None
        self.msaa_sample_count = 4

        self.texture_size = (1024, 1024)
        self.rotation = 0.0
        self.eye = Vec3(0.0, 2.0, 4.0)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.light_pos = Vec3(0.0, 2.0, 2.0)

        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 100.0, PerspMode.WebGPU
        )
        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._create_render_buffer()
            self._init_buffers()
            self._create_render_pipeline()
            self.start_update_timer(16)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            exit(1)

    def _init_buffers(self):
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        width = self.ratio * self.width()
        height = self.ratio * self.height()
        self.pipelines.append(
            TeapotPipeline(
                self.device,
                self.eye,
                self.light_pos,
                self.view,
                self.project,
                width,
                height,
            )
        )
        self.pipelines.append(
            FloorPipeline(
                self.device,
                self.eye,
                self.light_pos,
                self.view,
                self.project,
                width,
                height,
            )
        )

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.update_uniform_buffers()
        for pipeline in self.pipelines:
            pipeline.paint(
                self.colour_buffer_texture_view,
                self.multisample_texture_view,
                self.depth_buffer_view,
            )
        self._update_colour_buffer()

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers.
        """
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z
        for pipeline in self.pipelines:
            pipeline.update_uniform_buffers(self.mouse_global_tx)

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        self.frame_buffer = np.zeros([self.height(), self.width(), 4], dtype=np.uint8)

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            event: The resize event object.
        """
        # Update the stored width and height, considering high-DPI displays
        # Update projection matrix
        self.project = perspective(
            45.0, width / height if height > 0 else 1, 0.1, 350.0, PerspMode.WebGPU
        )

        for pipeline in self.pipelines:
            pipeline.width = width
            pipeline.height = height

        self.update()

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()

        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)

        self.update()
        super().keyPressEvent(event)

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
        if num_pixels.y() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.y() < 0:
            self.model_position.z -= self.ZOOM
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
