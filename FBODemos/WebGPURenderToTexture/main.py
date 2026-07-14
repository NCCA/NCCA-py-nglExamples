#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from TeapotPipeline import TeapotPipeline
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Render To Texture")
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
        self.first_pass_pipeline = None
        self.pipeline = None
        self.msaa_sample_count = 4
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
            self._init_buffers()
            self._create_render_pipeline()
            self._create_render_buffer()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
        self.startTimer(16)

    def _init_buffers(self): ...

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline. First load the shader
        """
        self.first_pass_pipeline = TeapotPipeline(
            self.device,
            self.eye,
            self.light_pos,
            self.view,
            self.project,
            512,
            512,
        )

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized
        It's crucial to update the viewport and projection matrix here.

        Args:
            width: The new width of the window.
            height: The new height of the window.

        """

        self.update()

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        try:
            self.update_uniform_buffers()

            self.first_pass_pipeline.paint(
                self.colour_buffer_texture_view,
                self.multisample_texture_view,
                self.depth_buffer_view,
            )
            self._update_colour_buffer()
        except Exception:
            pass

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

        self.first_pass_pipeline.update_uniform_buffers(
            Mat4.rotate_y(self.rotation) @ Mat4.rotate_x(self.rotation)
        )

    def timerEvent(self, event):
        self.rotation += 1.0
        self.update()

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)


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
    win.resize(800, 600)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
