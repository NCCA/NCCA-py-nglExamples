#!/usr/bin/env -S uv run --active --script
"""WebGPU render to texture demo.

A two pass demo that mirrors the OpenGL SimpleFBO example. The first pass
renders a rotating teapot into an offscreen texture; the second pass draws a
ground plane and a sphere textured with that offscreen render.
"""

import argparse
import sys
import traceback

import wgpu
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from ScenePipeline import ScenePipeline
from TeapotPipeline import TeapotPipeline
from wgpu.utils import get_default_device

# Size of the square offscreen texture the teapot is rendered into.
TEXTURE_SIZE = 1024


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
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming
        self.first_pass_pipeline = None
        self.scene_pipeline = None
        self.msaa_sample_count = 4
        self.rotation = 0.0
        # Camera for the teapot rendered into the texture (square aspect).
        self.eye = Vec3(0.0, 2.0, 4.0)
        self.teapot_view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.teapot_project = perspective(45.0, 1.0, 0.1, 100.0, PerspMode.WebGPU)
        self.light_pos = Vec3(0.0, 2.0, 2.0)
        # Camera for the final scene (plane + sphere).
        self.view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
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
            self._create_render_pipeline()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            traceback.print_exc()
        self.startTimer(16)

    def _create_render_pipeline(self) -> None:
        """
        Create the two render pipelines: the teapot (first pass) and the
        textured scene (second pass) that samples the teapot texture.
        """
        self.first_pass_pipeline = TeapotPipeline(
            self.device,
            self.eye,
            self.light_pos,
            self.teapot_view,
            self.teapot_project,
            TEXTURE_SIZE,
            TEXTURE_SIZE,
        )
        self.scene_pipeline = ScenePipeline(
            self.device,
            self.first_pass_pipeline.texture_view,
        )

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized. Update the projection matrix
        for the final scene so the aspect ratio stays correct.
        """
        self.project = perspective(
            45.0, float(width) / max(height, 1), 0.1, 100.0, PerspMode.WebGPU
        )
        self.update()

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        Pass one renders the teapot into the offscreen texture, pass two draws
        the plane and sphere textured with that result into the widget buffer.
        """
        if self.first_pass_pipeline is None or self.scene_pipeline is None:
            return
        try:
            self.update_uniform_buffers()
            # Pass one: teapot -> offscreen texture.
            self.first_pass_pipeline.paint()
            # Pass two: textured plane + sphere -> widget colour buffer.
            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.multisample_texture_view,
                        "resolve_target": self.colour_buffer_texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.4, 0.4, 0.4, 1.0),
                    }
                ],
                depth_stencil_attachment={
                    "view": self.depth_buffer_view,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            )
            self.scene_pipeline.render(render_pass)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception:
            traceback.print_exc()

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for both passes.
        """
        # Mouse driven camera transform for the final scene.
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        # Pass one: spin the teapot in the texture.
        self.first_pass_pipeline.update_uniform_buffers(
            Mat4.rotate_y(self.rotation) @ Mat4.rotate_x(self.rotation)
        )

        # Pass two: plane at the origin, sphere lifted one unit above it.
        plane_mvp = self.project @ self.view @ self.mouse_global_tx
        sphere_tx = Mat4()
        sphere_tx[3, 1] = 1.0
        sphere_mvp = self.project @ self.view @ self.mouse_global_tx @ sphere_tx
        self.scene_pipeline.update_uniforms(plane_mvp, sphere_mvp)

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
            self.close()
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()

        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Begin a rotate (LMB) or pan (RMB) interaction."""
        position: QPoint = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Rotate or pan the scene camera while a button is held."""
        position: QPoint = event.position()
        if self.rotate and event.buttons() == Qt.MouseButton.LeftButton:
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
        elif self.translate and event.buttons() == Qt.MouseButton.RightButton:
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the current rotate or pan interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotate = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.translate = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom the scene camera in and out."""
        num_pixels = event.angleDelta()
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
    win.resize(800, 600)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
