#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlankWebGPU")
        self.pipeline = None
        self.msaa_sample_count = 4
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
        except Exception as e:  # noqa: BLE001 - wgpu exposes backend-specific errors.
            print(f"Failed to initialize WebGPU: {e}")

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline. First load the shader
        """
        with open("Shader.wgsl", "r") as shader:
            shader_code = shader.read()
        shader_module = self.device.create_shader_module(code=shader_code)

        ## create a triangle buffer to render not it needs to be float32
        vertices = np.array(
            [-0.5, -0.5, 0.0, 0.0, 0.5, 0.0, 0.5, -0.5, 0.0], dtype=np.float32
        )
        self.vertex_buffer = self.device.create_buffer(
            size=vertices.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(self.vertex_buffer, 0, vertices.tobytes())

        # Create a pipeline layout (no bind groups needed for this simple example)
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[])
        print(vertices.itemsize)
        self.pipeline = self.device.create_render_pipeline(
            label="template_pipeline",
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        # Define the structure of our vertex buffer
                        "array_stride": 3 * vertices.itemsize,  # 3 floats x 4 bytes
                        "step_mode": "vertex",
                        "attributes": [
                            # Attribute 0: Position (vec3<f32>)
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            multisample={
                "count": self.msaa_sample_count,
            },
        )

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Parameters
        ----------
            width
                The new width of the window.
            height
                The new height of the window.

        """

        self.update()

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        try:
            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.multisample_texture_view,
                        "resolve_target": self.colour_buffer_texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.3, 0.3, 0.3, 1.0),
                    }
                ]
            )
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.set_pipeline(self.pipeline)
            render_pass.draw(3)  # Draw 3 vertices
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:  # noqa: BLE001 - wgpu exposes backend-specific errors.
            print(f"Failed to paint WebGPU content: {e}")

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Parameters
        ----------
            event
                The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        self.update()

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
