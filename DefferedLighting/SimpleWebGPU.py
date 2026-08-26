#!/usr/bin/env -S uv run --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
from FloorPipeline import FloorPipeline
from LightingPipeline import LightingPipeline
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget
from TeapotPipeline import TeapotPipeline
from wgpu.utils import get_default_device


class WebGPUScene(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deferred Rendering with 2 Lights")
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
        self.msaa_sample_count = 1

        self.texture_size = (1024, 1024)
        self.rotation = 0.0
        self.eye = Vec3(0.0, 2.0, 4.0)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))

        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 100.0, PerspMode.WebGPU
        )
        self.text_buffer: list[tuple[int, int, str, int, str, QColor]] = []
        self.frame_buffer = None
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        # get the device pixel ratio for mac displays.
        self.ratio = self.devicePixelRatio()
        # create the numpy buffer for the final framebuffer render
        self._initialize_buffer()
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
            self._create_g_buffer()
            self._create_render_buffer()
            self._init_buffers()
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:  # noqa: BLE001 - wgpu exposes backend-specific errors.
            print(f"Failed to initialize WebGPU: {e}")
            sys.exit(1)

    def _init_buffers(self):
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_g_buffer(self):
        width = int(self.ratio * self.width())
        height = int(self.ratio * self.height())
        self.g_buffer = {
            "position": self.device.create_texture(
                size=(width, height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format=wgpu.TextureFormat.rgba16float,
            ),
            "normal": self.device.create_texture(
                size=(width, height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format=wgpu.TextureFormat.rgba16float,
            ),
            "albedo": self.device.create_texture(
                size=(width, height, 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                format=wgpu.TextureFormat.rgba8unorm,
            ),
        }
        self.g_buffer_views = {
            "position": self.g_buffer["position"].create_view(),
            "normal": self.g_buffer["normal"].create_view(),
            "albedo": self.g_buffer["albedo"].create_view(),
        }

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        width = self.ratio * self.width()
        height = self.ratio * self.height()
        teapot_pipeline = TeapotPipeline(
            self.device,
            self.eye,
            self.view,
            self.project,
            width,
            height,
        )
        self.pipelines.append(teapot_pipeline)
        self.pipelines.append(
            FloorPipeline(
                self.device,
                self.eye,
                self.view,
                self.project,
                width,
                height,
            )
        )
        self.pipelines.append(
            LightingPipeline(
                self.device,
                self.g_buffer,
                teapot_pipeline.view_buffer,
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
        command_encoder = self.device.create_command_encoder()

        # Geometry Pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.g_buffer_views["position"],
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                },
                {
                    "view": self.g_buffer_views["normal"],
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                },
                {
                    "view": self.g_buffer_views["albedo"],
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                },
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        self.pipelines[0].draw(render_pass)
        self.pipelines[1].draw(render_pass)
        render_pass.end()

        # Lighting Pass
        self.pipelines[2].paint(command_encoder, self.colour_buffer_texture_view)

        self.device.queue.submit([command_encoder.finish()])
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
        self.pipelines[0].update_uniform_buffers(self.mouse_global_tx)
        self.pipelines[1].update_uniform_buffers(self.mouse_global_tx)

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Parameters
        ----------
            event
                The resize event object.
        """
        # Update the stored width and height, considering high-DPI displays
        # Update projection matrix
        self.project = perspective(
            45.0, width / height if height > 0 else 1, 0.1, 350.0, PerspMode.WebGPU
        )

        # Recreate render buffers for the new window size
        self._create_g_buffer()
        self._create_render_buffer()

        # Resize the numpy buffer to match new window dimensions
        if self.frame_buffer is not None:
            self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)
        for pipeline in self.pipelines:
            pipeline.width = width
            pipeline.height = height

        self.update()

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

        Parameters
        ----------
            event
                The QMouseEvent object containing the new mouse position.
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

        Parameters
        ----------
            event
                The QMouseEvent object.
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

        Parameters
        ----------
            event
                The QMouseEvent object.
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

        Parameters
        ----------
            event
                The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
        self.update()

    def start_update_timer(self, interval_ms: int) -> None:
        """
        Starts the update timer with the given interval.

        Parameters
        ----------
            interval_ms : int
                The interval in milliseconds.
        """
        self._update_timer.start(interval_ms)

    def stop_update_timer(self) -> None:
        """Stops the update timer."""
        self._update_timer.stop()

    def resizeEvent(self, event) -> None:
        """
        Called whenever the window is resized.

        Parameters
        ----------
            event
                The resize event object.
        """
        # Update the stored width and height, considering high-DPI displays
        width = int(event.size().width() * self.ratio)
        height = int(event.size().height() * self.ratio)
        self.texture_size = (width, height)

        self.resizeWebGPU(width, height)

        # Recreate render buffers for the new window size
        self._create_render_buffer()

        # Resize the numpy buffer to match new window dimensions
        if self.frame_buffer is not None:
            self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

        return super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        """
        Handle the paint event to render the WebGPU content.

        Parameters
        ----------
            event : QPaintEvent
                The paint event.
        """
        self.paintWebGPU()
        painter = QPainter(self)

        if self.frame_buffer is not None:
            self._present_image(painter, self.frame_buffer)
        # Define a base height for font scaling 600 is a sensible default for most desktop environments
        base_height = 600.0
        scale_factor = self.height() / base_height

        for x, y, text, size, font, colour in self.text_buffer:
            scaled_size = int(size * scale_factor)
            painter.setPen(colour)
            painter.setFont(QFont(font, scaled_size))
            draw_y = y
            if y < 0:
                draw_y = self.height() + y
            painter.drawText(x, draw_y, text)
        self.text_buffer.clear()

        return super().paintEvent(event)

    def _initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)
        self.texture_size = (width, height)

    def _create_render_buffer(self):
        # This is the texture that the multisampled texture will be resolved to
        colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture = colour_buffer_texture
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()

        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.msaa_sample_count,
        )
        self.depth_buffer_view = depth_texture.create_view()

        # Calculate aligned buffer size for texture copy
        buffer_size = self._calculate_aligned_buffer_size()
        self.readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def render_text(
        self,
        x: int,
        y: int,
        text: str,
        size: int = 10,
        font: str = "Arial",
        colour: QColor = Qt.black,
    ) -> None:
        """
        Add text to the buffer to be rendered on the canvas.

        The size of the text will be scaled based on the window's height.

        Parameters
        ----------
            x : int
                The x-coordinate of the text.
            y : int
                The y-coordinate of the text. A negative value will position the text relative to the bottom of the window.
            text : str
                The text to render.
            size : int, optional
                The base font size of the text. This will be scaled. Defaults to 10.
            font : str, optional
                The font family of the text. Defaults to "Arial".
            colour : QColor, optional
                The colour of the text. Defaults to Qt.black.
        """
        self.text_buffer.append((x, y, text, size, font, colour))

    def _calculate_aligned_row_size(self) -> int:
        """
        Calculate the aligned row size for texture copy operations.
        Many GPUs require row alignment to 256 or 512 bytes.
        """
        bytes_per_pixel = 4  # RGBA8 = 4 bytes per pixel
        raw_row_size = self.texture_size[0] * bytes_per_pixel

        # Align to 256 bytes (common GPU requirement)
        alignment = 256
        aligned_row_size = ((raw_row_size + alignment - 1) // alignment) * alignment

        return aligned_row_size

    def _calculate_aligned_buffer_size(self) -> int:
        """
        Calculate the aligned buffer size needed for texture copy operations.
        Many GPUs require row alignment to 256 or 512 bytes.
        """
        aligned_row_size = self._calculate_aligned_row_size()
        return aligned_row_size * self.texture_size[1]

    def _update_colour_buffer(self) -> None:
        """
        Update the colour buffer with the rendered texture data.
        """
        # Use the aligned row size calculation
        bytes_per_row = self._calculate_aligned_row_size()

        try:
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                {"texture": self.colour_buffer_texture},
                {
                    "buffer": self.readback_buffer,
                    "bytes_per_row": bytes_per_row,  # Aligned row stride
                    "rows_per_image": self.texture_size[
                        1
                    ],  # Number of rows in the texture
                },
                (
                    self.texture_size[0],
                    self.texture_size[1],
                    1,
                ),  # Copy size: width, height, depth
            )
            self.device.queue.submit([command_encoder.finish()])

            # Map the buffer for reading
            self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)

            # Access the mapped memory
            raw_data = self.readback_buffer.read_mapped()
            width, height = self.texture_size

            # Create a strided view of the raw data and then copy it to a contiguous array.
            # This is necessary because the raw data from the buffer includes padding bytes
            # to meet row alignment requirements, so we can't just reshape it.
            strided_view = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw_data, dtype=np.uint8),
                shape=(height, width, 4),
                strides=(bytes_per_row, 4, 1),
            )
            self.frame_buffer = np.copy(strided_view)

            # Unmap the buffer when done
            self.readback_buffer.unmap()
        except Exception as e:  # noqa: BLE001 - readback failures keep the last frame usable.
            print(f"Failed to update colour buffer: {e}")
            # Fallback: create a simple gray buffer if texture copy fails
            if self.frame_buffer is not None:
                self.frame_buffer.fill(128)

    def _present_image(self, painter, image_data: np.ndarray) -> None:
        """
        Present the image data on the canvas.

        Parameters
        ----------
            image_data : np.ndarray
                The image data to render.
        """
        height, width, _ = image_data.shape
        image = QImage(
            image_data.data,
            width,
            height,
            image_data.strides[0],
            QImage.Format.Format_RGBA8888,
        )

        rect1 = QRect(0, 0, width, height)
        rect2 = self.rect()
        painter.drawImage(rect2, image, rect1)


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
