#!/usr/bin/env -S uv run --script
"""
A PySide6 application demonstrating a textured cube using WebGPU.

This script sets up a window, initializes a WebGPU context, and renders a
textured cube. It includes standard mouse and keyboard controls for interacting
with the 3D scene (rotate, pan, zoom).
"""

import argparse
import sys
import traceback
from typing import Optional

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Image, Mat4, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication


class WebGPUTextureScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for rendering a textured cube.

    This class handles the specifics of setting up the WebGPU pipeline,
    creating buffers for a cube, loading a texture, and managing user
    input for camera control.

    Attributes
    ----------
        pipeline : Optional[wgpu.GPURenderPipeline]
            The render pipeline.
        msaa_sample_count : int
            The number of samples for multisample anti-aliasing.
        mouse_global_tx : Mat4
            The transformation matrix controlled by the mouse.
        view : Mat4
            The camera's view matrix.
        project : Mat4
            The camera's projection matrix.
        model_position : Vec3
            The position of the cube in the scene.
        rotate : bool
            Flag indicating if rotation mode is active.
        translate : bool
            Flag indicating if translation mode is active.
        spin_x_face : int
            The rotation angle around the X-axis.
        spin_y_face : int
            The rotation angle around the Y-axis.
        original_x_rotation : int
            The initial X position for mouse rotation.
        original_y_rotation : int
            The initial Y position for mouse rotation.
        original_x_pos : int
            The initial X position for mouse translation.
        original_y_pos : int
            The initial Y position for mouse translation.
        INCREMENT : float
            The movement speed for panning.
        ZOOM : float
            The movement speed for zooming.
    """

    def __init__(self) -> None:
        """
        Initializes the WebGPU scene, sets up camera and mouse attributes,
        and starts the WebGPU initialization process.
        """
        super().__init__()
        self.setWindowTitle("WebGPU Texture Cube")
        self.pipeline: Optional[wgpu.GPURenderPipeline] = None
        self.msaa_sample_count: int = 4

        # --- Camera and Transformation Attributes ---
        self.mouse_global_tx: Mat4 = Mat4()
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()

        # --- Mouse Control Attributes ---
        self.rotate: bool = False
        self.translate: bool = False
        self.spin_x_face: int = 0
        self.spin_y_face: int = 0
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.1

        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context, buffers, texture, and pipeline.
        This method orchestrates the setup of all necessary WebGPU resources.
        """
        try:
            self.device: wgpu.GPUDevice = wgpu.utils.get_default_device()
            self._create_render_buffer()  # From base class
            self._create_cube_buffers()
            self._create_texture()
            self._create_uniform_buffer()
            self._create_render_pipeline()

            self.view = look_at(eye=Vec3(0, 1, 4), look=Vec3(0, 0, 0), up=Vec3(0, 1, 0))

        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            traceback.print_exc()

    def _create_cube_buffers(self) -> None:
        """Create vertex and UV buffers for the cube."""
        # fmt: off
        # Define the 36 vertices for the 12 triangles that make up the cube.
        # Each vertex is defined by 3 float values (X, Y, Z).
        # The vertices are specified with counter-clockwise (CCW) winding order.
        vertices = np.array([
            # Back face (-Z)
            -1,  1, -1,   1,  1, -1,   -1, -1, -1,
             1,  1, -1,   1, -1, -1,   -1, -1, -1,
            # Front face (+Z)
            -1,  1,  1,  -1, -1,  1,    1,  1,  1,
             1,  1,  1,  -1, -1,  1,    1, -1,  1,
            # Top face (+Y)
            -1,  1, -1,  -1,  1,  1,    1,  1, -1,
             1,  1, -1,  -1,  1,  1,    1,  1,  1,
            # Bottom face (-Y)
            -1, -1, -1,   1, -1, -1,   -1, -1,  1,
            -1, -1,  1,   1, -1, -1,    1, -1,  1,
            # Left face (-X)
            -1,  1, -1,  -1, -1, -1,   -1,  1,  1,
            -1,  1,  1,  -1, -1, -1,   -1, -1,  1,
            # Right face (+X)
             1,  1, -1,   1,  1,  1,    1, -1, -1,
             1, -1, -1,   1,  1,  1,    1, -1,  1
        ], dtype=np.float32)

        # Define the UV coordinates corresponding to each vertex.
        # Each UV coordinate is defined by 2 float values (U, V).
        uvs = np.array([
            # back face
            0, 1,   1, 1,   0, 0,   1, 1,   1, 0,   0, 0,
            # front face
            0, 1,   0, 0,   1, 1,   1, 1,   0, 0,   1, 0,
            # top face
            0, 0,   0, 1,   1, 0,   1, 0,   0, 1,   1, 1,
            # bottom face
            0, 1,   1, 1,   0, 0,   0, 0,   1, 1,   1, 0,
            # left face
            1, 1,   1, 0,   0, 1,   0, 1,   1, 0,   0, 0,
            # right face
            0, 1,   0, 0,   1, 1,   1, 1,   0, 0,   1, 0,
        ], dtype=np.float32)
        # fmt: on
        self.vertex_buffer: wgpu.GPUBuffer = self.device.create_buffer_with_data(
            data=vertices, usage=wgpu.BufferUsage.VERTEX
        )
        self.uv_buffer: wgpu.GPUBuffer = self.device.create_buffer_with_data(
            data=uvs, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_texture(self) -> None:
        """Load image and create texture and sampler."""
        try:
            image = Image("crate.bmp")
            rgb_data = image.get_pixels()
            # wgpu expects RGBA data, so we convert RGB to RGBA if needed
            if rgb_data.shape[2] == 3:
                image_data = np.empty((*rgb_data.shape[:2], 4), dtype=np.uint8)
                image_data[:, :, :3] = rgb_data
                image_data[:, :, 3] = 255  # Set alpha to fully opaque
            else:
                image_data = rgb_data  # Assume it's already compatible
        except FileNotFoundError:
            print("Error: crate.bmp not found. Creating a placeholder texture.")
            # Create a placeholder checkerboard texture if the image is not found.
            size = 64
            image_data = np.zeros((size, size, 4), dtype=np.uint8)
            image_data[0 : size // 2, 0 : size // 2] = [255, 0, 255, 255]  # Magenta
            image_data[size // 2 : size, 0 : size // 2] = [0, 0, 0, 255]  # Black
            image_data[0 : size // 2, size // 2 : size] = [0, 0, 0, 255]  # Black
            image_data[size // 2 : size, size // 2 : size] = [
                255,
                0,
                255,
                255,
            ]  # Magenta

        texture_size = (image_data.shape[1], image_data.shape[0], 1)
        self.texture: wgpu.GPUTexture = self.device.create_texture(
            size=texture_size,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.texture_view: wgpu.GPUTextureView = self.texture.create_view()
        self.sampler: wgpu.GPUSampler = self.device.create_sampler()

        bytes_per_pixel = 4
        # Write the image data to the GPU texture.
        self.device.queue.write_texture(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            image_data.tobytes(),
            {
                "bytes_per_row": image_data.shape[1] * bytes_per_pixel,
                "rows_per_image": image_data.shape[0],
            },
            texture_size,
        )

    def _create_uniform_buffer(self) -> None:
        """Create a uniform buffer for the MVP matrix."""
        # The buffer size is 64 bytes, which is the size of a 4x4 float matrix (4*4*4 bytes).
        self.uniform_buffer: wgpu.GPUBuffer = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def _create_render_pipeline(self) -> None:
        """Create the render pipeline, including shader and layouts."""
        try:
            with open("TextureShader.wgsl", "r") as f:
                shader_code = f.read()
        except FileNotFoundError:
            print("Error: TextureShader.wgsl not found. Cannot create pipeline.")
            return
        shader_module: wgpu.GPUShaderModule = self.device.create_shader_module(
            code=shader_code
        )

        # Define the layout of data that will be bound to the pipeline.
        self.bind_group_layout: wgpu.GPUBindGroupLayout = (
            self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,  # Corresponds to @binding(0) in the shader
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,  # Corresponds to @binding(1) in the shader
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float},
                    },
                    {
                        "binding": 2,  # Corresponds to @binding(2) in the shader
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                ]
            )
        )

        # Create the bind group, which is an instance of the layout with actual resources.
        self.bind_group: wgpu.GPUBindGroup = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": 64,
                    },
                },
                {"binding": 1, "resource": self.texture_view},
                {"binding": 2, "resource": self.sampler},
            ],
        )

        pipeline_layout: wgpu.GPUPipelineLayout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        # Create the render pipeline itself.
        self.pipeline = self.device.create_render_pipeline(
            label="textured_cube_pipeline",
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {  # Vertex buffer for position
                        "array_stride": 3 * 4,  # 3 floats * 4 bytes/float
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    },
                    {  # Vertex buffer for UV coordinates
                        "array_stride": 2 * 4,  # 2 floats * 4 bytes/float
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 1}
                        ],
                    },
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.back,  # Cull back-facing triangles
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Update projection matrix on window resize.

        Parameters
        ----------
            width : int
                The new width of the window.
            height : int
                The new height of the window.
        """
        self.project = perspective(45.0, float(width) / height, 0.01, 100.0)
        self.update()

    def paintWebGPU(self) -> None:
        """Render the scene."""
        if not self.pipeline:
            return
        # Calculate model transformation from mouse input
        rot_x = Mat4.rotate_x(self.spin_x_face)
        rot_y = Mat4.rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Apply translation
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        # Calculate the final Model-View-Projection (MVP) matrix.
        mvp = self.project @ self.view @ self.mouse_global_tx
        self.device.queue.write_buffer(self.uniform_buffer, 0, mvp.to_numpy().tobytes())

        # Create a command encoder to record rendering commands.
        command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder()
        render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
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

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 99)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_vertex_buffer(1, self.uv_buffer)
        render_pass.draw(36, 1, 0, 0)  # Draw 36 vertices (12 triangles)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events.

        Parameters
        ----------
            event : QKeyEvent
                The key event.
        """
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Space:
            # Reset camera and model position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handle mouse move events for rotation and translation.

        Parameters
        ----------
            event : QMouseEvent
                The mouse event.
        """
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handle mouse press events to initiate rotation or translation.

        Parameters
        ----------
            event : QMouseEvent
                The mouse event.
        """
        position: QPoint = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Handle mouse release events to end rotation or translation.

        Parameters
        ----------
            event : QMouseEvent
                The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotate = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.translate = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """

        Handle wheel events for zooming.

        Parameters
        ----------
            event : QWheelEvent
                The wheel event.
        """
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


def main() -> None:
    """
    The main entry point for the application.
    Initializes the QApplication, creates the main window, and starts the event loop.
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

    win = WebGPUTextureScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
