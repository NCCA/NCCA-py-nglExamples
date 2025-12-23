#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, Vec3, look_at, ortho
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device

SIM_WIDTH = 200
SIM_HEIGHT = 200


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, num_points=10000):
        super().__init__()
        self.setWindowTitle("WebGPU 2D Pan and Zoom")
        self.device = None
        self.pipeline = None
        self.vertex_buffer = None
        self.num_points = num_points
        self.msaa_sample_count = 4
        self.project: Mat4 = ortho(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, -SIM_HEIGHT / 2, SIM_HEIGHT / 2, 0, 100
        )
        self.ratio = self.devicePixelRatio()
        self.animate = True
        self.project = Mat4()
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.is_panning = False
        self.last_mouse_pos = None

        self.gen_points()
        self._initialize_web_gpu()
        self.update()

    def gen_points(self):
        self.points = np.zeros((self.num_points, 7), dtype=np.float32)

        self.points[:, 0] = np.random.uniform(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, self.num_points
        )  # x
        self.points[:, 1] = np.random.uniform(
            -SIM_HEIGHT / 2, SIM_HEIGHT / 2, self.num_points
        )  # y
        self.points[:, 2:5] = np.random.uniform(0.2, 1.0, (self.num_points, 3))  # color
        self.points[:, 5] = np.random.uniform(-1, 1, self.num_points)  # vx
        self.points[:, 6] = np.random.uniform(-1, 1, self.num_points)  # vy

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._init_buffers()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(10)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _init_buffers(self):
        vertex_data = np.zeros((self.num_points, 6), dtype=np.float32)
        vertex_data[:, 0:2] = self.points[:, 0:2]
        vertex_data[:, 2] = 0.0
        vertex_data[:, 3:6] = self.points[:, 2:5]

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

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

        # This is the multisampled texture that will be rendered to
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

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

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("PointShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="particle_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.point_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
            },
        )

        # Create a uniform buffer
        self.uniform_data = np.zeros((), dtype=[("projection_matrix", "float32", (16))])

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            width: The new width of the window.
            height: The new height of the window.
        """
        aspect = width / height if height > 0 else 1
        self.project = ortho(-10 * aspect, 10 * aspect, -10, 10, -100.0, 100.0)
        self.update()

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(
            10,
            20,
            f"WebGPU 2D Pan and Zoom :- {self.num_points}",
            size=20,
            colour=Qt.yellow,
        )
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
                ],
                depth_stencil_attachment={
                    "view": self.depth_buffer_view,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            )
            self.update_uniform_buffers()
            render_pass.set_viewport(
                0, 0, self.texture_size[0], self.texture_size[1], 0, 1
            )
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.num_points)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        model = Mat4.translate(self.pan_x, self.pan_y, 0) @ Mat4.scale(
            self.zoom, self.zoom, 1.0
        )
        projection_matrix = (self.project @ model).to_numpy().astype(np.float32)

        self.uniform_data["projection_matrix"] = projection_matrix.flatten()
        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_S, Qt.Key_Space):
            self.animate = not self.animate
        elif key == Qt.Key_R:
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.zoom = 0.5
        elif key == Qt.Key_B:
            self.animate = not self.animate
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.is_panning:
                delta = event.position() - self.last_mouse_pos
                self.pan_x += delta.x() / (self.width() / 20 * self.zoom)
                self.pan_y -= delta.y() / (self.height() / 20 * self.zoom)
                self.last_mouse_pos = event.position()
                self.update()
        else:
            self.is_panning = False
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        elif delta < 0:
            self.zoom *= 0.9
        self.update()
        super().wheelEvent(event)

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def timerEvent(self, event) -> None:
        if self.animate:
            # Update velocities
            self.points[:, 0] += self.points[:, 5]  # x += vx
            self.points[:, 1] += self.points[:, 6]  # y += vy

            # Bounce off walls
            self.points[:, 5] = np.where(
                (self.points[:, 0] < -SIM_WIDTH / 2)
                | (self.points[:, 0] > SIM_WIDTH / 2),
                -self.points[:, 5],
                self.points[:, 5],
            )
            self.points[:, 6] = np.where(
                (self.points[:, 1] < -SIM_HEIGHT / 2)
                | (self.points[:, 1] > SIM_HEIGHT / 2),
                -self.points[:, 6],
                self.points[:, 6],
            )
            vertex_data = np.zeros((self.num_points, 6), dtype=np.float32)
            vertex_data[:, 0:2] = self.points[:, 0:2]
            vertex_data[:, 3:6] = self.points[:, 2:5]

            self.device.queue.write_buffer(self.vertex_buffer, 0, vertex_data.tobytes())

        self.update()


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU points demo")
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=10000,
        help="The number of points to generate.",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = WebGPUScene(num_points=args.points)
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
