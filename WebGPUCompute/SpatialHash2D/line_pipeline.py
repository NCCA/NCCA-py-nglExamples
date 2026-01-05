"""
Generic line rendering pipeline for WebGPU.
Handles line rendering with customizable width, color, and projection.
"""

from typing import Optional, Tuple, Union

import numpy as np
import wgpu
from webgpu_constants import NGLToWebGPU

_LINE_SHADER = """
// LineShader.wgsl
struct Uniforms {
    MVP: mat4x4<f32>,
    line_width: f32,
}

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.MVP * vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0); // Grey color for grid lines
}
"""


class LinePipeline:
    """
    A reusable pipeline for rendering lines in WebGPU.

    Features:
    - Line strips or line segments
    - Per-vertex or per-line colors
    - Configurable line width
    - Projection matrix support
    - MSAA support
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        data_type: str = "Vec3",
        texture_format: wgpu.TextureFormat = wgpu.TextureFormat.rgba8unorm,
        depth_format: wgpu.TextureFormat = wgpu.TextureFormat.depth24plus,
        msaa_sample_count: int = 4,
        stride: int = 0,
        topology: wgpu.PrimitiveTopology = wgpu.PrimitiveTopology.line_strip,
    ):
        """
        Initialize the line rendering pipeline.

        Args:
            device: WebGPU device
            texture_format: Color attachment format
            depth_format: Depth attachment format
            msaa_sample_count: Number of MSAA samples
            shader_path: Path to the WGSL shader file
            topology: Line topology (line_strip or line_list)
        """
        self.device = device
        self.texture_format = texture_format
        self.depth_format = depth_format
        self.msaa_sample_count = msaa_sample_count
        self.topology = topology

        self._data_type = data_type

        if stride != 0:
            self._stride = stride
        else:
            self._stride = NGLToWebGPU.stride_from_type(self._data_type)

        # Buffers
        self.vertex_buffer: Optional[wgpu.GPUBuffer] = None
        self.color_buffer: Optional[wgpu.GPUBuffer] = None
        self.uniform_buffer: Optional[wgpu.GPUBuffer] = None
        self.bind_group: Optional[wgpu.GPUBindGroup] = None

        # Uniform data
        self.uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (4, 4)),
                ("line_width", "float32"),
                ("padding", np.uint32, 3),
            ],
        )
        self.uniform_data["line_width"] = 1.0  # Default line width

        # Create the pipeline
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the render pipeline and buffers."""
        # Load shader
        shader_module = self.device.create_shader_module(code=_LINE_SHADER)

        # Create render pipeline
        self.pipeline = self.device.create_render_pipeline(
            label="line_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": self._stride,  # vec2 position
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": NGLToWebGPU.vertex_format(self._data_type),
                                "offset": 0,
                                "shader_location": 0,
                            },
                        ],
                    },
                    {
                        "array_stride": NGLToWebGPU.stride_from_type(
                            "Vec3"
                        ),  # vec3 color
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": NGLToWebGPU.vertex_format("Vec3"),
                                "offset": 0,
                                "shader_location": 1,
                            },
                        ],
                    },
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={
                "topology": self.topology,
                "strip_index_format": (
                    wgpu.IndexFormat.uint32
                    if self.topology == wgpu.PrimitiveTopology.line_strip
                    else None
                ),
            },
            depth_stencil={
                "format": self.depth_format,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        # Create uniform buffer
        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        # Create bind group
        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def set_data(
        self,
        positions: Union[np.ndarray, wgpu.GPUBuffer],
        colors: Optional[Union[np.ndarray, wgpu.GPUBuffer]] = None,
    ) -> None:
        """
        Set the point data for rendering.

        Args:
            positions: Nx2 array of point positions or a pre-existing GPUBuffer.
            colors: Nx3 array of point colors (RGB) or a pre-existing GPUBuffer.
                    If None, uses white.
        """
        # Handle positions
        if isinstance(positions, wgpu.GPUBuffer):
            # If a buffer is passed, we just use it.
            # Note: The old buffer is not destroyed, it is up to the caller to manage it.
            self.vertex_buffer = positions
            self.num_points = positions.size // self._stride
        else:  # numpy array
            self.num_points = len(positions)
            data_bytes = positions.astype(np.float32).tobytes()
            # Ensure buffer exists and is large enough
            if self.vertex_buffer is None or self.position_buffer.size < len(
                data_bytes
            ):
                if self.vertex_buffer:
                    self.vertex_buffer.destroy()
                self.vertex_buffer = self.device.create_buffer_with_data(
                    data=data_bytes,
                    usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
                )
            else:
                self.device.queue.write_buffer(self.position_buffer, 0, data_bytes)

        # Handle colors
        if isinstance(colors, wgpu.GPUBuffer):
            self.color_buffer = colors
        else:
            # We need to pad the color data to match the 16-byte stride for vec3.
            # Create a new array with 4 components (RGBA) and copy RGB data.
            # The vertex format is float32x3, so only the first 3 components are used.
            padded_colors = np.zeros((self.num_points, 4), dtype=np.float32)
            if colors is not None:  # numpy array
                padded_colors[:, :3] = colors.astype(np.float32)
            else:  # colors is None, create default white colors
                padded_colors[:, :3] = 1.0

            colors_bytes = padded_colors.tobytes()
            # Ensure buffer exists and is large enough
            if self.color_buffer is None or self.color_buffer.size < len(colors_bytes):
                if self.color_buffer:
                    self.color_buffer.destroy()
                self.color_buffer = self.device.create_buffer_with_data(
                    data=colors_bytes,
                    usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
                )
            else:
                self.device.queue.write_buffer(self.color_buffer, 0, colors_bytes)

    def update_uniforms(
        self, mvp: Optional[np.ndarray] = None, line_width: Optional[float] = None
    ) -> None:
        """
        Update uniform buffer values.

        Args:
            MVP: 4x4 projection matrix
            line_width: Width of lines
        """
        if mvp is not None:
            self.uniform_data["MVP"] = mvp

        if line_width is not None:
            self.uniform_data["line_width"] = line_width

        self.device.queue.write_buffer(
            self.uniform_buffer, 0, self.uniform_data.tobytes()
        )

    def render(
        self, render_pass: wgpu.GPURenderPassEncoder, num_vertices: Optional[int] = None
    ) -> None:
        """
        Render the lines.

        Args:
            render_pass: Active render pass encoder
            num_vertices: Number of vertices to render (defaults to all)
        """
        if self.vertex_buffer is None or self.color_buffer is None:
            return

        count = num_vertices if num_vertices is not None else self.num_vertices

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_vertex_buffer(1, self.color_buffer)
        render_pass.draw(count)

    def cleanup(self) -> None:
        """Release resources."""
        if self.vertex_buffer:
            self.vertex_buffer.destroy()
        if self.color_buffer:
            self.color_buffer.destroy()
        if self.uniform_buffer:
            self.uniform_buffer.destroy()
