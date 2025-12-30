"""
Generic point rendering pipeline for WebGPU.
Handles point rendering with customizable size, color, and projection.
"""

from typing import Optional, Tuple, Union

import numpy as np
import wgpu
from ncca.ngl import Vec3
from webgpu_constants import NGLToWebGPU

_POINT_SHADER = """
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    MVP : mat4x4<f32>,
    size: f32,
};

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) colour: vec3<f32>,
};

// We now need to pass uv to the fragment shader
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColour: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexIn, @builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var output: VertexOut;
    let quad_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), // bottom-left
        vec2<f32>(1.0, -1.0),  // bottom-right
        vec2<f32>(-1.0, 1.0),   // top-left
        vec2<f32>(1.0, 1.0)    // top-right
    );

    let offset = quad_offsets[vertex_index];
    let pos = vec4<f32>(input.position.xy + offset * uniforms.size, 0.0, 1.0);

    output.position = uniforms.MVP * pos;
    output.fragColour = input.colour;
    // convert offset from -1 -> 1 to 0 -> 1 for uv
    output.uv = offset * 0.5 + 0.5;

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    let center = vec2<f32>(0.5, 0.5); // Center of the quad in UV space
    let dist = distance(fragData.uv, center); // Distance from center
    let radius = 0.5; // Circle radius (quad is 1.0 in UV space)

    if (dist > radius)
    {
        discard; // Remove pixels outside the circle
    }

    return vec4<f32>(fragData.fragColour, 1.0); // Simple color output
}
"""


class PointPipeline:
    """
    A reusable pipeline for rendering points in WebGPU.

    Features:
    - Instanced rendering of points as quads
    - Per-point colors
    - Configurable point size
    - Model, View Projection matrix support pass a projection only for 2D
    - MSAA support
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        data_type,
        texture_format: wgpu.TextureFormat = wgpu.TextureFormat.rgba8unorm,
        depth_format: wgpu.TextureFormat = wgpu.TextureFormat.depth24plus,
        msaa_sample_count: int = 4,
        stride: int = 0,
    ):
        """
        Initialize the point rendering pipeline.

        Args:
            device: WebGPU device
            texture_format: Color attachment format
            depth_format: Depth attachment format
            msaa_sample_count: Number of MSAA samples
            shader_path: Path to the WGSL shader file
            stride: The stride of the vertex buffer. If 0, it is inferred from data_type.
        """
        self.device = device
        self.texture_format = texture_format
        self.depth_format = depth_format
        self.msaa_sample_count = msaa_sample_count
        self._data_type = data_type
        if stride != 0:
            self._stride = stride
        else:
            self._stride = NGLToWebGPU.stride_from_type(self._data_type)
        # Buffers
        self.position_buffer: Optional[wgpu.GPUBuffer] = None
        self.color_buffer: Optional[wgpu.GPUBuffer] = None
        self.uniform_buffer: Optional[wgpu.GPUBuffer] = None
        self.bind_group: Optional[wgpu.GPUBindGroup] = None

        # Uniform data
        self.uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (4, 4)),
                ("size", "float32"),
                ("padding", np.uint32, 3),
            ],
        )
        self.uniform_data["size"] = 1.0  # Default point size

        # Create the pipeline
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the render pipeline and buffers."""
        # Load shader
        shader_module = self.device.create_shader_module(code=_POINT_SHADER)

        # Create render pipeline
        self.pipeline = self.device.create_render_pipeline(
            label="point_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": self._stride,
                        "step_mode": "instance",
                        "attributes": [
                            {
                                "format": NGLToWebGPU.vertex_format(self._data_type),
                                "offset": 0,
                                "shader_location": 0,
                            },
                        ],
                    },
                    {
                        "array_stride": NGLToWebGPU.stride_from_type(Vec3),
                        "step_mode": "instance",
                        "attributes": [
                            {
                                "format": NGLToWebGPU.vertex_format(Vec3),
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
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip},
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
            label="point_pipeline_uniform_buffer",
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
            self.position_buffer = positions
            self.num_points = positions.size // self._stride
        else:  # numpy array
            self.num_points = len(positions)
            data_bytes = positions.astype(np.float32).tobytes()
            # Ensure buffer exists and is large enough
            if self.position_buffer is None or self.position_buffer.size < len(
                data_bytes
            ):
                if self.position_buffer:
                    self.position_buffer.destroy()
                self.position_buffer = self.device.create_buffer_with_data(
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
        self, mvp: Optional[np.ndarray] = None, point_size: Optional[float] = None
    ) -> None:
        """
        Update uniform buffer values.

        Args:
            mvp: 4x4 mode view projection matrix (can just be projection if 2D)
            point_size: Size of points in world units
        """
        if mvp is not None:
            self.uniform_data["MVP"] = mvp

        if point_size is not None:
            self.uniform_data["size"] = point_size

        self.device.queue.write_buffer(
            self.uniform_buffer, 0, self.uniform_data.tobytes()
        )

    def render(
        self, render_pass: wgpu.GPURenderPassEncoder, num_points: Optional[int] = None
    ) -> None:
        """
        Render the points.

        Args:
            render_pass: Active render pass encoder
            num_points: Number of points to render (defaults to all)
        """
        if self.position_buffer is None or self.color_buffer is None:
            return

        count = num_points if num_points is not None else self.num_points

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.position_buffer)
        render_pass.set_vertex_buffer(1, self.color_buffer)
        render_pass.draw(4, count)  # 4 vertices per quad, instanced

    def cleanup(self) -> None:
        """Release resources."""
        if self.position_buffer:
            self.position_buffer.destroy()
        if self.color_buffer:
            self.color_buffer.destroy()
        if self.uniform_buffer:
            self.uniform_buffer.destroy()
