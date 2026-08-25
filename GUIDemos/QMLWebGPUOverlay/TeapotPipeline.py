"""Diffuse teapot render pipeline for the WebGPU overlay scene.

Mirrors SimpleWebGPU/TeapotPipeline.py but uses a minimal single-light diffuse
shader (DiffuseShader.wgsl) and is driven by matrices/colour pushed in from the
ncca.ngl.qml panels each frame rather than a fixed material.
"""

from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PrimData, Prims, Vec3


class TeapotPipeline:
    """Owns the teapot vertex buffer, diffuse pipeline and its uniform buffers."""

    def __init__(self, device, light_pos: Vec3, width: int, height: int) -> None:
        self.device = device
        self.light_pos = light_pos
        self.width = width
        self.height = height
        self._create_render_pipeline()
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        shader_path = Path(__file__).parent / "DiffuseShader.wgsl"
        shader_code = shader_path.read_text()
        shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="diffuse_teapot_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,  # pos3, normal3, uv2 as per ngl
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
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
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 4,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

        # Transform UBO (MVP, MV, normalMatrix) - rewritten every frame.
        transform_dtype = np.dtype(
            [
                ("MVP", np.float32, (4, 4)),
                ("MV", np.float32, (4, 4)),
                ("normal_matrix", np.float32, (4, 4)),
            ]
        )
        self.transform_uniforms = np.zeros((), dtype=transform_dtype)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transform_uniform_buffer",
        )

        # Lighting UBO (Colour, lightPos, lightDiffuse). vec4s throughout to
        # keep the std140 layout trivial. Colour is rewritten when the panel
        # changes it; light values are constant.
        lighting_dtype = np.dtype(
            {
                "names": ["Colour", "lightPos", "lightDiffuse"],
                "formats": [(np.float32, 4), (np.float32, 4), (np.float32, 4)],
                "offsets": [0, 16, 32],
                "itemsize": 48,
            }
        )
        self.lighting_uniforms = np.zeros((), dtype=lighting_dtype)
        self.lighting_uniforms["Colour"] = (1.0, 1.0, 0.0, 1.0)
        self.lighting_uniforms["lightPos"] = (*self.light_pos.to_numpy(), 1.0)
        self.lighting_uniforms["lightDiffuse"] = (1.0, 1.0, 1.0, 1.0)
        self.lighting_buffer = self.device.create_buffer(
            size=self.lighting_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="lighting_uniform_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.transform_buffer,
                        "offset": 0,
                        "size": self.transform_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.lighting_buffer,
                        "offset": 0,
                        "size": self.lighting_buffer.size,
                    },
                },
            ],
        )

    def update_uniform_buffers(
        self, model: Mat4, view: Mat4, project: Mat4, colour: Vec3
    ) -> None:
        """Recompute the transform matrices and push colour/transforms to the GPU."""
        model_view = view @ model
        mvp = project @ model_view
        normal_matrix = model_view.copy().inverse().transposed()

        self.transform_uniforms["MVP"] = mvp.to_numpy()
        self.transform_uniforms["MV"] = model_view.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.device.queue.write_buffer(
            buffer=self.transform_buffer,
            buffer_offset=0,
            data=self.transform_uniforms.tobytes(),
        )

        self.lighting_uniforms["Colour"] = (colour.x, colour.y, colour.z, 1.0)
        self.device.queue.write_buffer(
            buffer=self.lighting_buffer,
            buffer_offset=0,
            data=self.lighting_uniforms.tobytes(),
        )

    def paint(self, texture_view, multi_sample_view, depth_buffer_view) -> None:
        """Render the teapot into the supplied colour/MSAA/depth views."""
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": multi_sample_view,
                    "resolve_target": texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_viewport(0, 0, self.width, self.height, 0, 1)
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.teapot_size)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
