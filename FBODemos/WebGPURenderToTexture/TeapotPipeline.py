import numpy as np
import wgpu
from ncca.ngl import PrimData, Prims


class TeapotPipeline:
    """First pass of the render-to-texture demo.

    Owns an offscreen colour texture and renders a PBR shaded teapot into it.
    The resolved single sample texture is exposed as ``texture_view`` so the
    second pass can sample it onto other geometry.
    """

    def __init__(self, device, eye, light_pos, view, project, width, height):
        self.device = device
        self.pipeline = None
        self.transform_buffer = None
        self.material_buffer = None
        self.light_buffer = None
        self.view_buffer = None
        self.bind_group_0 = None
        self.bind_group_1 = None
        self.transform_uniforms = None
        self.material_uniforms = None
        self.light_uniforms = None
        self.view_uniforms = None
        self.eye = eye
        self.light_pos = light_pos
        self.view = view
        self.project = project
        self.width = width
        self.height = height
        self.sample_count = 4
        self._create_render_pipeline()
        self._create_render_targets()
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_targets(self) -> None:
        """Create the offscreen textures the teapot is rendered into.

        We render into a multisampled colour target and resolve it into a
        single sample texture. Only the resolved texture can be sampled in the
        second pass, so that is the one we hand out via ``texture_view``.
        """
        size = (self.width, self.height, 1)
        # Resolved single sample colour texture - this is what pass two samples.
        self.colour_texture = self.device.create_texture(
            size=size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.texture_view = self.colour_texture.create_view()
        # Multisampled colour target we actually render into.
        self.msaa_texture = self.device.create_texture(
            size=size,
            sample_count=self.sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.msaa_view = self.msaa_texture.create_view()
        # Multisampled depth buffer to match the colour target.
        self.depth_texture = self.device.create_texture(
            size=size,
            sample_count=self.sample_count,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_view = self.depth_texture.create_view()

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("PBRShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="teapot_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
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

        # Create uniform buffers

        # Transforms UBO
        transform_dtype = np.dtype(
            [
                ("MVP", np.float32, (4, 4)),
                ("normal_matrix", np.float32, (4, 4)),
                ("M", np.float32, (4, 4)),
            ]
        )
        self.transform_uniforms = np.zeros((), dtype=transform_dtype)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transform_uniform_buffer",
        )

        # Material UBO
        # Note: WGSL structures have specific padding rules (std140).
        material_dtype = np.dtype(
            {
                "names": ["albedo", "metallic", "roughness", "ao"],
                "formats": [(np.float32, 3), np.float32, np.float32, np.float32],
                "offsets": [0, 12, 16, 20],
                "itemsize": 32,
            }
        )

        self.material_uniforms = np.zeros((), dtype=material_dtype)
        self.material_uniforms["albedo"] = (0.950, 0.71, 0.29)
        self.material_uniforms["metallic"] = 1.02
        self.material_uniforms["roughness"] = 0.38
        self.material_uniforms["ao"] = 0.2
        self.material_buffer = self.device.create_buffer_with_data(
            data=self.material_uniforms.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="material_uniform_buffer",
        )

        # Light UBO
        light_dtype = np.dtype(
            {
                "names": ["lightPosition", "lightColor"],
                "formats": [(np.float32, 3), (np.float32, 3)],
                "offsets": [0, 16],
                "itemsize": 32,
            }
        )
        self.light_uniforms = np.zeros((), dtype=light_dtype)
        self.light_uniforms["lightPosition"] = self.light_pos.to_numpy()
        self.light_uniforms["lightColor"] = (400.0, 400.0, 400.0)
        self.light_buffer = self.device.create_buffer_with_data(
            data=self.light_uniforms.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="light_uniform_buffer",
        )

        # View UBO
        view_dtype = np.dtype(
            {
                "names": ["camPos", "exposure"],
                "formats": [(np.float32, 3), np.float32],
                "offsets": [0, 12],
                "itemsize": 16,
            }
        )
        self.view_uniforms = np.zeros((), dtype=view_dtype)
        self.view_uniforms["camPos"] = self.eye.to_numpy()
        self.view_uniforms["exposure"] = 2.2
        self.view_buffer = self.device.create_buffer_with_data(
            data=self.view_uniforms.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="view_uniform_buffer",
        )

        # Create bind groups
        bind_group_layout_0 = self.pipeline.get_bind_group_layout(0)
        self.bind_group_0 = self.device.create_bind_group(
            layout=bind_group_layout_0,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.transform_buffer,
                        "offset": 0,
                        "size": self.transform_buffer.size,
                    },
                }
            ],
        )

        bind_group_layout_1 = self.pipeline.get_bind_group_layout(1)
        self.bind_group_1 = self.device.create_bind_group(
            layout=bind_group_layout_1,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.material_buffer,
                        "offset": 0,
                        "size": self.material_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.light_buffer,
                        "offset": 0,
                        "size": self.light_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.view_buffer,
                        "offset": 0,
                        "size": self.view_buffer.size,
                    },
                },
            ],
        )

    def update_uniform_buffers(self, model) -> None:
        """
        update the uniform buffers.
        """
        model_view = self.view @ model
        MVP = self.project @ model_view
        normal_matrix = model_view.copy()
        normal_matrix = normal_matrix.inverse().transposed()

        self.transform_uniforms["M"] = model_view.to_numpy()
        self.transform_uniforms["MVP"] = MVP.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()

        self.device.queue.write_buffer(
            buffer=self.transform_buffer,
            buffer_offset=0,
            data=self.transform_uniforms.tobytes(),
        )

    def paint(self) -> None:
        """Render the teapot into the offscreen colour texture.

        The resolved result is available afterwards via ``self.texture_view``.
        """
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.msaa_view,
                    "resolve_target": self.texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.4, 0.5, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_viewport(0, 0, self.width, self.height, 0, 1)
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group_0, [], 0, 999999)
        render_pass.set_bind_group(1, self.bind_group_1, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.teapot_size)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
