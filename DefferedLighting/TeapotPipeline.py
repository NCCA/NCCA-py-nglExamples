import numpy as np
import wgpu
from ncca.ngl import PrimData, Prims


class TeapotPipeline:
    def __init__(self, device, eye, view, project, width, height):
        self.device = device
        self.pipeline = None
        self.transform_buffer = None
        self.material_buffer = None
        self.view_buffer = None
        self.bind_group_0 = None
        self.bind_group_1 = None
        self.transform_uniforms = None
        self.material_uniforms = None
        self.view_uniforms = None
        self.eye = eye
        self.view = view
        self.project = project
        self._create_render_pipeline()
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )
        self.width = width
        self.height = height

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("GBuffer.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="teapot_gbuffer_pipeline",
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
                "targets": [
                    {"format": wgpu.TextureFormat.rgba16float},  # Position
                    {"format": wgpu.TextureFormat.rgba16float},  # Normal
                    {"format": wgpu.TextureFormat.rgba8unorm},  # Albedo
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 1,
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
                }
            ],
        )

    def update_uniform_buffers(self, model) -> None:
        """
        update the uniform buffers.
        """
        model_view = self.view @ model
        MVP = self.project @ model_view
        normal_matrix = model.copy()
        normal_matrix = normal_matrix.inverse().transposed()

        self.transform_uniforms["M"] = model.to_numpy()
        self.transform_uniforms["MVP"] = MVP.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()

        self.device.queue.write_buffer(
            buffer=self.transform_buffer,
            buffer_offset=0,
            data=self.transform_uniforms.tobytes(),
        )

    def draw(self, render_pass) -> None:
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group_0, [], 0, 999999)
        render_pass.set_bind_group(1, self.bind_group_1, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.teapot_size)
