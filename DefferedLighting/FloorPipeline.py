from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import PrimData, Transform, Vec3

SHADER_DIR = Path(__file__).parent


class FloorPipeline:
    def __init__(self, device, eye, view, project, width, height):
        self.device = device
        self.pipeline = None
        self.transform_buffer = None
        self.material_buffer = None
        self.bind_group_0 = None
        self.bind_group_1 = None
        self.transform_uniforms = None
        self.material_uniforms = None
        self.eye = eye
        self.view = view
        self.project = project
        self._create_render_pipeline()
        grid = PrimData.triangle_plane(20, 20, 1, 1, Vec3(0, 1, 0))
        self.grid_size = grid.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=grid, usage=wgpu.BufferUsage.VERTEX
        )
        self.width = width
        self.height = height

    def _create_render_pipeline(self):
        """
        Create a render pipeline.
        """
        with open(SHADER_DIR / "GBufferChecker.wgsl", "r") as f:
            shader_code = f.read()

        shader_module = self.device.create_shader_module(code=shader_code)
        # build pipeline for the floor
        self.pipeline = self.device.create_render_pipeline(
            label="floor_gbuffer_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,  # x,y,z,nx,ny,nz,u,v as per ngl
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
        # create uniform buffers
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
                "names": ["colour1", "colour2", "checkSize", "checkOn"],
                "formats": [(np.float32, 4), (np.float32, 4), np.float32, np.uint32],
                "offsets": [0, 16, 32, 36],
                "itemsize": 48,  # padded to a multiple of 16
            }
        )

        self.material_uniforms = np.zeros((), dtype=material_dtype)
        self.material_uniforms["colour1"] = (0.9, 0.9, 0.9, 1.0)
        self.material_uniforms["colour2"] = (0.6, 0.6, 0.6, 1.0)
        self.material_uniforms["checkSize"] = 60
        self.material_uniforms["checkOn"] = 1
        self.material_buffer = self.device.create_buffer_with_data(
            data=self.material_uniforms.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="material_uniform_buffer",
        )

        # Create bind groups
        self.bind_group_0 = self.device.create_bind_group(
            label="transform_bind_group",
            layout=self.pipeline.get_bind_group_layout(0),
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

        self.bind_group_1 = self.device.create_bind_group(
            label="material_light_bind_group",
            layout=self.pipeline.get_bind_group_layout(1),
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

    def set_projection(self, project) -> None:
        """
        Update the projection matrix after a window resize.

        Parameters
        ----------
            project : Mat4
                The new projection matrix.
        """
        self.project = project

    def update_uniform_buffers(self, model) -> None:
        """
        update the uniform buffers.
        """
        tx = Transform()
        tx.set_position(0, -0.45, 0)
        model_matrix = model @ tx.matrix()
        model_view = self.view @ model_matrix
        MVP = self.project @ model_view
        normal_matrix = model_matrix.copy()
        normal_matrix = normal_matrix.inverse().transposed()

        self.transform_uniforms["MVP"] = MVP.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.transform_uniforms["M"] = model_matrix.to_numpy()

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
        render_pass.draw(self.grid_size)
