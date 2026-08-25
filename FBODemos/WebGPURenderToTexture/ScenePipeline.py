import wgpu
from ncca.ngl import PrimData, Vec3


class ScenePipeline:
    """Second pass of the render-to-texture demo.

    Draws a ground plane and a sphere, both textured with the teapot texture
    produced by the first pass. Each object gets its own MVP so the sphere can
    sit above the plane while sharing the one texture and sampler.
    """

    def __init__(self, device, texture_view):
        self.device = device
        self.texture_view = texture_view
        self._create_geometry()
        self._create_render_pipeline()

    def _create_geometry(self) -> None:
        """Build the plane and sphere vertex buffers.

        Both come from PrimData as an interleaved position/normal/uv array with
        an 8 float stride; the second pass only needs position and uv.
        """
        plane = PrimData.triangle_plane(2.0, 2.0, 20, 20, Vec3(0, 1, 0))
        self.plane_size = plane.size // 8
        self.plane_buffer = self.device.create_buffer_with_data(
            data=plane, usage=wgpu.BufferUsage.VERTEX
        )
        sphere = PrimData.sphere(0.4, 80)
        self.sphere_size = sphere.size // 8
        self.sphere_buffer = self.device.create_buffer_with_data(
            data=sphere, usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """Create the textured pipeline plus a bind group per object."""
        with open("SceneShader.wgsl", "r") as f:
            shader_module = self.device.create_shader_module(code=f.read())

        self.sampler = self.device.create_sampler(
            mag_filter="linear", min_filter="linear"
        )
        # One 4x4 float MVP (64 bytes) per object.
        self.plane_uniform = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.sphere_uniform = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        self.pipeline = self.device.create_render_pipeline(
            label="scene_pipeline",
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
                            {"format": "float32x2", "offset": 24, "shader_location": 1},
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

        layout = self.pipeline.get_bind_group_layout(0)
        self.plane_bind_group = self._create_bind_group(layout, self.plane_uniform)
        self.sphere_bind_group = self._create_bind_group(layout, self.sphere_uniform)

    def _create_bind_group(self, layout, uniform_buffer):
        return self.device.create_bind_group(
            layout=layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": uniform_buffer,
                        "offset": 0,
                        "size": 64,
                    },
                },
                {"binding": 1, "resource": self.texture_view},
                {"binding": 2, "resource": self.sampler},
            ],
        )

    def update_uniforms(self, plane_mvp, sphere_mvp) -> None:
        """Upload the per object MVP matrices for this frame."""
        self.device.queue.write_buffer(
            self.plane_uniform, 0, plane_mvp.to_numpy().tobytes()
        )
        self.device.queue.write_buffer(
            self.sphere_uniform, 0, sphere_mvp.to_numpy().tobytes()
        )

    def render(self, render_pass) -> None:
        """Record the plane and sphere draws into an existing render pass."""
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.plane_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.plane_buffer)
        render_pass.draw(self.plane_size)
        render_pass.set_bind_group(0, self.sphere_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.sphere_buffer)
        render_pass.draw(self.sphere_size)
