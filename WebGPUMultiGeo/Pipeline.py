import numpy as np
import wgpu
from MeshData import MeshData
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective

_FLOAT_SIZE = np.dtype(np.float32).itemsize
_TEXTURE_FORMAT = wgpu.TextureFormat.rgba8unorm


class Pipeline:
    def __init__(self, device, camera):
        self.device = device
        self.pipeline = None
        self.transform_buffer = None
        self.material_buffer = None
        self.light_buffer = None
        self.view_buffer = None
        self.bind_group_0 = None
        self.vertex_uniforms = None
        self.light_uniforms = None
        self.camera = camera
        self.prim_buffers = {}
        self.mesh_data = MeshData(self.device)
        self._create_lights()
        self._create_global_transforms()
        self._create_buffers()
        self._create_render_pipeline()

    def _create_buffers(self):
        for prim in Prims:
            try:
                self.mesh_data.add_mesh(prim.value, PrimData.primitive(prim))
                print(f"Added mesh: {prim.value}")
            except Exception:
                pass  # some prims need to call the create functions instead

        self.mesh_data.add_mesh(
            "floor", PrimData.triangle_plane(10, 10, 20, 20, Vec3(0, 1, 0))
        )
        self.mesh_data.add_mesh("light1", PrimData.sphere(0.1, 20).flatten())
        self.mesh_data.add_mesh("light2", PrimData.sphere(0.1, 20).flatten())
        self.mesh_data.add_mesh("light3", PrimData.sphere(0.1, 20).flatten())

        self.mesh_data.create_buffers()

    def _create_global_transforms(self):
        self.transforms_data = np.zeros(
            1,
            dtype=[
                ("view", "float32", (16)),
                ("projection", "float32", (16)),
            ],
        )
        self.transforms_data["view"] = Mat4().to_numpy().flatten()
        self.transforms_data["projection"] = Mat4().to_numpy().flatten()

        self.transforms_buffer = self.device.create_buffer_with_data(
            data=self.transforms_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transforms_buffer",
        )

    def _create_lights(self):
        # going to use 3 point lighting here
        self.light_uniform_data = np.zeros(
            (3),
            dtype=[
                ("light_pos", "float32", (4)),
                ("light_diffuse", "float32", (4)),
                ("camera_pos", "float32", (4)),
            ],
        )
        self.light_uniform_data[0]["light_pos"] = np.array(
            [0.0, 2.0, 2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data[0]["camera_pos"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 1.0],
            dtype=np.float32,
        )

        self.light_uniform_data[1]["light_pos"] = np.array(
            [-2.0, 2.0, -2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data[1]["camera_pos"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 1.0],
            dtype=np.float32,
        )

        self.light_uniform_data[2]["light_pos"] = np.array(
            [2.0, 2.0, -2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data[2]["camera_pos"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 1.0],
            dtype=np.float32,
        )

        self.light_uniform_buffer = self.device.create_buffer_with_data(
            data=self.light_uniform_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="light_uniform_data",
        )

    def update_lights(self, one_state, two_state, three_state):
        off = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.light_uniform_data[0]["light_diffuse"] = (
            np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32) if one_state else off
        )
        self.light_uniform_data[1]["light_diffuse"] = (
            np.array([0.4, 0.4, 0.4, 1.0], dtype=np.float32) if two_state else off
        )
        self.light_uniform_data[2]["light_diffuse"] = (
            np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32) if three_state else off
        )
        self.device.queue.write_buffer(
            self.light_uniform_buffer, 0, self.light_uniform_data.tobytes()
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("DiffuseShader.wgsl", "r") as f:
            shader_code = f.read()
            shader = self.device.create_shader_module(code=shader_code)
        label = "diffuse_triangle_pipeline"
        vertex = {
            "module": shader,
            "entry_point": "vertex_main",
            "buffers": [
                {
                    "array_stride": 8 * _FLOAT_SIZE,  # x,y,z nx,ny,nz,u,v
                    "attributes": [
                        {
                            "shader_location": 0,
                            "offset": 0 * _FLOAT_SIZE,
                            "format": "float32x3",
                        },
                        {
                            "shader_location": 1,
                            "offset": 3 * _FLOAT_SIZE,
                            "format": "float32x3",
                        },
                        {
                            "shader_location": 2,
                            "offset": 6 * _FLOAT_SIZE,
                            "format": "float32x2",
                        },
                    ],
                }
            ],
        }
        fragment = {
            "module": shader,
            "entry_point": "fragment_main",
            "targets": [{"format": _TEXTURE_FORMAT}],
        }

        bind_group_layout_0 = self.device.create_bind_group_layout(
            label="vertex_storage_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                        "has_dynamic_offset": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                        "has_dynamic_offset": False,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "has_dynamic_offset": False,
                    },
                },
            ],
        )

        # Create the bind group
        self.bind_group_0 = self.device.create_bind_group(
            label="vertex_uniform_bind_group",
            layout=bind_group_layout_0,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.mesh_data.storage_buffer},
                },
                {"binding": 1, "resource": {"buffer": self.light_uniform_buffer}},
                {"binding": 2, "resource": {"buffer": self.transforms_buffer}},
            ],
        )

        layout = self.device.create_pipeline_layout(
            label="diffuse_triangle_pipeline_layout",
            bind_group_layouts=[bind_group_layout_0],
        )
        # finally create the pipeline
        self.pipeline = self.device.create_render_pipeline(
            label=label,
            layout=layout,
            vertex=vertex,
            fragment=fragment,
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
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

    def render(
        self,
        texture_view,
        multisample_texture_view,
        depth_buffer_view,
        size,
        scene_objects,
    ):
        """
        Renders a complete scene with the given objects.
        This method consolidates the entire render pass.
        """
        # This sets the camera shared by all objects
        self.transforms_data["view"] = self.camera.view.to_numpy().flatten()
        self.transforms_data["projection"] = self.camera.projection.to_numpy().flatten()
        # 1. Update CPU-side storage buffer with new data from scene objects
        for name, transform, colour in scene_objects:
            self._update_mesh_storage_buffer(name, transform, colour)

        # 2. Begin the render pass (this also uploads the buffer to the GPU)
        self._begin_render_pass(
            size, texture_view, multisample_texture_view, depth_buffer_view
        )

        # 3. Issue draw calls for each object
        for name, _, _ in scene_objects:
            self._render_mesh(name)

        # 4. End the render pass and submit
        self._end_render_pass()

    def _update_mesh_storage_buffer(
        self, name: str, model: Mat4, colour: tuple
    ) -> None:
        """
        (Internal) Update the storage buffer for a single mesh.
        """

        normal_matrix = model.copy()
        normal_matrix.inverse().transpose()
        self.mesh_data.update_mesh_data(
            name,
            model.to_numpy(),
            normal_matrix.to_numpy(),
            colour,
        )

    def _begin_render_pass(
        self, size, texture_view, multisample_texture_view, depth_buffer_view
    ):
        self.command_encoder = self.device.create_command_encoder()
        # Before rendering, write all the updated mesh data to the GPU buffer
        self.mesh_data.write_buffers()
        self.device.queue.write_buffer(
            self.transforms_buffer, 0, self.transforms_data.tobytes()
        )

        self.render_pass = self.command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": multisample_texture_view,
                    "resolve_target": texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.3, 0.3, 0.3, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        self.render_pass.set_viewport(0, 0, size[0], size[1], 0, 1)
        self.render_pass.set_pipeline(self.pipeline)
        # Set the bind group once. The shader will use instance_index to get the right data.
        self.render_pass.set_bind_group(0, self.bind_group_0)
        # Set the consolidated vertex buffer once
        self.render_pass.set_vertex_buffer(0, self.mesh_data.vertex_buffer)

    def _render_mesh(self, name: str) -> None:
        """
        (Internal) Draws a single mesh using the consolidated buffers.
        """
        mesh_info = self.mesh_data.get_mesh_info(name)
        if mesh_info is None:
            return

        self.render_pass.draw(
            vertex_count=mesh_info["vertex_count"],
            instance_count=1,
            first_vertex=mesh_info["first_vertex"],
            first_instance=mesh_info["instance_index"],
        )

    def _end_render_pass(self) -> None:
        self.render_pass.end()
        self.device.queue.submit([self.command_encoder.finish()])
