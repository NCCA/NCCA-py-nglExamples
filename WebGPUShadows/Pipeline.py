import numpy as np
import wgpu
from MeshData import MeshData
from ncca.ngl import (
    FirstPersonCamera,
    PerspMode,
    PrimData,
    Prims,
    Vec3,
    look_at,
    perspective,
)

_FLOAT_SIZE = np.dtype(np.float32).itemsize
_TEXTURE_FORMAT = wgpu.TextureFormat.rgba8unorm
SHADOW_MAP_SIZE = 1024 * 2


class Pipeline:
    def __init__(self, device, camera: FirstPersonCamera):
        self.device = device
        self.camera = camera
        self.pipeline = None
        self.transform_buffer = None
        self.material_buffer = None
        self.light_buffer = None
        self.view_buffer = None
        self.bind_group_0 = None
        self.bind_group_1 = None
        self.scene_uniform_buffer = None
        self.scene_uniform_data = None
        self.light_uniforms = None
        self.eye = camera.eye
        self.view = camera.view
        self.project = camera.projection
        self.prim_buffers = {}
        self.mesh_data = MeshData(self.device)
        self._create_lights()
        self._create_buffers()
        self._create_shadow_stuff()
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

        self.mesh_data.create_buffers()

    def _create_shadow_stuff(self):
        self.shadow_map_size = (SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
        # This is the texture that the shadow map will be written to
        shadow_texture = self.device.create_texture(
            size=self.shadow_map_size,
            sample_count=1,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.shadow_view = shadow_texture.create_view()
        self.shadow_sampler = self.device.create_sampler(
            compare=wgpu.CompareFunction.less
        )
        light_pos = self.light_uniform_data[0]["light_pos"]
        self.light_eye = Vec3(light_pos[0], light_pos[1], light_pos[2])
        self.light_view = look_at(self.light_eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.light_project = perspective(60.0, 1.0, 0.1, 100.0, PerspMode.WebGPU)
        self._create_shadow_pipeline()

    def _create_shadow_pipeline(self):
        with open("ShadowShader.wgsl", "r") as f:
            shader_code = f.read()
            shader = self.device.create_shader_module(code=shader_code)
        label = "shadow_pipeline"
        vertex = {
            "module": shader,
            "entry_point": "vertex_main",
            "buffers": [
                {
                    "array_stride": 8 * _FLOAT_SIZE,
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
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

        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )

        self.shadow_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.mesh_data.storage_buffer}},
                {"binding": 1, "resource": {"buffer": self.scene_uniform_buffer}},
            ],
        )

        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        self.shadow_pipeline = self.device.create_render_pipeline(
            label=label,
            layout=layout,
            vertex=vertex,
            fragment=None,  # No fragment shader needed for depth-only pass
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
                "depth_bias": 2,  # slope_scale * 2^R + bias, where R is the depth format's resolution
                "depth_bias_slope_scale": 2.0,
                "depth_bias_clamp": 0.0,
            },
            multisample={"count": 1},
        )

    def _create_lights(self):
        # Single light setup
        self.light_uniform_data = np.zeros(
            (1),
            dtype=[("light_pos", "float32", (4)), ("light_diffuse", "float32", (4))],
        )
        self.light_uniform_data["light_pos"] = np.array(
            [0.0, 2.0, 2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data["light_diffuse"] = np.array(
            [1.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        self.light_uniform_buffer = self.device.create_buffer_with_data(
            data=self.light_uniform_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="light_uniform_data",
        )
        scene_dtype = np.dtype(
            [
                ("proj", "float32", (4, 4)),
                ("view", "float32", (4, 4)),
                ("light_proj", "float32", (4, 4)),
                ("light_view", "float32", (4, 4)),
                ("camera_pos", "float32", (4)),
                ("shadow_map_size", "float32", (4)),
            ]
        )
        self.scene_uniform_data = np.zeros((1), dtype=scene_dtype)
        self.scene_uniform_data["shadow_map_size"] = np.array(
            [SHADOW_MAP_SIZE, 0, 0, 0], dtype=np.float32
        )
        self.scene_uniform_buffer = self.device.create_buffer(
            size=self.scene_uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="scene_uniform_buffer",
        )

    def update_lights(self, one_state):
        off = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.light_uniform_data["light_diffuse"] = (
            np.array([100.0, 100.0, 100.0, 1.0], dtype=np.float32) if one_state else off
        )
        self.device.queue.write_buffer(
            self.light_uniform_buffer, 0, self.light_uniform_data
        )

    def update_light_positions(self, positions):
        if len(positions) != len(self.light_uniform_data):
            return

        for i, pos in enumerate(positions):
            self.light_uniform_data[i]["light_pos"] = pos.to_numpy()

        # Update the shadow-casting light's view matrix
        light0_pos = positions[0]
        self.light_eye = Vec3(light0_pos.x, light0_pos.y, light0_pos.z)
        self.light_view = look_at(self.light_eye, Vec3(0, 0, 0), Vec3(0, 1, 0))

        # Write the updated light data to the GPU
        self.device.queue.write_buffer(
            self.light_uniform_buffer, 0, self.light_uniform_data
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
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.depth},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.comparison},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
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
                {"binding": 2, "resource": self.shadow_view},
                {"binding": 3, "resource": self.shadow_sampler},
                {"binding": 4, "resource": {"buffer": self.scene_uniform_buffer}},
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
        # Update view and projection from camera each frame
        self.view = self.camera.view
        self.project = self.camera.projection
        # Update scene uniform data
        self.scene_uniform_data["proj"] = self.project.to_numpy()
        self.scene_uniform_data["view"] = self.view.to_numpy()
        self.scene_uniform_data["light_proj"] = self.light_project.to_numpy()
        self.scene_uniform_data["light_view"] = self.light_view.to_numpy()
        self.scene_uniform_data["camera_pos"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 1.0]
        )
        self.device.queue.write_buffer(
            self.scene_uniform_buffer, 0, self.scene_uniform_data
        )

        # 1. Update CPU-side storage buffer with new data from scene objects
        for name, transform, colour in scene_objects:
            self._update_mesh_storage_buffer(name, transform, colour)
        # 2. Upload the buffer to the GPU
        self.mesh_data.write_buffers()

        command_encoder = self.device.create_command_encoder()

        # 3. Shadow Pass
        shadow_pass = command_encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": self.shadow_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        shadow_pass.set_pipeline(self.shadow_pipeline)
        shadow_pass.set_bind_group(0, self.shadow_bind_group)
        shadow_pass.set_vertex_buffer(0, self.mesh_data.vertex_buffer)
        for name, _, _ in scene_objects:
            if "light" not in name:  # Don't render lights in shadow pass
                self._render_mesh(shadow_pass, name)
        shadow_pass.end()

        # 4. Main Render Pass
        render_pass = command_encoder.begin_render_pass(
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
        render_pass.set_viewport(0, 0, size[0], size[1], 0, 1)
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group_0)
        render_pass.set_vertex_buffer(0, self.mesh_data.vertex_buffer)
        for name, _, _ in scene_objects:
            self._render_mesh(render_pass, name)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])

    def _update_mesh_storage_buffer(self, name: str, model, colour) -> None:
        normal_matrix = model.copy()
        normal_matrix = normal_matrix.inverse().transposed()
        self.mesh_data.update_mesh_data(name, model, normal_matrix, colour)

    def _render_mesh(self, pass_encoder, name: str) -> None:
        """
        (Internal) Draws a single mesh using the consolidated buffers.
        """
        mesh_info = self.mesh_data.get_mesh_info(name)
        if mesh_info is None:
            return

        pass_encoder.draw(
            vertex_count=mesh_info["vertex_count"],
            instance_count=1,
            first_vertex=mesh_info["first_vertex"],
            first_instance=mesh_info["instance_index"],
        )
