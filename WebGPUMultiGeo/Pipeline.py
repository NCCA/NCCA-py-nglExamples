import numpy as np
import wgpu
from MeshData import MeshData
from ncca.ngl import (
    FirstPersonCamera,
    Mat4,
    PrimData,
    Prims,
    Vec3,
)

_FLOAT_SIZE = np.dtype(np.float32).itemsize
_TEXTURE_FORMAT = wgpu.TextureFormat.rgba8unorm


class Pipeline:
    """
    Manages the WebGPU render pipeline, buffers, and drawing process.

    This class encapsulates the setup of the render pipeline, creation of
    GPU buffers for mesh data, lights, and transformations, and the
    rendering of a scene composed of multiple objects.
    """

    def __init__(self, device: wgpu.GPUDevice, camera: FirstPersonCamera):
        """
        Initializes the render pipeline.

        Parameters
        ----------
            device : wgpu.GPUDevice
                The WebGPU device.
            camera : FirstPersonCamera
                The camera used for the scene.
        """
        self.device: wgpu.GPUDevice = device
        self.camera: FirstPersonCamera = camera
        self.pipeline: wgpu.GPURenderPipeline | None = None
        self.transforms_buffer: wgpu.GPUBuffer | None = None
        self.light_uniform_buffer: wgpu.GPUBuffer | None = None
        self.bind_group_0: wgpu.GPUBindGroup | None = None
        self.mesh_data: MeshData = MeshData(self.device)
        self.transforms_data: np.ndarray | None = None
        self.light_uniform_data: np.ndarray | None = None
        self.command_encoder: wgpu.GPUCommandEncoder | None = None
        self.render_pass: wgpu.GPURenderPassEncoder | None = None

        self._create_lights()
        self._create_global_transforms()
        self._create_buffers()
        self._create_render_pipeline()

    def _create_buffers(self) -> None:
        """
        (Internal) Creates and populates the mesh buffers.
        """
        for prim in Prims:
            try:
                self.mesh_data.add_geometry(prim.value, PrimData.primitive(prim))
                self.mesh_data.add_mesh(prim.value, prim.value)
                print(f"Added mesh: {prim.value}")
            except ValueError:
                continue

        self.mesh_data.add_geometry(
            "floor", PrimData.triangle_plane(10, 10, 20, 20, Vec3(0, 1, 0))
        )
        self.mesh_data.add_mesh("floor", "floor")
        # Add light geometry once
        light_geom = PrimData.sphere(0.1, 20).flatten()
        self.mesh_data.add_geometry("light", light_geom)

        # Create instances of the light
        self.mesh_data.add_mesh("light1", "light")
        self.mesh_data.add_mesh("light2", "light")
        self.mesh_data.add_mesh("light3", "light")

        self.mesh_data.create_buffers()

    def _create_global_transforms(self) -> None:
        """
        (Internal) Creates the uniform buffer for global transformations (view, projection).
        """
        self.transforms_data = np.zeros(
            1,
            dtype=[
                ("view", "float32", (16,)),
                ("projection", "float32", (16,)),
                ("camera_pos", "float32", (4,)),
            ],
        )

        self.transforms_buffer = self.device.create_buffer_with_data(
            data=self.transforms_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transforms_buffer",
        )

    def _create_lights(self) -> None:
        """
        (Internal) Creates the storage buffer for light data.
        """
        self.light_uniform_data = np.zeros(
            3,
            dtype=[
                ("light_pos", "float32", (4,)),
                ("light_diffuse", "float32", (4,)),
            ],
        )
        self.light_uniform_data[0]["light_pos"] = np.array(
            [0.0, 2.0, 2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data[1]["light_pos"] = np.array(
            [-2.0, 2.0, -2.0, 1.0], dtype=np.float32
        )
        self.light_uniform_data[2]["light_pos"] = np.array(
            [2.0, 2.0, -2.0, 1.0], dtype=np.float32
        )

        self.light_uniform_buffer = self.device.create_buffer_with_data(
            data=self.light_uniform_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="light_uniform_data",
        )

    def update_lights(
        self, one_state: bool, two_state: bool, three_state: bool
    ) -> None:
        """
        Updates the diffuse color of the lights based on their state.

        Parameters
        ----------
            one_state : bool
                State of the first light.
            two_state : bool
                State of the second light.
            three_state : bool
                State of the third light.
        """
        off = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if self.light_uniform_data is not None:
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
        (Internal) Creates the main render pipeline from the shader.
        """
        with open("DiffuseShader.wgsl", "r") as f:
            shader_code = f.read()
            shader = self.device.create_shader_module(code=shader_code)
        label = "diffuse_triangle_pipeline"
        vertex_layouts = [
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
        ]

        bind_group_layout_0 = self.device.create_bind_group_layout(
            label="vertex_storage_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ],
        )

        self.bind_group_0 = self.device.create_bind_group(
            label="vertex_uniform_bind_group",
            layout=bind_group_layout_0,
            entries=[
                {"binding": 0, "resource": {"buffer": self.mesh_data.storage_buffer}},
                {"binding": 1, "resource": {"buffer": self.light_uniform_buffer}},
                {"binding": 2, "resource": {"buffer": self.transforms_buffer}},
            ],
        )

        layout = self.device.create_pipeline_layout(
            label="diffuse_triangle_pipeline_layout",
            bind_group_layouts=[bind_group_layout_0],
        )

        self.pipeline = self.device.create_render_pipeline(
            label=label,
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": vertex_layouts,
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": _TEXTURE_FORMAT}],
            },
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
        texture_view: wgpu.GPUTextureView,
        multisample_texture_view: wgpu.GPUTextureView,
        depth_buffer_view: wgpu.GPUTextureView,
        size: tuple[int, int],
        scene_objects: list[tuple[str, Mat4, tuple[float, float, float, float]]],
    ) -> None:
        """
        Renders a complete scene with the given objects.

        This method consolidates the entire render pass, from updating buffers
        to submitting draw calls.

        Parameters
        ----------
            texture_view : wgpu.GPUTextureView
                The view to render to.
            multisample_texture_view : wgpu.GPUTextureView
                The multisample view.
            depth_buffer_view : wgpu.GPUTextureView
                The depth buffer view.
            size : Tuple[int, int]
                The viewport size (width, height).
            scene_objects : List[...]
                A list of objects to render, each defined as a
                                       tuple of (name, transform_matrix, color).
        """
        if self.transforms_data is None:
            return
        self.transforms_data["view"] = self.camera.view.to_numpy().flatten()
        self.transforms_data["projection"] = self.camera.projection.to_numpy().flatten()
        self.transforms_data["camera_pos"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 1.0],
            dtype=np.float32,
        )
        for name, transform, colour in scene_objects:
            self._update_mesh_storage_buffer(name, transform, colour)

        self._begin_render_pass(
            size, texture_view, multisample_texture_view, depth_buffer_view
        )

        for name, _, _ in scene_objects:
            self._render_mesh(name)

        self._end_render_pass()

    def _update_mesh_storage_buffer(
        self, name: str, model: Mat4, colour: tuple
    ) -> None:
        """
        (Internal) Update the storage buffer for a single mesh instance.
        """
        normal_matrix = model.copy()
        normal_matrix = normal_matrix.inverse().transposed()
        self.mesh_data.update_mesh_data(name, model, normal_matrix, colour)

    def _begin_render_pass(
        self,
        size: tuple[int, int],
        texture_view: wgpu.GPUTextureView,
        multisample_texture_view: wgpu.GPUTextureView,
        depth_buffer_view: wgpu.GPUTextureView,
    ) -> None:
        """
        (Internal) Begins a render pass and sets up the pipeline and buffers.
        """
        self.command_encoder = self.device.create_command_encoder()
        self.mesh_data.write_buffers()
        if self.transforms_buffer and self.transforms_data is not None:
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
        if self.pipeline and self.bind_group_0 and self.mesh_data.vertex_buffer:
            self.render_pass.set_pipeline(self.pipeline)
            self.render_pass.set_bind_group(0, self.bind_group_0)
            self.render_pass.set_vertex_buffer(0, self.mesh_data.vertex_buffer)

    def _render_mesh(self, name: str) -> None:
        """
        (Internal) Draws a single mesh instance.
        """
        mesh_info = self.mesh_data.get_mesh_info(name)
        if mesh_info is None or self.render_pass is None:
            return

        self.render_pass.draw(
            vertex_count=mesh_info["vertex_count"],
            instance_count=1,
            first_vertex=mesh_info["first_vertex"],
            first_instance=mesh_info["instance_index"],
        )

    def _end_render_pass(self) -> None:
        """
        (Internal) Ends the render pass and submits the command buffer.
        """
        if self.render_pass:
            self.render_pass.end()
        if self.command_encoder:
            self.device.queue.submit([self.command_encoder.finish()])
