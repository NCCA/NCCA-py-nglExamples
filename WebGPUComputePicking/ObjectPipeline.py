"""
WebGPU pipelines for the compute-picking demo.

Two things live here:

ObjectPipeline
    Renders the scene objects. Geometry for every mesh is consolidated into
    a single vertex buffer; per-object data (model / normal matrix / colour /
    selected flag / integer pick ID) lives in a storage buffer indexed by
    ``instance_index`` so all objects render in one pass. The same shader
    module provides two fragment entry points, so two render pipelines are
    built: the shaded one (MSAA rgba8unorm) and the ID one, which writes the
    object's integer ID to a single-sampled ``r32uint`` attachment. Integer
    targets cannot be multisampled - which is fine, because blending or
    averaging object IDs would be meaningless anyway.

PickResolver
    Owns the compute side: a kernel (PickCompute.wgsl) that inspects the 9x9
    pixel block around the click in the ID texture and atomicMin-reduces it
    to one u32. The CPU reads back 4 bytes instead of a whole image, and the
    IDs are real integers rather than IDs smuggled through float colours.
"""

from pathlib import Path

import numpy as np
import wgpu

_FLOAT = np.dtype(np.float32).itemsize
_STRIDE = 8 * _FLOAT  # interleaved position(3) + normal(3) + uv(2)
_CAPACITY = 64  # max number of object instances the storage buffer can hold

# must match PickCompute.wgsl
PICK_BLOCK = 9
_ID_BITS = 20
_NO_HIT = 0xFFFFFFFF

_INSTANCE_DTYPE = np.dtype(
    [
        ("model", "float32", (4, 4)),
        ("normal_matrix", "float32", (4, 4)),
        ("colour", "float32", (4,)),
        ("flags", "float32", (4,)),  # x = selected, y = pick id
    ]
)

_GLOBALS_DTYPE = np.dtype(
    [
        ("view", "float32", (4, 4)),
        ("projection", "float32", (4, 4)),
        ("light_pos", "float32", (4,)),
        ("light_diffuse", "float32", (4,)),
        ("params", "float32", (4,)),
    ]
)


class ObjectPipeline:
    """Diffuse + barycentric-wireframe pipeline with an integer-ID pick pipeline."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self._geometry: dict[str, np.ndarray] = {}
        self._mesh_info: dict[str, tuple[int, int]] = {}  # name -> (first, count)
        self._order: list[str] = []  # instance_index -> mesh name for this frame

        self.vertex_buffer: wgpu.GPUBuffer | None = None
        self.instance_data = np.zeros(_CAPACITY, dtype=_INSTANCE_DTYPE)
        self.globals_data = np.zeros((), dtype=_GLOBALS_DTYPE)

        self._build_buffers()

    # ------------------------------------------------------------------
    # geometry registration
    # ------------------------------------------------------------------
    def add_mesh(self, name: str, prim_data_flat: np.ndarray) -> None:
        """Register a mesh from a flat interleaved (pos,normal,uv) float array."""
        self._geometry[name] = np.asarray(prim_data_flat, dtype=np.float32).ravel()

    def build(self) -> None:
        """Consolidate all registered geometry and create both pipelines."""
        first = 0
        chunks = []
        for name, data in self._geometry.items():
            count = data.size // 8
            self._mesh_info[name] = (first, count)
            chunks.append(data)
            first += count
        vertex_data = np.concatenate(chunks).astype(np.float32)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
            label="object_vertex_buffer",
        )
        self._build_pipelines()

    def _build_buffers(self) -> None:
        self.globals_buffer = self.device.create_buffer(
            size=self.globals_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="object_globals_buffer",
        )
        self.instance_buffer = self.device.create_buffer(
            size=self.instance_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="object_instance_buffer",
        )

    def _build_pipelines(self) -> None:
        shader_src = (Path(__file__).parent / "ObjectShader.wgsl").read_text()
        shader = self.device.create_shader_module(code=shader_src)

        bind_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ],
        )
        self.bind_group = self.device.create_bind_group(
            layout=bind_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.instance_buffer}},
                {"binding": 1, "resource": {"buffer": self.globals_buffer}},
            ],
        )
        layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_layout])

        vertex_state = {
            "module": shader,
            "entry_point": "vertex_main",
            "buffers": [
                {
                    "array_stride": _STRIDE,
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
                        {
                            "shader_location": 1,
                            "offset": 3 * _FLOAT,
                            "format": "float32x3",
                        },
                    ],
                }
            ],
        }
        primitive_state = {
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }
        depth_state = {
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        }

        self.pipeline = self.device.create_render_pipeline(
            label="object_shaded_pipeline",
            layout=layout,
            vertex=vertex_state,
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive=primitive_state,
            depth_stencil=depth_state,
            multisample={"count": 4},
        )
        # ID pass: single-sampled r32uint target (uint formats can't MSAA,
        # and averaging object IDs would be nonsense anyway)
        self.pick_pipeline = self.device.create_render_pipeline(
            label="object_id_pipeline",
            layout=layout,
            vertex=vertex_state,
            fragment={
                "module": shader,
                "entry_point": "fragment_pick",
                "targets": [{"format": wgpu.TextureFormat.r32uint}],
            },
            primitive=primitive_state,
            depth_stencil=depth_state,
            multisample={"count": 1},
        )

    # ------------------------------------------------------------------
    # per-frame data
    # ------------------------------------------------------------------
    def update_globals(
        self,
        view: np.ndarray,
        projection: np.ndarray,
        light_pos: tuple[float, float, float],
        light_diffuse: tuple[float, float, float, float],
    ) -> None:
        self.globals_data["view"] = view
        self.globals_data["projection"] = projection
        self.globals_data["light_pos"] = (*light_pos, 1.0)
        self.globals_data["light_diffuse"] = light_diffuse
        self.device.queue.write_buffer(
            self.globals_buffer, 0, self.globals_data.tobytes()
        )

    def set_instances(self, instances: list[dict]) -> None:
        """instances: list of dicts with mesh/model/normal_matrix/colour/pick_id/selected."""
        self._order = []
        for i, inst in enumerate(instances):
            self._order.append(inst["mesh"])
            self.instance_data[i]["model"] = inst["model"]
            self.instance_data[i]["normal_matrix"] = inst["normal_matrix"]
            self.instance_data[i]["colour"] = inst["colour"]
            self.instance_data[i]["flags"] = (
                1.0 if inst["selected"] else 0.0,
                float(inst["pick_id"]),
                0.0,
                0.0,
            )
        self.device.queue.write_buffer(
            self.instance_buffer, 0, self.instance_data.tobytes()
        )

    def _draw(self, render_pass: wgpu.GPURenderPassEncoder, pipeline) -> None:
        if self.vertex_buffer is None:
            return
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        for i, name in enumerate(self._order):
            first, count = self._mesh_info[name]
            render_pass.draw(count, 1, first, i)

    def render(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        self._draw(render_pass, self.pipeline)

    def render_ids(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        self._draw(render_pass, self.pick_pipeline)


class PickResolver:
    """Reduces the ID texture to a single object ID with a compute shader."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        shader_src = (Path(__file__).parent / "PickCompute.wgsl").read_text()
        shader = self.device.create_shader_module(code=shader_src)

        self.params_buffer = device.create_buffer(
            size=16,  # vec2<i32> pos + padding
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="pick_params",
        )
        self.result_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC,
            label="pick_result",
        )
        self.readback_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            label="pick_readback",
        )

        self._bind_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {"sample_type": wgpu.TextureSampleType.uint},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ],
        )
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bind_layout])
        self.pipeline = device.create_compute_pipeline(
            label="pick_resolve_pipeline",
            layout=layout,
            compute={"module": shader, "entry_point": "pick_main"},
        )

    def resolve(self, id_texture_view, x: int, y: int) -> int | None:
        """Run the pick reduction at pixel (x, y); return the object ID or None.

        One dispatch of a single 9x9 workgroup; the only CPU readback is the
        4-byte packed result.
        """
        # bind group is cheap to rebuild and the texture view changes on resize
        bind_group = self.device.create_bind_group(
            layout=self._bind_layout,
            entries=[
                {"binding": 0, "resource": id_texture_view},
                {"binding": 1, "resource": {"buffer": self.params_buffer}},
                {"binding": 2, "resource": {"buffer": self.result_buffer}},
            ],
        )
        self.device.queue.write_buffer(
            self.params_buffer, 0, np.array([x, y, 0, 0], dtype=np.int32).tobytes()
        )
        # the kernel reduces with atomicMin, so seed with "no hit" (all ones)
        self.device.queue.write_buffer(
            self.result_buffer, 0, np.array([_NO_HIT], dtype=np.uint32).tobytes()
        )

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1, 1, 1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(self.result_buffer, 0, self.readback_buffer, 0, 4)
        self.device.queue.submit([encoder.finish()])

        self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        packed = int(np.frombuffer(self.readback_buffer.read_mapped(), np.uint32)[0])
        self.readback_buffer.unmap()

        if packed == _NO_HIT:
            return None
        return packed & ((1 << _ID_BITS) - 1)
