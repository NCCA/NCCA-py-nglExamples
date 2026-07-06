"""
Custom WebGPU pipeline for the selectable scene objects.

The stock ``PipelineFactory`` pipelines are flat-colour only, so the objects -
which need diffuse shading *and* the single-pass barycentric wireframe used to
mark a selection - get this one bespoke pipeline. Everything else in the demo
(grid, gizmo, and the picking pass for the gizmo) uses the factory pipelines.

Geometry for every object mesh is consolidated into a single vertex buffer.
Per-object data (model / normal matrix / colour / pick colour / selected flag)
lives in a storage buffer indexed by ``instance_index`` so all objects render
in one pass. A ``render_mode`` uniform switches the shader between diffuse
shading and flat colour-ID output for picking.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import wgpu

_FLOAT = np.dtype(np.float32).itemsize
_STRIDE = 8 * _FLOAT  # interleaved position(3) + normal(3) + uv(2)
_CAPACITY = 64  # max number of object instances the storage buffer can hold

_INSTANCE_DTYPE = np.dtype(
    [
        ("model", "float32", (4, 4)),
        ("normal_matrix", "float32", (4, 4)),
        ("colour", "float32", (4,)),
        ("pick_colour", "float32", (4,)),
        ("flags", "float32", (4,)),  # x = selected
    ]
)

_GLOBALS_DTYPE = np.dtype(
    [
        ("view", "float32", (4, 4)),
        ("projection", "float32", (4, 4)),
        ("light_pos", "float32", (4,)),
        ("light_diffuse", "float32", (4,)),
        ("params", "float32", (4,)),  # x = render_mode (0 shade, 1 pick)
    ]
)


class ObjectPipeline:
    """Diffuse + barycentric-wireframe pipeline with colour-ID pick mode."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self._geometry: Dict[str, np.ndarray] = {}
        self._mesh_info: Dict[str, Tuple[int, int]] = {}  # name -> (first, count)
        self._order: List[str] = []  # instance_index -> mesh name for this frame

        self.vertex_buffer: wgpu.GPUBuffer | None = None
        self.instance_data = np.zeros(_CAPACITY, dtype=_INSTANCE_DTYPE)
        self.globals_data = np.zeros((), dtype=_GLOBALS_DTYPE)

        self._build_globals_buffer()

    # ------------------------------------------------------------------
    # geometry registration
    # ------------------------------------------------------------------
    def add_mesh(self, name: str, prim_data_flat: np.ndarray) -> None:
        """Register a mesh from a flat interleaved (pos,normal,uv) float array."""
        self._geometry[name] = np.asarray(prim_data_flat, dtype=np.float32).ravel()

    def build(self) -> None:
        """Consolidate all registered geometry and create the pipeline."""
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
        self._build_pipeline()

    def _build_globals_buffer(self) -> None:
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

    def _build_pipeline(self) -> None:
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

        self.pipeline = self.device.create_render_pipeline(
            label="object_diffuse_wireframe_pipeline",
            layout=layout,
            vertex={
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
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": 4},
        )

    # ------------------------------------------------------------------
    # per-frame data
    # ------------------------------------------------------------------
    def update_globals(
        self,
        view: np.ndarray,
        projection: np.ndarray,
        light_pos: Tuple[float, float, float],
        light_diffuse: Tuple[float, float, float, float],
        render_mode: int,
    ) -> None:
        self.globals_data["view"] = view
        self.globals_data["projection"] = projection
        self.globals_data["light_pos"] = (*light_pos, 1.0)
        self.globals_data["light_diffuse"] = light_diffuse
        self.globals_data["params"] = (float(render_mode), 0.0, 0.0, 0.0)
        self.device.queue.write_buffer(
            self.globals_buffer, 0, self.globals_data.tobytes()
        )

    def set_instances(self, instances: List[dict]) -> None:
        """instances: list of dicts with mesh/model/normal_matrix/colour/pick_colour/selected."""
        self._order = []
        for i, inst in enumerate(instances):
            self._order.append(inst["mesh"])
            self.instance_data[i]["model"] = inst["model"]
            self.instance_data[i]["normal_matrix"] = inst["normal_matrix"]
            self.instance_data[i]["colour"] = inst["colour"]
            self.instance_data[i]["pick_colour"] = inst["pick_colour"]
            self.instance_data[i]["flags"] = (
                1.0 if inst["selected"] else 0.0,
                0.0,
                0.0,
                0.0,
            )
        self.device.queue.write_buffer(
            self.instance_buffer, 0, self.instance_data.tobytes()
        )

    def render(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        if self.vertex_buffer is None:
            return
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        for i, name in enumerate(self._order):
            first, count = self._mesh_info[name]
            render_pass.draw(count, 1, first, i)
