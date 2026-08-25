"""The WebGPU pipelines used to draw a BVH scene."""

from pathlib import Path

import numpy as np
import wgpu
from bvh_scene_webgpu import INSTANCE_DTYPE, LINE_VERTEX_DTYPE
from ncca.ngl import Mat4, PrimData

_CAMERA_DTYPE = np.dtype(
    [
        ("view_projection", np.float32, (4, 4)),
        ("light_position", np.float32, (4,)),
    ]
)


class BvhWebGPURenderer:
    """Own the mesh and line pipelines used by the WebGPU viewport."""

    def __init__(self, device: wgpu.GPUDevice, shader_path: Path) -> None:
        self.device = device
        self._instance_capacity = 1
        self._line_capacity = 1
        self.joint_count = 0
        self.bone_count = 0
        self.ground_vertex_count = 0
        self.trace_vertex_count = 0

        shader = device.create_shader_module(code=shader_path.read_text())
        self._camera_layout = self._create_camera_layout()
        self._instance_layout = self._create_instance_layout()
        self._camera_buffers, self._camera_bind_groups = self._create_camera_bindings()
        self._instance_buffer = self._create_instance_buffer(self._instance_capacity)
        self._instance_bind_group = self._create_instance_bind_group()
        self._line_buffer = self._create_line_buffer(self._line_capacity)
        self._mesh_pipeline = self._create_mesh_pipeline(shader)
        self._line_pipeline = self._create_line_pipeline(shader)

        sphere = PrimData.sphere(0.25, 16).astype(np.float32, copy=False)
        cylinder = PrimData.cylinder(0.15, 1.0, 12, 1).astype(np.float32, copy=False)
        self._sphere_vertex_count = sphere.size // 8
        self._cylinder_vertex_count = cylinder.size // 8
        self._sphere_buffer = device.create_buffer_with_data(
            data=sphere.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._cylinder_buffer = device.create_buffer_with_data(
            data=cylinder.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_camera_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )

    def _create_instance_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            ]
        )

    def _create_camera_bindings(
        self,
    ) -> tuple[list[wgpu.GPUBuffer], list[wgpu.GPUBindGroup]]:
        buffers = []
        bind_groups = []
        for index in range(4):
            buffer = self.device.create_buffer(
                size=_CAMERA_DTYPE.itemsize,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"bvh_camera_{index}",
            )
            buffers.append(buffer)
            bind_groups.append(
                self.device.create_bind_group(
                    layout=self._camera_layout,
                    entries=[{"binding": 0, "resource": {"buffer": buffer}}],
                )
            )
        return buffers, bind_groups

    def _create_instance_buffer(self, capacity: int) -> wgpu.GPUBuffer:
        return self.device.create_buffer(
            size=capacity * INSTANCE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="bvh_instances",
        )

    def _create_instance_bind_group(self) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(
            layout=self._instance_layout,
            entries=[{"binding": 0, "resource": {"buffer": self._instance_buffer}}],
        )

    def _create_line_buffer(self, capacity: int) -> wgpu.GPUBuffer:
        return self.device.create_buffer(
            size=capacity * LINE_VERTEX_DTYPE.itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="bvh_lines",
        )

    def _create_mesh_pipeline(
        self, shader: wgpu.GPUShaderModule
    ) -> wgpu.GPURenderPipeline:
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._camera_layout, self._instance_layout]
        )
        return self.device.create_render_pipeline(
            label="bvh_mesh_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "mesh_vertex",
                "buffers": [
                    {
                        "array_stride": 32,
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
                "module": shader,
                "entry_point": "mesh_fragment",
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

    def _create_line_pipeline(
        self, shader: wgpu.GPUShaderModule
    ) -> wgpu.GPURenderPipeline:
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._camera_layout]
        )
        return self.device.create_render_pipeline(
            label="bvh_line_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "line_vertex",
                "buffers": [
                    {
                        "array_stride": LINE_VERTEX_DTYPE.itemsize,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x4", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "line_fragment",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less_equal,
            },
            multisample={"count": 4},
        )

    def update_instances(
        self, instances: np.ndarray, joint_count: int, bone_count: int
    ) -> None:
        self.joint_count = joint_count
        self.bone_count = bone_count
        if not len(instances):
            return
        if len(instances) > self._instance_capacity:
            self._instance_buffer.destroy()
            self._instance_capacity = len(instances)
            self._instance_buffer = self._create_instance_buffer(
                self._instance_capacity
            )
            self._instance_bind_group = self._create_instance_bind_group()
        self.device.queue.write_buffer(self._instance_buffer, 0, instances.tobytes())

    def update_lines(self, ground: np.ndarray, traces: np.ndarray) -> None:
        vertices = np.concatenate((ground, traces))
        self.ground_vertex_count = len(ground)
        self.trace_vertex_count = len(traces)
        if len(vertices) > self._line_capacity:
            self._line_buffer.destroy()
            self._line_capacity = len(vertices)
            self._line_buffer = self._create_line_buffer(self._line_capacity)
        if len(vertices):
            self.device.queue.write_buffer(self._line_buffer, 0, vertices.tobytes())

    def update_camera(self, index: int, view_projection: Mat4) -> None:
        data = np.zeros((), dtype=_CAMERA_DTYPE)
        data["view_projection"] = view_projection.to_numpy()
        data["light_position"] = (20.0, 40.0, 30.0, 1.0)
        self.device.queue.write_buffer(self._camera_buffers[index], 0, data.tobytes())

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        camera_index: int,
        draw_traces: bool,
    ) -> None:
        camera_group = self._camera_bind_groups[camera_index]
        render_pass.set_pipeline(self._line_pipeline)
        render_pass.set_bind_group(0, camera_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self._line_buffer)
        render_pass.draw(self.ground_vertex_count)
        if draw_traces and self.trace_vertex_count:
            render_pass.draw(self.trace_vertex_count, 1, self.ground_vertex_count)

        render_pass.set_pipeline(self._mesh_pipeline)
        render_pass.set_bind_group(0, camera_group, [], 0, 999999)
        render_pass.set_bind_group(1, self._instance_bind_group, [], 0, 999999)
        if self.joint_count:
            render_pass.set_vertex_buffer(0, self._sphere_buffer)
            render_pass.draw(self._sphere_vertex_count, self.joint_count)
        if self.bone_count:
            render_pass.set_vertex_buffer(0, self._cylinder_buffer)
            render_pass.draw(
                self._cylinder_vertex_count,
                self.bone_count,
                0,
                self.joint_count,
            )
