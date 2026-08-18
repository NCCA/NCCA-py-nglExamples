"""The WebGPU pipeline used to draw a skinned mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu
from mesh import SkinnedMesh
from ncca.ngl import Image, Mat4, Vec3, logger

SHADER_PATH = Path(__file__).with_name("skin_webgpu.wgsl")

_CAMERA_DTYPE = np.dtype(
    [
        ("view_projection", np.float32, (4, 4)),
        ("model", np.float32, (4, 4)),
        ("eye_position", np.float32, (4,)),
        ("light_position", np.float32, (4,)),
    ]
)

_VERTEX_BUFFER_LAYOUTS = [
    {
        "array_stride": 12,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
    },
    {
        "array_stride": 12,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 1}],
    },
    {
        "array_stride": 8,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 2}],
    },
    {
        "array_stride": 16,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x4", "offset": 0, "shader_location": 3}],
    },
    {
        "array_stride": 16,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x4", "offset": 0, "shader_location": 4}],
    },
]


class SkinWebGPURenderer:
    """Own the wgpu pipeline, buffers and textures used to draw one SkinnedMesh."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.index_count = 0
        self.submeshes: list = []
        self._bone_capacity = 1
        self._texture_cache: dict[str | None, wgpu.GPUBindGroup] = {}
        self._fallback_texture: wgpu.GPUBindGroup | None = None

        shader = device.create_shader_module(code=SHADER_PATH.read_text())
        self._camera_layout = self._create_camera_layout()
        self._bone_layout = self._create_bone_layout()
        self._texture_layout = self._create_texture_layout()
        self._camera_buffers, self._camera_bind_groups = self._create_camera_bindings()
        self._bone_buffer = self._create_bone_buffer(self._bone_capacity)
        self._bone_bind_group = self._create_bone_bind_group()
        self._sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear
        )
        self._pipeline = self._create_pipeline(shader)

    # ------------------------------------------------------------ layouts

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

    def _create_bone_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            ]
        )

    def _create_texture_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
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
                label=f"skin_camera_{index}",
            )
            buffers.append(buffer)
            bind_groups.append(
                self.device.create_bind_group(
                    layout=self._camera_layout,
                    entries=[{"binding": 0, "resource": {"buffer": buffer}}],
                )
            )
        return buffers, bind_groups

    def _create_bone_buffer(self, capacity: int) -> wgpu.GPUBuffer:
        return self.device.create_buffer(
            size=capacity * 64,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="skin_bones",
        )

    def _create_bone_bind_group(self) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(
            layout=self._bone_layout,
            entries=[{"binding": 0, "resource": {"buffer": self._bone_buffer}}],
        )

    def _create_pipeline(self, shader: wgpu.GPUShaderModule) -> wgpu.GPURenderPipeline:
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[
                self._camera_layout,
                self._bone_layout,
                self._texture_layout,
            ]
        )
        return self.device.create_render_pipeline(
            label="skin_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": _VERTEX_BUFFER_LAYOUTS,
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

    # --------------------------------------------------------------- mesh

    def set_mesh(self, mesh: SkinnedMesh) -> None:
        """Upload a new mesh's geometry, grow the bone buffer if needed, load textures."""
        self.submeshes = mesh.submeshes
        self.index_count = len(mesh.indices)

        self._position_buffer = self._vertex_buffer(mesh.positions)
        self._normal_buffer = self._vertex_buffer(mesh.normals)
        self._uv_buffer = self._vertex_buffer(mesh.texcoords)
        self._bone_id_buffer = self._vertex_buffer(mesh.bone_ids)
        self._bone_weight_buffer = self._vertex_buffer(mesh.bone_weights)
        self._index_buffer = self.device.create_buffer_with_data(
            data=mesh.indices.astype(np.uint32, copy=False),
            usage=wgpu.BufferUsage.INDEX,
        )

        bone_count = max(len(mesh.bone_names), 1)
        if bone_count > self._bone_capacity:
            self._bone_buffer.destroy()
            self._bone_capacity = bone_count
            self._bone_buffer = self._create_bone_buffer(self._bone_capacity)
            self._bone_bind_group = self._create_bone_bind_group()

        self._texture_cache = {}
        for submesh in mesh.submeshes:
            self._texture_bind_group(submesh.texture_path)

    def _vertex_buffer(self, data: np.ndarray) -> wgpu.GPUBuffer:
        return self.device.create_buffer_with_data(
            data=data.astype(np.float32, copy=False),
            usage=wgpu.BufferUsage.VERTEX,
        )

    def update_bones(self, transforms: list[Mat4]) -> None:
        """Rewrite the bone storage buffer for the current animation frame."""
        if not transforms:
            return
        matrices = np.stack([t.to_numpy() for t in transforms], axis=0).astype(
            np.float32, copy=False
        )
        self.device.queue.write_buffer(self._bone_buffer, 0, matrices.tobytes())

    def update_camera(
        self,
        index: int,
        view_projection: Mat4,
        model: Mat4,
        eye_position: Vec3,
        light_position: Vec3,
    ) -> None:
        """Write one pane's (index 0-3) camera/model/light uniform."""
        data = np.zeros((), dtype=_CAMERA_DTYPE)
        data["view_projection"] = view_projection.to_numpy()
        data["model"] = model.to_numpy()
        data["eye_position"] = (eye_position.x, eye_position.y, eye_position.z, 1.0)
        data["light_position"] = (
            light_position.x,
            light_position.y,
            light_position.z,
            1.0,
        )
        self.device.queue.write_buffer(self._camera_buffers[index], 0, data.tobytes())

    # ----------------------------------------------------------- textures

    def _fallback_bind_group(self) -> wgpu.GPUBindGroup:
        if self._fallback_texture is None:
            self._fallback_texture = self._create_texture_bind_group(
                np.full((1, 1, 4), 255, dtype=np.uint8)
            )
        return self._fallback_texture

    def _texture_bind_group(self, texture_path: str | None) -> wgpu.GPUBindGroup:
        if texture_path in self._texture_cache:
            return self._texture_cache[texture_path]
        bind_group = self._load_texture_bind_group(texture_path)
        self._texture_cache[texture_path] = bind_group
        return bind_group

    def _load_texture_bind_group(self, texture_path: str | None) -> wgpu.GPUBindGroup:
        if texture_path is None:
            return self._fallback_bind_group()
        try:
            pixels = Image(texture_path).get_pixels()
            if pixels.shape[2] == 3:
                rgba = np.empty((*pixels.shape[:2], 4), dtype=np.uint8)
                rgba[:, :, :3] = pixels
                rgba[:, :, 3] = 255
            else:
                rgba = pixels
        except Exception as error:
            # impasse.errors.AssimpError and a missing/corrupt image file both
            # land here -- keep the mesh visible (flat white) rather than
            # losing it, same fallback the OpenGL path uses.
            logger.warning(
                f"Could not load texture {texture_path!r} ({error}); "
                "using a flat fallback"
            )
            return self._fallback_bind_group()
        return self._create_texture_bind_group(rgba)

    def _create_texture_bind_group(self, rgba: np.ndarray) -> wgpu.GPUBindGroup:
        height, width = rgba.shape[:2]
        texture = self.device.create_texture(
            size=(width, height, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"bytes_per_row": width * 4, "rows_per_image": height},
            (width, height, 1),
        )
        return self.device.create_bind_group(
            layout=self._texture_layout,
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": self._sampler},
            ],
        )

    # -------------------------------------------------------------- draw

    def render(self, render_pass: wgpu.GPURenderPassEncoder, camera_index: int) -> None:
        if self.index_count == 0:
            return
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(
            0, self._camera_bind_groups[camera_index], [], 0, 999999
        )
        render_pass.set_bind_group(1, self._bone_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self._position_buffer)
        render_pass.set_vertex_buffer(1, self._normal_buffer)
        render_pass.set_vertex_buffer(2, self._uv_buffer)
        render_pass.set_vertex_buffer(3, self._bone_id_buffer)
        render_pass.set_vertex_buffer(4, self._bone_weight_buffer)
        render_pass.set_index_buffer(self._index_buffer, wgpu.IndexFormat.uint32)
        for submesh in self.submeshes:
            bind_group = self._texture_bind_group(submesh.texture_path)
            render_pass.set_bind_group(2, bind_group, [], 0, 999999)
            render_pass.draw_indexed(submesh.index_count, 1, submesh.index_offset)
