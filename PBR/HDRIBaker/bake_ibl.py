"""Bake the split-sum IBL maps offscreen and read them back to numpy.

The GPU work is the same as ``PBR/HDRI/HDRIWebGPU.py`` — reproject the
equirect panorama to a cube, convolve it to an irradiance cube, GGX-prefilter
a roughness mip chain, and integrate the BRDF lookup table — but here it runs
on a headless device with no window, and every result is copied back off the
GPU into a numpy array so it can be saved to a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu
import wgpu.utils
from bake_settings import BakeSettings, prefilter_key
from ncca.ngl import PerspMode, PrimData, Prims, Vec3, look_at, perspective

_FLOAT = np.dtype(np.float32).itemsize
_VERTEX_STRIDE = 8 * _FLOAT
BAKE_FORMAT = wgpu.TextureFormat.rgba16float
LUT_FORMAT = wgpu.TextureFormat.rg16float
_SHADER_DIR = Path(__file__).resolve().parent / "shaders"

_CAPTURE_PROJECTION = perspective(90.0, 1.0, 0.1, 10.0, PerspMode.WebGPU)
_CAPTURE_VIEWS = [
    look_at(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
    look_at(Vec3(0, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, -1)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, -1, 0)),
]
_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4))],
        "offsets": [0, 64],
        "itemsize": 128,
    }
)
_IRRADIANCE_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "sample_delta"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), np.float32],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)
_PREFILTER_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "roughness", "sample_count", "env_resolution"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            np.float32,
            np.uint32,
            np.float32,
        ],
        "offsets": [0, 64, 128, 132, 136],
        "itemsize": 144,
    }
)
_BRDF_UNIFORM_DTYPE = np.dtype(
    {
        "names": ["sample_count"],
        "formats": [np.uint32],
        "offsets": [0],
        "itemsize": 16,
    }
)


def bake_maps(
    image: np.ndarray,
    settings: BakeSettings | None = None,
    source: str = "",
) -> dict:
    """Bake every IBL map from an ``(H, W, 3)`` float32 equirect image."""
    settings = settings or BakeSettings()
    settings.validate()

    device = wgpu.utils.get_default_device()
    baker = _Baker(device, settings)
    rgba = np.dstack([image, np.ones(image.shape[:2], np.float32)]).astype(np.float32)
    equirect = baker.upload_2d(rgba, BAKE_FORMAT)

    env = baker.bake_cube("Equirect2Cube.wgsl", settings.env_size, "2d", equirect)
    irradiance = baker.bake_cube(
        "Irradiance.wgsl",
        settings.irradiance_size,
        "cube",
        env,
        capture_dtype=_IRRADIANCE_CAPTURE_DTYPE,
        extra={"sample_delta": settings.irradiance_sample_delta},
    )
    prefilter = baker.bake_prefilter(env)
    lut = baker.bake_brdf()

    out = {
        "env": baker.read_cube(env, settings.env_size, 0),
        "irradiance": baker.read_cube(irradiance, settings.irradiance_size, 0),
        "brdf_lut": baker.read_2d(lut, settings.lut_size, settings.lut_size, 2, 0),
    }
    for mip in range(settings.prefilter_mips):
        size = settings.prefilter_size >> mip
        out[prefilter_key(mip)] = baker.read_cube(prefilter, size, mip)
    out["meta"] = {
        "source": source,
        "settings": settings.to_meta(),
        "prefilter_mips": settings.prefilter_mips,
        "prefilter_roughness": settings.roughness_levels(),
        "format": "rgba16float / rg16float",
    }
    return out


class _Baker:
    def __init__(self, device: "wgpu.GPUDevice", settings: BakeSettings) -> None:
        self.device = device
        self.settings = settings
        cube = PrimData.primitive(Prims.CUBE.value).astype(np.float32)
        self.cube_buffer = device.create_buffer_with_data(
            data=cube, usage=wgpu.BufferUsage.VERTEX
        )
        self.cube_count = cube.size // 8
        self.sampler = device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
        )

    # ---- IO / upload (ported from HDRIWebGPU._upload_2d) --------------------
    def upload_2d(self, data: np.ndarray, fmt: str) -> "wgpu.GPUTexture":
        height, width = data.shape[:2]
        half = data.astype(np.float16)
        tex = self.device.create_texture(
            size=(width, height, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": tex},
            half.tobytes(),
            {"bytes_per_row": width * 4 * 2, "rows_per_image": height},
            (width, height, 1),
        )
        return tex

    # ---- pipelines / bake (ported from HDRIWebGPU) -------------------------
    def _make_cube_pipeline(
        self, shader_name: str, src_view_dim: str, capture_dtype: np.dtype
    ) -> dict:
        with open(_SHADER_DIR / shader_name, "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        capture_bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        tex_view_dim = (
            wgpu.TextureViewDimension.cube
            if src_view_dim == "cube"
            else wgpu.TextureViewDimension.d2
        )
        source_bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": tex_view_dim,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ],
        )
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[capture_bgl, source_bgl]
        )
        pipeline = self.device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": _VERTEX_STRIDE,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": BAKE_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        capture_buffer = self.device.create_buffer(
            size=capture_dtype.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        capture_bind_group = self.device.create_bind_group(
            layout=capture_bgl,
            entries=[{"binding": 0, "resource": {"buffer": capture_buffer}}],
        )
        return {
            "pipeline": pipeline,
            "capture_bgl": capture_bgl,
            "source_bgl": source_bgl,
            "capture_buffer": capture_buffer,
            "capture_bind_group": capture_bind_group,
        }

    def _source_bind_group(self, source_bgl, src, view_dim: str):
        view = src.create_view(
            dimension=(
                wgpu.TextureViewDimension.cube
                if view_dim == "cube"
                else wgpu.TextureViewDimension.d2
            )
        )
        return self.device.create_bind_group(
            layout=source_bgl,
            entries=[
                {"binding": 0, "resource": view},
                {"binding": 1, "resource": self.sampler},
            ],
        )

    def bake_cube(
        self,
        shader_name: str,
        size: int,
        src_view_dim: str,
        src,
        capture_dtype: np.dtype = _CAPTURE_DTYPE,
        extra: dict | None = None,
    ) -> "wgpu.GPUTexture":
        """Bake `shader_name` into all six faces of a new `size`^2 cube texture.

        `extra` sets any shader-specific uniform fields beyond projection/view.
        """
        pipe = self._make_cube_pipeline(shader_name, src_view_dim, capture_dtype)
        source_bind_group = self._source_bind_group(
            pipe["source_bgl"], src, src_view_dim
        )

        cube = self.device.create_texture(
            size=(size, size, 6),
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
        )

        uniforms = np.zeros((), dtype=capture_dtype)
        uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
        for name, value in (extra or {}).items():
            uniforms[name] = value
        for face in range(6):
            uniforms["view"] = _CAPTURE_VIEWS[face].to_numpy()
            self.device.queue.write_buffer(
                pipe["capture_buffer"], 0, uniforms.tobytes()
            )
            self._render_face(
                pipe["pipeline"],
                pipe["capture_bind_group"],
                source_bind_group,
                cube.create_view(
                    dimension="2d", base_array_layer=face, array_layer_count=1
                ),
                size,
            )
        return cube

    def bake_prefilter(self, src) -> "wgpu.GPUTexture":
        pipe = self._make_cube_pipeline(
            "Prefilter.wgsl", "cube", _PREFILTER_CAPTURE_DTYPE
        )
        source_bind_group = self._source_bind_group(pipe["source_bgl"], src, "cube")

        size0 = self.settings.prefilter_size
        cube = self.device.create_texture(
            size=(size0, size0, 6),
            mip_level_count=self.settings.prefilter_mips,
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
        )

        for mip in range(self.settings.prefilter_mips):
            size = size0 >> mip
            uniforms = np.zeros((), dtype=_PREFILTER_CAPTURE_DTYPE)
            uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
            uniforms["roughness"] = self.settings.roughness_for_mip(mip)
            uniforms["sample_count"] = self.settings.prefilter_samples
            uniforms["env_resolution"] = float(self.settings.env_size)
            for face in range(6):
                uniforms["view"] = _CAPTURE_VIEWS[face].to_numpy()
                self.device.queue.write_buffer(
                    pipe["capture_buffer"], 0, uniforms.tobytes()
                )
                self._render_face(
                    pipe["pipeline"],
                    pipe["capture_bind_group"],
                    source_bind_group,
                    cube.create_view(
                        dimension="2d",
                        base_array_layer=face,
                        array_layer_count=1,
                        base_mip_level=mip,
                        mip_level_count=1,
                    ),
                    size,
                )
        return cube

    def _render_face(
        self, pipeline, capture_bind_group, source_bind_group, target_view, size: int
    ) -> None:
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ],
        )
        render_pass.set_viewport(0, 0, size, size, 0, 1)
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, capture_bind_group)
        render_pass.set_bind_group(1, source_bind_group)
        render_pass.set_vertex_buffer(0, self.cube_buffer)
        render_pass.draw(self.cube_count)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def bake_brdf(self) -> "wgpu.GPUTexture":
        with open(_SHADER_DIR / "BRDF.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        uniform_buffer = self.device.create_buffer(
            size=_BRDF_UNIFORM_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        uniforms = np.zeros((), dtype=_BRDF_UNIFORM_DTYPE)
        uniforms["sample_count"] = self.settings.brdf_samples
        self.device.queue.write_buffer(uniform_buffer, 0, uniforms.tobytes())
        bind_group = self.device.create_bind_group(
            layout=bgl,
            entries=[{"binding": 0, "resource": {"buffer": uniform_buffer}}],
        )

        pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[bgl]),
            vertex={"module": shader, "entry_point": "vertex_main"},
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": LUT_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        lut_size = self.settings.lut_size
        lut = self.device.create_texture(
            size=(lut_size, lut_size, 1),
            format=LUT_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
        )
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": lut.create_view(),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ],
        )
        render_pass.set_viewport(0, 0, lut_size, lut_size, 0, 1)
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        return lut

    # ---- readback (NEW) ----------------------------------------------------
    def read_2d(
        self, texture, width: int, height: int, channels: int, mip: int, layer: int = 0
    ) -> np.ndarray:
        """Copy one mip/layer of a float16 texture back to an (H,W,C) array."""
        bytes_per_pixel = channels * 2  # float16
        # copy_texture_to_buffer requires bytes_per_row to be a multiple of 256
        row_bytes = width * bytes_per_pixel
        padded = (row_bytes + 255) & ~255
        buffer = self.device.create_buffer(
            size=padded * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": mip,
                "origin": (0, 0, layer),
            },
            {"buffer": buffer, "bytes_per_row": padded, "rows_per_image": height},
            (width, height, 1),
        )
        self.device.queue.submit([encoder.finish()])

        buffer.map_sync(wgpu.MapMode.READ)
        raw = buffer.read_mapped()
        buffer.unmap()
        flat = np.frombuffer(bytes(raw), dtype=np.float16)
        rows = flat.reshape(height, padded // 2)
        return np.ascontiguousarray(
            rows[:, : width * channels].reshape(height, width, channels)
        )

    def read_cube(self, texture, size: int, mip: int) -> np.ndarray:
        """Read all six faces of a cube mip into a (6, size, size, 4) array."""
        faces = [self.read_2d(texture, size, size, 4, mip, layer) for layer in range(6)]
        return np.stack(faces, axis=0)
