#!/usr/bin/env -S uv run --script
"""HDRI image-based lighting (WebGPU). See README.md.

This bakes the same split-sum IBL textures as the OpenGL ``main.py`` -
env cube, irradiance cube, prefiltered specular cube and the BRDF LUT -
but on wgpu-py, rendering into individual cube-array layers (and mip
levels) rather than a single cubemap object. On top of that it draws the
same 7x7 PBR teapot grid (rows sweep metallic, columns sweep roughness)
lit by the split-sum IBL, then the baked HDRI as a skybox behind it.

Controls: left mouse rotates, wheel zooms, the arrow keys move the camera
(up/down fly forward/back, left/right strafe), `I` toggles IBL ambient, `E`
cycles the env/irradiance/prefilter debug cubes, Space resets the camera,
Escape quits.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
import wgpu.utils
from exr_loader import load_equirect_hdr
from ncca.ngl import (
    FirstPersonCamera,
    Mat4,
    PerspMode,
    PrimData,
    Prims,
    Vec3,
    look_at,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

_FLOAT = np.dtype(np.float32).itemsize
_VERTEX_STRIDE = 8 * _FLOAT  # pos3, normal3, uv2 (only pos is used here)

HDRI_DIR = Path(__file__).resolve().parent
HDRI_PATH = HDRI_DIR / "images" / "historic_cloister_passage_1k.exr"

ENV_SIZE, IRRADIANCE_SIZE, PREFILTER_SIZE, PREFILTER_MIPS, LUT_SIZE = (
    512,
    32,
    128,
    5,
    512,
)

BAKE_FORMAT = wgpu.TextureFormat.rgba16float
LUT_FORMAT = wgpu.TextureFormat.rg16float
PRESENT_FORMAT = wgpu.TextureFormat.rgba8unorm

# Cycled by the `E` key: which cube to draw as the skybox, and at what lod.
DEBUG_VIEWS = ("env", "irradiance", "prefilter mip 2")

# The six view matrices that look out the six cube faces from the origin
# (LearnOpenGL "Diffuse irradiance"). Order matches +X, -X, +Y, -Y, +Z, -Z,
# i.e. wgpu's cube-map array-layer order.
_CAPTURE_PROJECTION = perspective(90.0, 1.0, 0.1, 10.0, PerspMode.WebGPU)
_CAPTURE_VIEWS = [
    look_at(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
    look_at(Vec3(0, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, -1)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, -1, 0)),
]

# Matches CaptureUniforms in Equirect2Cube.wgsl / Irradiance.wgsl: two mat4s.
_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4))],
        "offsets": [0, 64],
        "itemsize": 128,
    }
)
# Prefilter.wgsl adds a roughness float (+ padding to a vec4).
_PREFILTER_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "roughness"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), np.float32],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)
# Matches SkyboxUniforms in Skybox.wgsl: mvp + lod (padded to a vec4).
_SKYBOX_DTYPE = np.dtype(
    {
        "names": ["mvp", "lod"],
        "formats": [(np.float32, (4, 4)), np.float32],
        "offsets": [0, 64],
        "itemsize": 80,
    }
)

# Dynamic uniform offsets must be a multiple of the device alignment; 256 is
# the maximum any device requires, so it is always a safe stride to use.
_DYNAMIC_STRIDE = 256

# One padded slot of per-teapot transforms (matches Transforms in PBR.wgsl).
_TRANSFORM_DTYPE = np.dtype(
    {
        "names": ["MVP", "M", "normalMatrix", "material"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
        ],
        "offsets": [0, 64, 128, 192],
        "itemsize": _DYNAMIC_STRIDE,
    }
)

# Lights, camera and the IBL toggle, shared by every teapot in a frame
# (matches Scene in PBR.wgsl).
_PBR_SCENE_DTYPE = np.dtype(
    {
        "names": ["lightPositions", "lightColors", "camPos", "useIBL"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
            np.uint32,
        ],
        "offsets": [0, 64, 128, 144],
        "itemsize": 160,
    }
)

# Matches main.py's initializeGL: 4 analytic point lights around the grid.
_LIGHT_POSITIONS = [
    (-10.0, 4.0, -10.0),
    (10.0, 4.0, -10.0),
    (-10.0, 4.0, 10.0),
    (10.0, 4.0, 10.0),
]
_LIGHT_COLOUR = (300.0, 300.0, 300.0)

GRID_SIZE = 7

# Arrow keys drive the camera; paintWebGPU keeps repainting while any of these
# is held so movement is smooth rather than one step per key event.
_MOVE_KEYS: set = {Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right}

# Multiplier on the camera's own speed so a held arrow key crosses the scene in
# a second or two rather than crawling.
_MOVE_SPEED = 6.0


class HDRIScene(WebGPUWidget):
    """WebGPU widget that bakes the split-sum IBL textures and draws the
    baked HDRI as a skybox."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WebGPU HDRI Image-Based Lighting")
        # Grab keyboard focus so the arrow keys reach keyPressEvent rather than
        # being swallowed by Qt's focus navigation.
        self.setFocusPolicy(Qt.StrongFocus)
        self.msaa_sample_count = 4

        self.debug_view = 0
        self.use_ibl = True
        self.rotate = False
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.keys_pressed: set = set()

        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 0.0

        self.camera = FirstPersonCamera(
            Vec3(0, 0, 30), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
        )
        self._update_projection(max(self.width(), 1), max(self.height(), 1))

        self._initialize_web_gpu()
        self.startTimer(16)
        self.update()

    # ------------------------------------------------------------------ setup
    def _initialize_web_gpu(self) -> None:
        try:
            self.device = wgpu.utils.get_default_device()
            self._create_render_buffer()
            self._load_geometry()
            self._create_sampler()
            self._create_skybox_pipeline()

            img = load_equirect_hdr(HDRI_PATH)  # (H, W, 3) f32
            rgba = np.dstack([img, np.ones(img.shape[:2], dtype=np.float32)]).astype(
                np.float32
            )
            self.equirect_tex = self._upload_2d(rgba, BAKE_FORMAT)

            self.env_cube = self._bake_cube(
                "Equirect2Cube.wgsl", ENV_SIZE, src_view_dim="2d", src=self.equirect_tex
            )
            self.irradiance_cube = self._bake_cube(
                "Irradiance.wgsl",
                IRRADIANCE_SIZE,
                src_view_dim="cube",
                src=self.env_cube,
            )
            self.prefilter_cube = self._bake_prefilter(self.env_cube)
            self.brdf_lut = self._bake_brdf()

            self._create_pbr_pipeline()
            self._build_grid()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            traceback.print_exc()
            raise

    def _load_geometry(self) -> None:
        """Upload the unit cube (bake draws + skybox) and the teapot (grid)."""
        cube = PrimData.primitive(Prims.CUBE.value).astype(np.float32)
        self.cube_geometry = self._make_vertex_buffer(cube)
        teapot = PrimData.primitive(Prims.TEAPOT.value).astype(np.float32)
        self.teapot_geometry = self._make_vertex_buffer(teapot)

    def _make_vertex_buffer(self, data: np.ndarray) -> dict:
        buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        return {"buffer": buffer, "count": data.size // 8}

    def _create_sampler(self) -> None:
        self.linear_sampler = self.device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
        )

    # -------------------------------------------------------------- bake IO
    def _upload_2d(self, data: np.ndarray, fmt: str) -> "wgpu.GPUTexture":
        # rgba16float storage is half-precision; the source array is
        # float32 (from load_equirect_hdr), so it must be down-cast before
        # the raw bytes go to the GPU or the bit pattern is read back wrong.
        height, width = data.shape[:2]
        half_data = data.astype(np.float16)
        tex = self.device.create_texture(
            size=(width, height, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": tex},
            half_data.tobytes(),
            {"bytes_per_row": width * 4 * 2, "rows_per_image": height},
            (width, height, 1),
        )
        return tex

    def _make_cube_pipeline(
        self, shader_name: str, src_view_dim: str, capture_dtype: np.dtype
    ) -> dict:
        with open(HDRI_DIR / shader_name, "r") as f:
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
                {"binding": 1, "resource": self.linear_sampler},
            ],
        )

    def _bake_cube(
        self, shader_name: str, size: int, src_view_dim: str, src
    ) -> "wgpu.GPUTexture":
        """Bake `shader_name` into all six faces of a new `size`^2 cube texture."""
        pipe = self._make_cube_pipeline(shader_name, src_view_dim, _CAPTURE_DTYPE)
        source_bind_group = self._source_bind_group(
            pipe["source_bgl"], src, src_view_dim
        )

        cube = self.device.create_texture(
            size=(size, size, 6),
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )

        uniforms = np.zeros((), dtype=_CAPTURE_DTYPE)
        uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
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

    def _bake_prefilter(self, src) -> "wgpu.GPUTexture":
        pipe = self._make_cube_pipeline(
            "Prefilter.wgsl", "cube", _PREFILTER_CAPTURE_DTYPE
        )
        source_bind_group = self._source_bind_group(pipe["source_bgl"], src, "cube")

        cube = self.device.create_texture(
            size=(PREFILTER_SIZE, PREFILTER_SIZE, 6),
            mip_level_count=PREFILTER_MIPS,
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )

        for mip in range(PREFILTER_MIPS):
            size = PREFILTER_SIZE >> mip
            roughness = mip / (PREFILTER_MIPS - 1)
            uniforms = np.zeros((), dtype=_PREFILTER_CAPTURE_DTYPE)
            uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
            uniforms["roughness"] = roughness
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
        render_pass.set_vertex_buffer(0, self.cube_geometry["buffer"])
        render_pass.draw(self.cube_geometry["count"])
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def _bake_brdf(self) -> "wgpu.GPUTexture":
        with open(HDRI_DIR / "BRDF.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[]),
            vertex={"module": shader, "entry_point": "vertex_main"},
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": LUT_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        lut = self.device.create_texture(
            size=(LUT_SIZE, LUT_SIZE, 1),
            format=LUT_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
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
        render_pass.set_viewport(0, 0, LUT_SIZE, LUT_SIZE, 0, 1)
        render_pass.set_pipeline(pipeline)
        render_pass.draw(3)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        return lut

    # --------------------------------------------------------- PBR teapots
    def _create_pbr_pipeline(self) -> None:
        """Create the split-sum IBL PBR pipeline and its bind group layouts."""
        with open(HDRI_DIR / "PBR.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        # @group(0) per-teapot transforms, addressed with a dynamic offset.
        self.pbr_transform_bgl = self.device.create_bind_group_layout(
            label="pbr_transform_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "has_dynamic_offset": True,
                    },
                }
            ],
        )
        # @group(1) lights, camera and the IBL toggle, shared across the frame.
        self.pbr_scene_bgl = self.device.create_bind_group_layout(
            label="pbr_scene_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        # @group(2) the baked IBL textures + a linear/mip-linear sampler.
        self.pbr_ibl_bgl = self.device.create_bind_group_layout(
            label="pbr_ibl_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ],
        )

        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[
                self.pbr_transform_bgl,
                self.pbr_scene_bgl,
                self.pbr_ibl_bgl,
            ]
        )
        self.pbr_pipeline = self.device.create_render_pipeline(
            label="pbr_ibl_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": _VERTEX_STRIDE,
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
                "entry_point": "fragment_main",
                "targets": [{"format": PRESENT_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        self.pbr_scene_uniforms = np.zeros((), dtype=_PBR_SCENE_DTYPE)
        for i, pos in enumerate(_LIGHT_POSITIONS):
            self.pbr_scene_uniforms["lightPositions"][i] = (*pos, 1.0)
            self.pbr_scene_uniforms["lightColors"][i] = (*_LIGHT_COLOUR, 1.0)
        self.pbr_scene_buffer = self.device.create_buffer(
            size=self.pbr_scene_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="pbr_scene_buffer",
        )
        self.pbr_scene_bind_group = self.device.create_bind_group(
            layout=self.pbr_scene_bgl,
            entries=[{"binding": 0, "resource": {"buffer": self.pbr_scene_buffer}}],
        )

        self.pbr_ibl_bind_group = self.device.create_bind_group(
            layout=self.pbr_ibl_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": self.irradiance_cube.create_view(
                        dimension=wgpu.TextureViewDimension.cube
                    ),
                },
                {
                    "binding": 1,
                    "resource": self.prefilter_cube.create_view(
                        dimension=wgpu.TextureViewDimension.cube
                    ),
                },
                {
                    "binding": 2,
                    "resource": self.brdf_lut.create_view(
                        dimension=wgpu.TextureViewDimension.d2
                    ),
                },
                {"binding": 3, "resource": self.linear_sampler},
            ],
        )

    def _build_grid(self) -> None:
        """Lay out the 7x7 teapot grid: rows sweep metallic, columns sweep
        roughness, matching main.py's `_draw_teapot_grid`."""
        self.grid_objects = []  # (model Mat4, metallic, roughness)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                model = Mat4()
                model[3, 0] = (col - 3) * 3.0
                model[3, 1] = (row - 3) * 3.0
                model[3, 2] = 0.0
                metallic = row / 6.0
                roughness = max(0.05, col / 6.0)
                self.grid_objects.append((model, metallic, roughness))

        count = len(self.grid_objects)
        self.grid_transform_uniforms = np.zeros(count, dtype=_TRANSFORM_DTYPE)
        self.grid_transform_buffer = self.device.create_buffer(
            size=self.grid_transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="grid_transform_buffer",
        )
        self.grid_transform_bind_group = self.device.create_bind_group(
            layout=self.pbr_transform_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.grid_transform_buffer,
                        "offset": 0,
                        "size": _DYNAMIC_STRIDE,
                    },
                }
            ],
        )

    def _write_pbr_scene_uniforms(self) -> None:
        self.pbr_scene_uniforms["camPos"] = (
            self.camera.eye.x,
            self.camera.eye.y,
            self.camera.eye.z,
            1.0,
        )
        self.pbr_scene_uniforms["useIBL"] = 1 if self.use_ibl else 0
        self.device.queue.write_buffer(
            self.pbr_scene_buffer, 0, self.pbr_scene_uniforms.tobytes()
        )

    def _write_grid_transform_uniforms(self) -> None:
        view = self.camera.view
        projection = self.camera.projection
        for i, (model, metallic, roughness) in enumerate(self.grid_objects):
            model_view = view @ model
            mvp = projection @ model_view
            normal_matrix = model_view.copy().inverse().transposed()
            slot = self.grid_transform_uniforms[i]
            slot["MVP"] = mvp.to_numpy()
            slot["M"] = model.to_numpy()
            slot["normalMatrix"] = normal_matrix.to_numpy()
            slot["material"] = (metallic, roughness, 1.0, 0.0)
        self.device.queue.write_buffer(
            self.grid_transform_buffer, 0, self.grid_transform_uniforms.tobytes()
        )

    def _draw_grid(self, render_pass: "wgpu.GPURenderPassEncoder") -> None:
        render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
        render_pass.set_pipeline(self.pbr_pipeline)
        render_pass.set_bind_group(1, self.pbr_scene_bind_group)
        render_pass.set_bind_group(2, self.pbr_ibl_bind_group)
        for i in range(len(self.grid_objects)):
            render_pass.set_bind_group(
                0, self.grid_transform_bind_group, [i * _DYNAMIC_STRIDE]
            )
            render_pass.set_vertex_buffer(0, self.teapot_geometry["buffer"])
            render_pass.draw(self.teapot_geometry["count"])

    # ------------------------------------------------------------- skybox
    def _create_skybox_pipeline(self) -> None:
        with open(HDRI_DIR / "Skybox.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        self.skybox_uniform_bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        self.skybox_source_bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
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
            bind_group_layouts=[self.skybox_uniform_bgl, self.skybox_source_bgl]
        )
        self.skybox_pipeline = self.device.create_render_pipeline(
            label="skybox_pipeline",
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
                "targets": [{"format": PRESENT_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.less_equal,
            },
            multisample={"count": self.msaa_sample_count},
        )

        self.skybox_uniforms = np.zeros((), dtype=_SKYBOX_DTYPE)
        self.skybox_uniform_buffer = self.device.create_buffer(
            size=self.skybox_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.skybox_uniform_bind_group = self.device.create_bind_group(
            layout=self.skybox_uniform_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.skybox_uniform_buffer}}
            ],
        )

    def _skybox_source_bind_group(self):
        name = DEBUG_VIEWS[self.debug_view]
        if name == "irradiance":
            cube, lod = self.irradiance_cube, 0.0
        elif name == "prefilter mip 2":
            cube, lod = self.prefilter_cube, 2.0
        else:
            cube, lod = self.env_cube, 0.0
        view = cube.create_view(dimension=wgpu.TextureViewDimension.cube)
        bind_group = self.device.create_bind_group(
            layout=self.skybox_source_bgl,
            entries=[
                {"binding": 0, "resource": view},
                {"binding": 1, "resource": self.linear_sampler},
            ],
        )
        return bind_group, lod

    # ------------------------------------------------------------------- paint
    def paintWebGPU(self) -> None:
        if not hasattr(self, "skybox_pipeline") or not hasattr(self, "pbr_pipeline"):
            return
        current = self.timer.elapsed() * 0.001
        delta_time = min(current - self.last_frame, 0.05)
        self.last_frame = current
        self._update_camera_movement(delta_time)

        # Strip translation from the view so the skybox never moves relative
        # to the camera, whichever way we pan/orbit.
        view_rotation_only = self.camera.view.copy()
        view_rotation_only[3, 0] = 0.0
        view_rotation_only[3, 1] = 0.0
        view_rotation_only[3, 2] = 0.0
        mvp = self.camera.projection @ view_rotation_only

        source_bind_group, lod = self._skybox_source_bind_group()
        self.skybox_uniforms["mvp"] = mvp.to_numpy()
        self.skybox_uniforms["lod"] = lod
        self.device.queue.write_buffer(
            self.skybox_uniform_buffer, 0, self.skybox_uniforms.tobytes()
        )

        self._write_pbr_scene_uniforms()
        self._write_grid_transform_uniforms()

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.05, 0.05, 0.07, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        # Draw the teapot grid first (writes depth), then the skybox behind
        # it (depth test only, no depth write) sharing the same depth buffer.
        self._draw_grid(render_pass)
        render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
        render_pass.set_pipeline(self.skybox_pipeline)
        render_pass.set_bind_group(0, self.skybox_uniform_bind_group)
        render_pass.set_bind_group(1, source_bind_group)
        render_pass.set_vertex_buffer(0, self.cube_geometry["buffer"])
        render_pass.draw(self.cube_geometry["count"])
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        if self.keys_pressed & _MOVE_KEYS:
            self.update()

    # ------------------------------------------------------------------- input
    def _update_projection(self, w: int, h: int) -> None:
        """Rebuild the camera's projection for a `w`x`h` viewport.

        FirstPersonCamera.set_projection only returns a matrix (it never stores
        it, and rebuilds from self.aspect/near/far on zoom), so we set those
        fields and rebuild _projection here -- that keeps the aspect matching
        the window (so the grid isn't stretched) and wheel-zoom working.
        """
        self.camera.aspect = float(w) / float(max(h, 1))
        self.camera.near = 0.05
        self.camera.far = 350.0
        self.camera._projection = perspective(
            self.camera.zoom,
            self.camera.aspect,
            self.camera.near,
            self.camera.far,
            PerspMode.WebGPU,
        )

    def resizeWebGPU(self, width: int, height: int) -> None:
        ratio = self.devicePixelRatio()
        w = max(width * ratio, 1)
        h = max(height * ratio, 1)
        self._update_projection(w, h)
        self.update()

    def _update_camera_movement(self, delta_time: float) -> None:
        # Arrow keys: up/down move along the view direction (forward/back),
        # left/right strafe. camera.move(x, y, delta) shifts the eye by
        # front*x and right*y, so this reads directly off the held keys.
        forward = float(Qt.Key_Up in self.keys_pressed) - float(
            Qt.Key_Down in self.keys_pressed
        )
        strafe = float(Qt.Key_Right in self.keys_pressed) - float(
            Qt.Key_Left in self.keys_pressed
        )
        if forward or strafe:
            self.camera.move(forward * _MOVE_SPEED, strafe * _MOVE_SPEED, delta_time)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        self.keys_pressed.add(key)
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_E:
            self.debug_view = (self.debug_view + 1) % len(DEBUG_VIEWS)
        elif key == Qt.Key_I:
            self.use_ibl = not self.use_ibl
        elif key == Qt.Key_Space:
            self.camera = FirstPersonCamera(
                Vec3(0, 0, 30), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
            )
            self._update_projection(max(self.width(), 1), max(self.height(), 1))
        self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self.keys_pressed.discard(event.key())
        self.update()
        super().keyReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.camera.process_mouse_movement(diff_x, -diff_y)
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            position = event.position()
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        num_pixels = event.angleDelta().y()
        self.camera.process_mouse_scroll(num_pixels * 0.01)
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    win = HDRIScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
