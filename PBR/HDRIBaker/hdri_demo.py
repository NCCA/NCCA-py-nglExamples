#!/usr/bin/env -S uv run --script
"""HDRI image-based lighting demo, lit from pre-baked maps (WebGPU).

Where ``PBR/HDRI/HDRIWebGPU.py`` bakes the split-sum IBL textures on
startup, this demo loads them ready-made from an ``.npz`` written by
``bake_ibl.py`` (see ``ibl_maps.py``) - env cube, irradiance cube,
prefiltered specular cube and BRDF LUT - and uploads them straight to
the GPU.

Rather than the sibling grid, this shows a single teapot whose PBR
material you drive live from a floating QML control panel: metallic,
roughness and ambient-occlusion sliders, an albedo colour picker, an
IBL on/off toggle and an env/irradiance/prefilter cube selector. The
overlay reuses ``GUIDemos/QMLWebGPUOverlay``'s pattern - a transparent
``QQuickWidget`` on top of the offscreen WebGPU surface, with clicks
outside any panel forwarded through to the camera.

Controls: left mouse rotates, wheel zooms; the panel drives the material,
IBL and camera orbit. While orbiting, the arrow keys reshape the orbit -
up/down raise/lower the camera, left/right widen/tighten the radius.
Everything downstream is identical to the OpenGL ``main.py`` PBR shader,
with the baked HDRI drawn behind the teapot as a skybox.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml QML widget types)
import numpy as np
import wgpu
import wgpu.utils
from bake_settings import BakeSettings, prefilter_key
from ibl_maps import load_maps
from ncca.ngl import (
    FirstPersonCamera,
    Mat4,
    Obj,
    PerspMode,
    PrimData,
    Prims,
    Vec3,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from panel_registry import PanelRegistry
from PySide6.QtCore import QElapsedTimer, QEvent, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

_FLOAT = np.dtype(np.float32).itemsize
_VERTEX_STRIDE = 8 * _FLOAT  # pos3, normal3, uv2 (only pos is used here)

HDRI_DIR = Path(__file__).resolve().parent
SHADER_DIR = HDRI_DIR / "shaders"

BAKE_FORMAT = wgpu.TextureFormat.rgba16float
LUT_FORMAT = wgpu.TextureFormat.rg16float
PRESENT_FORMAT = wgpu.TextureFormat.rgba8unorm

# Cycled by the `E` key: which cube to draw as the skybox, and at what lod.
# The prefilter entry always shows the roughest mip the loaded file has --
# the mip count varies with the file's own settings, so it can't name a
# specific mip here.
DEBUG_VIEWS = ("env", "irradiance", "prefilter (roughest)")

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

# The teapot's transforms + material (matches Transforms in PBR.wgsl).
_TRANSFORM_DTYPE = np.dtype(
    {
        "names": ["MVP", "M", "normalMatrix", "material", "albedo"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
            (np.float32, 4),
        ],
        "offsets": [0, 64, 128, 192, 208],
        "itemsize": _DYNAMIC_STRIDE,
    }
)

# Lights, camera and the IBL toggle, shared by every teapot in a frame
# (matches Scene in PBR.wgsl).
_PBR_SCENE_DTYPE = np.dtype(
    {
        "names": [
            "lightPositions",
            "lightColors",
            "camPos",
            "useIBL",
            "maxReflectionLod",
        ],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
            np.uint32,
            np.float32,
        ],
        "offsets": [0, 64, 128, 144, 148],
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

# Default material for the single teapot, overwritten live by the QML panel.
_DEFAULT_METALLIC = 1.0
_DEFAULT_ROUGHNESS = 0.25
_DEFAULT_AO = 1.0
_DEFAULT_ALBEDO = (1.0, 1.0, 1.0)

# Arrow keys drive the camera; paintWebGPU keeps repainting while any of these
# is held so movement is smooth rather than one step per key event.
_MOVE_KEYS: set = {Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right}

# Multiplier on the camera's own speed so a held arrow key crosses the scene in
# a second or two rather than crawling.
_MOVE_SPEED = 6.0

# While orbiting, the arrow keys reshape the orbit instead of free-flying:
# up/down raise/lower the camera, left/right widen/tighten the radius. Units
# per second at which a held key changes each, and the closest the radius may
# get to the mesh.
_ORBIT_VERTICAL_SPEED = 4.0
_ORBIT_RADIAL_SPEED = 5.0
_ORBIT_MIN_RADIUS = 1.5

# A loaded OBJ can be modelled at any scale or origin, so centre it and fit its
# largest dimension to this many world units - roughly the teapot's size - so it
# always lands in front of the camera at a sensible size.
_MESH_FIT_SIZE = 2.5


class HDRIScene(WebGPUWidget):
    """WebGPU widget that loads pre-baked split-sum IBL textures and draws
    the baked HDRI as a skybox."""

    def __init__(self, maps_path) -> None:
        super().__init__()
        self.setWindowTitle("WebGPU HDRI Image-Based Lighting (baked maps)")
        # Grab keyboard focus so the arrow keys reach keyPressEvent rather than
        # being swallowed by Qt's focus navigation.
        self.setFocusPolicy(Qt.StrongFocus)
        self.msaa_sample_count = 4

        self.maps_path = maps_path
        self.debug_view = 0
        self.use_ibl = True
        self.rotate = False
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.keys_pressed: set = set()

        # Live PBR material for the single teapot, driven by the QML panel.
        self.metallic = _DEFAULT_METALLIC
        self.roughness = _DEFAULT_ROUGHNESS
        self.ao = _DEFAULT_AO
        self.albedo = _DEFAULT_ALBEDO

        # Auto-orbit: when on, the camera circles the origin (the teapot's
        # centre) at whatever radius/height it had when orbit was switched on,
        # so every side comes into view. orbit_speed is radians/second.
        self.orbit = False
        self.orbit_speed = 1.0
        self._orbit_angle = 0.0
        self._orbit_radius = 6.0
        self._orbit_height = 0.0

        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 0.0

        self.camera = FirstPersonCamera(
            Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
        )
        self._update_projection(max(self.width(), 1), max(self.height(), 1))

        self._initialize_web_gpu()
        # Drive a continuous ~60fps repaint via the base widget's render timer
        # (self.startTimer would need a timerEvent override, which WebGPUWidget
        # does not have). Without this the scene only repaints on camera drags,
        # so a single click on the IBL toggle or skybox combo - which changes
        # state but generates no follow-up events - would not flush a new frame.
        self.start_update_timer(16)

    # ------------------------------------------------------------------ setup
    def _initialize_web_gpu(self) -> None:
        try:
            self.device = wgpu.utils.get_default_device()
            self._create_render_buffer()
            self._load_geometry()
            self._create_sampler()
            self._create_skybox_pipeline()

            self._load_baked_maps()

            self._create_pbr_pipeline()
            self._build_teapot()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            traceback.print_exc()
            raise

    def _load_geometry(self) -> None:
        """Upload the unit cube (skybox) and the teapot (grid)."""
        cube = PrimData.primitive(Prims.CUBE.value).astype(np.float32)
        self.cube_geometry = self._make_vertex_buffer(cube)
        teapot = PrimData.primitive(Prims.TEAPOT.value).astype(np.float32)
        self.teapot_geometry = self._make_vertex_buffer(teapot)

    def _make_vertex_buffer(self, data: np.ndarray) -> dict:
        buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        return {"buffer": buffer, "count": data.size // 8}

    def load_obj(self, path) -> None:
        """Replace the drawn mesh with a triangulated OBJ loaded from ``path``.

        Raises ``ValueError`` if the file will not parse or is not triangular,
        so the menu handler can report it without crashing the app.
        """
        obj = Obj()
        try:
            loaded = obj.load(str(path))
        except Exception as err:
            # ncca.ngl raises ObjParse*Error on a malformed file rather than
            # returning False; fold both outcomes into one clear message.
            raise ValueError(f"could not parse OBJ file {path}: {err}") from err
        if not loaded:
            raise ValueError(f"could not parse OBJ file {path}")
        if not obj.is_triangular():
            raise ValueError("only triangulated OBJ meshes are supported")
        data = self._obj_to_vertex_array(obj)
        if data.size == 0:
            raise ValueError("OBJ file contained no faces")
        self.teapot_geometry = self._make_vertex_buffer(data)
        self.update()

    def _obj_to_vertex_array(self, obj) -> np.ndarray:
        """Flatten an Obj's triangles to the pos3/normal3/uv2 vertex layout."""
        has_normals = bool(obj.normals)
        has_uv = bool(obj.uv)
        verts: list[float] = []
        for face in obj.faces:
            for i in range(3):
                v = obj.vertex[face.vertex[i]]
                nx = ny = nz = 0.0
                u = uv = 0.0
                if has_normals:
                    n = obj.normals[face.normal[i]]
                    nx, ny, nz = n.x, n.y, n.z
                if has_uv:
                    t = obj.uv[face.uv[i]]
                    u, uv = t.x, 1.0 - t.y  # flip V to match the teapot's UVs
                verts.extend((v.x, v.y, v.z, nx, ny, nz, u, uv))
        return self._fit_mesh(np.array(verts, dtype=np.float32))

    def _fit_mesh(self, data: np.ndarray) -> np.ndarray:
        """Centre a flat vertex array on the origin and scale it to fit."""
        if data.size == 0:
            return data
        verts = data.reshape(-1, 8)
        pos = verts[:, 0:3]
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        extent = float((hi - lo).max())
        scale = _MESH_FIT_SIZE / extent if extent > 1e-6 else 1.0
        verts[:, 0:3] = (pos - (lo + hi) * 0.5) * scale
        return verts.reshape(-1)

    def _create_sampler(self) -> None:
        self.linear_sampler = self.device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
        )

    # ------------------------------------------------------------- map load
    def _upload_cube(self, faces: np.ndarray, size: int, mips: list | None = None):
        """Create a cube texture and fill it from (6,size,size,4) float16 arrays.

        `faces` is mip 0. `mips`, when given, is a list of extra
        (6, size>>m, size>>m, 4) arrays for mips 1..n.
        """
        levels = [faces] + (mips or [])
        tex = self.device.create_texture(
            size=(size, size, 6),
            mip_level_count=len(levels),
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        for mip, level in enumerate(levels):
            level_size = size >> mip
            data = np.ascontiguousarray(level.astype(np.float16))
            for face in range(6):
                self.device.queue.write_texture(
                    {"texture": tex, "mip_level": mip, "origin": (0, 0, face)},
                    data[face].tobytes(),
                    {"bytes_per_row": level_size * 4 * 2, "rows_per_image": level_size},
                    (level_size, level_size, 1),
                )
        return tex

    def _upload_lut(self, lut: np.ndarray):
        h, w = lut.shape[:2]
        data = np.ascontiguousarray(lut.astype(np.float16))
        tex = self.device.create_texture(
            size=(w, h, 1),
            format=LUT_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": tex},
            data.tobytes(),
            {"bytes_per_row": w * 2 * 2, "rows_per_image": h},
            (w, h, 1),
        )
        return tex

    def _upload_maps(self, maps: dict) -> None:
        """Upload a loaded map set to GPU cube/2D textures.

        Sizes come from the file's own settings, not from a constant here --
        a map set baked at any resolution has to land correctly.
        """
        settings: BakeSettings = maps["settings"]
        self.settings = settings
        self.env_cube = self._upload_cube(maps["env"], settings.env_size)
        self.irradiance_cube = self._upload_cube(
            maps["irradiance"], settings.irradiance_size
        )
        prefilter_mips = [
            maps[prefilter_key(m)] for m in range(1, settings.prefilter_mips)
        ]
        self.prefilter_cube = self._upload_cube(
            maps[prefilter_key(0)], settings.prefilter_size, prefilter_mips
        )
        self.brdf_lut = self._upload_lut(maps["brdf_lut"])

        # On startup this runs before the scene UBO exists, so there's
        # nothing to update yet -- the pipeline-creation site sets the
        # initial value instead. This guard only fires on reload_maps.
        if hasattr(self, "pbr_scene_uniforms"):
            self.pbr_scene_uniforms["maxReflectionLod"] = float(
                settings.prefilter_mips - 1
            )

    def _load_baked_maps(self) -> None:
        try:
            maps = load_maps(self.maps_path)
        except (OSError, ValueError) as err:
            raise SystemExit(f"Could not load IBL maps from {self.maps_path!r}: {err}")
        self._upload_maps(maps)

    def reload_maps(self, path) -> None:
        """Load a different IBL ``.npz`` at runtime and rebind it.

        Raises ``OSError``/``ValueError`` (from :func:`load_maps`) so the menu
        handler can report a bad file without tearing the whole app down.
        """
        maps = load_maps(path)
        self._upload_maps(maps)
        self.maps_path = path
        # The PBR bind group holds views of the irradiance/prefilter/LUT
        # textures we just replaced, so rebuild it; the skybox rebinds its
        # source per frame and picks up the new cubes on its own.
        self._create_ibl_bind_group()
        self.update()

    # --------------------------------------------------------- PBR teapots
    def _create_pbr_pipeline(self) -> None:
        """Create the split-sum IBL PBR pipeline and its bind group layouts."""
        with open(SHADER_DIR / "PBR.wgsl", "r") as f:
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
        self.pbr_scene_uniforms["maxReflectionLod"] = float(
            self.settings.prefilter_mips - 1
        )
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

        self._create_ibl_bind_group()

    def _create_ibl_bind_group(self) -> None:
        """(Re)bind the current IBL textures into @group(2) for the PBR pass."""
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

    def _build_teapot(self) -> None:
        """Set up the single teapot's transform/material uniform buffer."""
        self.teapot_model = Mat4()
        self.teapot_transform_uniforms = np.zeros((), dtype=_TRANSFORM_DTYPE)
        self.teapot_transform_buffer = self.device.create_buffer(
            size=self.teapot_transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="teapot_transform_buffer",
        )
        self.teapot_transform_bind_group = self.device.create_bind_group(
            layout=self.pbr_transform_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.teapot_transform_buffer,
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

    def _write_teapot_transform_uniforms(self) -> None:
        model = self.teapot_model
        model_view = self.camera.view @ model
        mvp = self.camera.projection @ model_view
        normal_matrix = model_view.copy().inverse().transposed()
        slot = self.teapot_transform_uniforms
        slot["MVP"] = mvp.to_numpy()
        slot["M"] = model.to_numpy()
        slot["normalMatrix"] = normal_matrix.to_numpy()
        slot["material"] = (self.metallic, self.roughness, self.ao, 0.0)
        slot["albedo"] = (*self.albedo, 1.0)
        self.device.queue.write_buffer(
            self.teapot_transform_buffer, 0, self.teapot_transform_uniforms.tobytes()
        )

    def _draw_teapot(self, render_pass: "wgpu.GPURenderPassEncoder") -> None:
        render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
        render_pass.set_pipeline(self.pbr_pipeline)
        render_pass.set_bind_group(0, self.teapot_transform_bind_group, [0])
        render_pass.set_bind_group(1, self.pbr_scene_bind_group)
        render_pass.set_bind_group(2, self.pbr_ibl_bind_group)
        render_pass.set_vertex_buffer(0, self.teapot_geometry["buffer"])
        render_pass.draw(self.teapot_geometry["count"])

    # --------------------------------------------------------- QML panel slots
    @Slot(float)
    def set_metallic(self, value: float) -> None:
        self.metallic = value
        self.update()

    @Slot(float)
    def set_roughness(self, value: float) -> None:
        # Clamp away from zero so perfectly smooth mirrors don't alias.
        self.roughness = max(0.05, value)
        self.update()

    @Slot(float)
    def set_ao(self, value: float) -> None:
        self.ao = value
        self.update()

    @Slot(float, float, float)
    def set_albedo(self, r: float, g: float, b: float) -> None:
        self.albedo = (r, g, b)
        self.update()

    @Slot(bool)
    def set_use_ibl(self, enabled: bool) -> None:
        self.use_ibl = enabled
        self.update()

    @Slot(int)
    def set_debug_view(self, index: int) -> None:
        self.debug_view = index % len(DEBUG_VIEWS)
        self.update()

    @Slot(bool)
    def set_orbit(self, enabled: bool) -> None:
        # Capture the current radius/height/angle so the orbit picks up from
        # wherever the camera is now, without a jump.
        if enabled:
            eye = self.camera.eye
            radius = math.hypot(eye.x, eye.z)
            self._orbit_radius = radius if radius > 1e-3 else 6.0
            self._orbit_height = eye.y
            self._orbit_angle = math.atan2(eye.x, eye.z)
        self.orbit = enabled
        self.update()

    @Slot(float)
    def set_orbit_speed(self, value: float) -> None:
        self.orbit_speed = value

    def _advance_orbit(self, delta_time: float) -> None:
        if not self.orbit:
            return
        self._orbit_angle += self.orbit_speed * delta_time
        r = self._orbit_radius
        a = self._orbit_angle
        self.camera.eye = Vec3(r * math.sin(a), self._orbit_height, r * math.cos(a))
        # Point the camera back at the origin by deriving yaw/pitch from the
        # eye->centre direction, then let the camera rebuild its view.
        front = Vec3(-self.camera.eye.x, -self.camera.eye.y, -self.camera.eye.z)
        front = front.normalized()
        self.camera.yaw = math.degrees(math.atan2(front.z, front.x))
        self.camera.pitch = math.degrees(math.asin(max(-1.0, min(1.0, front.y))))
        self.camera._update_camera_vectors()

    # ------------------------------------------------------------- skybox
    def _create_skybox_pipeline(self) -> None:
        with open(SHADER_DIR / "Skybox.wgsl", "r") as f:
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
        elif name == "prefilter (roughest)":
            cube, lod = self.prefilter_cube, float(self.settings.prefilter_mips - 1)
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
        self._advance_orbit(delta_time)

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
        self._write_teapot_transform_uniforms()

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
        # Draw the teapot first (writes depth), then the skybox behind it
        # (depth test only, no depth write) sharing the same depth buffer.
        self._draw_teapot(render_pass)
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
        vertical = float(Qt.Key_Up in self.keys_pressed) - float(
            Qt.Key_Down in self.keys_pressed
        )
        horizontal = float(Qt.Key_Right in self.keys_pressed) - float(
            Qt.Key_Left in self.keys_pressed
        )
        if self.orbit:
            # While orbiting the arrow keys reshape the orbit rather than
            # free-fly: up/down raise/lower the camera, left/right widen or
            # tighten the radius. _advance_orbit reads these back this frame.
            if vertical:
                self._orbit_height += vertical * _ORBIT_VERTICAL_SPEED * delta_time
            if horizontal:
                self._orbit_radius = max(
                    _ORBIT_MIN_RADIUS,
                    self._orbit_radius + horizontal * _ORBIT_RADIAL_SPEED * delta_time,
                )
            return
        # Free-fly: up/down move along the view direction (forward/back),
        # left/right strafe. camera.move(x, y, delta) shifts the eye by
        # front*x and right*y, so this reads directly off the held keys.
        if vertical or horizontal:
            self.camera.move(
                vertical * _MOVE_SPEED, horizontal * _MOVE_SPEED, delta_time
            )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        self.keys_pressed.add(key)
        if key == Qt.Key_Escape:
            # The scene is the central widget of a QMainWindow, so closing
            # itself only hides the scene and leaves the window (and app)
            # running - close the top-level window instead.
            self.window().close()
        elif key == Qt.Key_E:
            self.debug_view = (self.debug_view + 1) % len(DEBUG_VIEWS)
        elif key == Qt.Key_I:
            self.use_ibl = not self.use_ibl
        elif key == Qt.Key_Space:
            self.camera = FirstPersonCamera(
                Vec3(0, 0, 6), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
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


class OverlayQuickWidget(QQuickWidget):
    """Transparent QQuickWidget carrying the PBR control panels.

    Clicks (and wheel) that land outside any registered panel are forwarded
    to the WebGPU scene beneath so the camera still rotates/zooms; clicks on a
    panel go to QML as normal. Mirrors GUIDemos/QMLWebGPUOverlay.

    Drag ownership is decided once, on press: a drag that starts on a panel
    stays with QML for its whole life, and one that starts on empty space
    stays with the camera - even if the cursor briefly crosses the other's
    area mid-drag. Re-running hit_test per move event instead let a fast panel
    drag (whose cursor can outrun the panel's trailing rect) leak moves to the
    camera, panning it while repositioning a panel.
    """

    def __init__(self, scene: HDRIScene, registry: PanelRegistry, parent=None) -> None:
        super().__init__(parent)
        self._scene = scene
        self._registry = registry
        # While a mouse button is held, True routes the drag to the scene
        # (camera), False to QML (panel). Set on press, used until release.
        self._drag_to_scene = False
        ncca.ngl.qml.add_import_path(self.engine())
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setClearColor(Qt.GlobalColor.transparent)
        self.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        # Let the arrow keys / Space reach the scene rather than the overlay.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _forward_mouse(self, event: QMouseEvent) -> None:
        forwarded = QMouseEvent(
            event.type(),
            self._scene.mapFromGlobal(event.globalPosition()),
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        QApplication.sendEvent(self._scene, forwarded)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._drag_to_scene = not self._registry.hit_test(event.position())
        if self._drag_to_scene:
            event.ignore()
            self._forward_mouse(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Mid-drag (a button is held) stick with the press's decision; only a
        # button-free hover re-tests, so QML still gets hover events over panels.
        if event.buttons():
            to_scene = self._drag_to_scene
        else:
            to_scene = not self._registry.hit_test(event.position())
        if to_scene:
            event.ignore()
            self._forward_mouse(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_to_scene:
            event.ignore()
            self._forward_mouse(event)
        else:
            super().mouseReleaseEvent(event)
        self._drag_to_scene = False

    def wheelEvent(self, event) -> None:
        if self._registry.hit_test(event.position()):
            super().wheelEvent(event)
        else:
            event.ignore()
            QApplication.sendEvent(self._scene, event)


class MainWindow(QMainWindow):
    """Hosts the WebGPU teapot scene with the QML control overlay on top."""

    def __init__(self, maps_path: str) -> None:
        super().__init__()
        self.setWindowTitle("WebGPU HDRI IBL - single teapot with PBR controls")
        self.resize(1024, 720)

        self.scene = HDRIScene(maps_path)
        self.setCentralWidget(self.scene)

        self._build_menu()

        self.registry = PanelRegistry()
        self.overlay = OverlayQuickWidget(self.scene, self.registry, self.scene)
        self.overlay.rootContext().setContextProperty("panelRegistry", self.registry)
        self.overlay.rootContext().setContextProperty("scene", self.scene)
        self.overlay.setSource(QUrl.fromLocalFile(str(HDRI_DIR / "main.qml")))
        self.overlay.setGeometry(self.scene.rect())

    def _build_menu(self) -> None:
        """A File menu to swap the IBL maps or the drawn mesh at runtime."""
        file_menu = self.menuBar().addMenu("&File")

        load_maps_action = QAction("Load IBL &Maps…", self)
        load_maps_action.triggered.connect(self._on_load_maps)
        file_menu.addAction(load_maps_action)

        load_mesh_action = QAction("Load &Mesh…", self)
        load_mesh_action.triggered.connect(self._on_load_mesh)
        file_menu.addAction(load_mesh_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _on_load_maps(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load IBL maps",
            str(Path(self.scene.maps_path).parent),
            "IBL maps (*.npz)",
        )
        if not path:
            return
        try:
            self.scene.reload_maps(path)
        except (OSError, ValueError) as err:
            QMessageBox.critical(self, "Could not load IBL maps", str(err))

    def _on_load_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load OBJ mesh", str(HDRI_DIR), "Wavefront OBJ (*.obj)"
        )
        if not path:
            return
        try:
            self.scene.load_obj(path)
        except (OSError, ValueError) as err:
            QMessageBox.critical(self, "Could not load mesh", str(err))

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self.overlay.setGeometry(self.scene.rect())


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
        "--maps",
        default=str(HDRI_DIR / "ibl_maps.npz"),
        help="path to the .npz of baked IBL maps (default: bundled ibl_maps.npz)",
    )
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

    # Fusion so ncca.ngl.qml's controls and DraggablePanel's dark chrome render
    # consistently (the native macOS style ignores the Frame background); see
    # GUIDemos/QMLWebGPUOverlay/main.py for the full rationale.
    QQuickStyle.setStyle("Fusion")
    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    app.setOrganizationName("NCCA")
    app.setApplicationName("HDRIBakerDemo")

    win = MainWindow(args.maps)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
