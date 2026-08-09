#!/usr/bin/env -S uv run --script
"""
SciFiUI WebGPU -- the retro CRT terminal demo ported from OpenGL.

Run with:  uv run SciFiUI/WebGPUmain.py
Drag with the left mouse button inside the terrain panel to rotate the view.
Keys: Esc quit, Space hold/resume, Up/Down velocity, P phosphor, S scan fx,
R reset view.
"""

import argparse
import sys
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, ortho, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication

TERRAIN_ROWS = 42
TERRAIN_COLS = 130
TERRAIN_HALF_WIDTH = 7.0
TERRAIN_NEAR_Z = -3.5
TERRAIN_ROW_STEP = 0.8
TERRAIN_FLOOR_Y = -2.0

LOG_MESSAGES = [
    "NAV REF GRID LOCKED",
    "TERRAIN DELTA 0.031 NOMINAL",
    "ATMOSPHERE: N2 85% / AR 10% / CO2 5%",
    "HULL INTEGRITY 100%",
    "SIGNAL SOURCE BEARING 042 MK 12",
    "TRANSMISSION REPEATS EVERY 12 SEC",
    "CRYO BAY 2 ... STANDBY",
    "BIO SCAN SWEEP ... NEGATIVE",
    "ORDER 937 ... FILE SEALED",
    "AIRLOCK B PRESSURE EQUALIZED",
    "DUST STORM VECTOR 12KM SSW",
    "FUEL CELLS AT 87%",
    "LANDING STRUTS ... ARMED",
    "PROXIMITY ALERT DISENGAGED",
    "COOLANT LOOP 2 PURGED",
    "ANTENNA ARRAY REALIGNED",
    "GRAVITY 0.86G CONSTANT",
    "ALL SYSTEMS NOMINAL",
]

LOG_INTERVAL = 55


def matrix_uniform_values(matrix: Mat4) -> np.ndarray:
    return matrix.to_numpy().astype(np.float32).reshape(-1)


class DrawRange:
    def __init__(
        self, topology: str, first: int, count: int, colour: tuple[float, ...]
    ):
        self.topology = topology
        self.first = first
        self.count = count
        self.colour = colour


class ValueNoise:
    def __init__(self, size: int = 256, seed: int = 1979) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.grid = rng.random((size, size)).astype(np.float32)

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        xi = np.floor(x).astype(np.int64)
        zi = np.floor(z).astype(np.int64)
        xf = (x - xi).astype(np.float32)
        zf = (z - zi).astype(np.float32)
        xi %= self.size
        zi %= self.size
        xj = (xi + 1) % self.size
        zj = (zi + 1) % self.size
        u = xf * xf * (3.0 - 2.0 * xf)
        v = zf * zf * (3.0 - 2.0 * zf)
        g = self.grid
        a = g[zi, xi]
        b = g[zi, xj]
        c = g[zj, xi]
        d = g[zj, xj]
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


class TerrainData:
    def __init__(self) -> None:
        self.noise = ValueNoise()
        self.xs = np.linspace(
            -TERRAIN_HALF_WIDTH, TERRAIN_HALF_WIDTH, TERRAIN_COLS, dtype=np.float32
        )
        self.envelope = np.exp(-(np.abs(self.xs / (TERRAIN_HALF_WIDTH * 0.55)) ** 3.0))
        self.line_verts = np.zeros((TERRAIN_ROWS, TERRAIN_COLS, 3), dtype=np.float32)
        self.skirt_verts = np.zeros(
            (TERRAIN_ROWS, TERRAIN_COLS * 2, 3), dtype=np.float32
        )
        for row in range(TERRAIN_ROWS):
            z = TERRAIN_NEAR_Z - row * TERRAIN_ROW_STEP
            self.line_verts[row, :, 0] = self.xs
            self.line_verts[row, :, 2] = z
            self.skirt_verts[row, 0::2, 0] = self.xs
            self.skirt_verts[row, 1::2, 0] = self.xs
            self.skirt_verts[row, 1::2, 1] = TERRAIN_FLOOR_Y
            self.skirt_verts[row, :, 2] = z

    def update(self, flight_t: float) -> None:
        row_idx = np.arange(TERRAIN_ROWS, dtype=np.float32)[:, None]
        nx = (self.xs[None, :] + TERRAIN_HALF_WIDTH) * 0.55
        nz = (row_idx * TERRAIN_ROW_STEP + flight_t) * 0.35
        n1 = self.noise.sample(nx, nz)
        n2 = self.noise.sample(nx * 2.7 + 11.3, nz * 2.3 + 7.7)
        n3 = self.noise.sample(nx * 1.3 + 51.0, nz * 1.1 + 23.0)
        heights = self.envelope[None, :] * (1.5 * n1 + 0.5 * n2 + 3.0 * np.power(n3, 6))
        self.line_verts[:, :, 1] = heights
        self.skirt_verts[:, 0::2, 1] = heights

    def line_segments(self) -> np.ndarray:
        segments = np.empty((TERRAIN_ROWS, (TERRAIN_COLS - 1) * 2, 3), dtype=np.float32)
        segments[:, 0::2, :] = self.line_verts[:, :-1, :]
        segments[:, 1::2, :] = self.line_verts[:, 1:, :]
        return segments

    def skirt_triangles(self) -> np.ndarray:
        top_left = self.skirt_verts[:, 0:-2:2, :]
        bottom_left = self.skirt_verts[:, 1:-1:2, :]
        top_right = self.skirt_verts[:, 2::2, :]
        bottom_right = self.skirt_verts[:, 3::2, :]
        triangles = np.empty(
            (TERRAIN_ROWS, (TERRAIN_COLS - 1) * 6, 3), dtype=np.float32
        )
        triangles[:, 0::6, :] = top_left
        triangles[:, 1::6, :] = bottom_left
        triangles[:, 2::6, :] = top_right
        triangles[:, 3::6, :] = top_right
        triangles[:, 4::6, :] = bottom_left
        triangles[:, 5::6, :] = bottom_right
        return triangles


class UIBatchData:
    def __init__(self) -> None:
        self._verts: list[tuple[float, float, float]] = []
        self.ranges: list[DrawRange] = []

    @property
    def vertices(self) -> np.ndarray:
        return np.array(self._verts, dtype=np.float32)

    def begin(self) -> None:
        self._verts = []
        self.ranges = []

    def _push(
        self, topology: str, pts: list[tuple[float, float]], colour: tuple
    ) -> None:
        first = len(self._verts)
        self._verts.extend((float(x), float(y), 0.0) for x, y in pts)
        self.ranges.append(DrawRange(topology, first, len(pts), colour))

    def rect(self, x, y, w, h, colour) -> None:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]
        self._push("triangle-list", pts, colour)

    def outline(self, x, y, w, h, colour) -> None:
        pts = [(x, y), (x + w, y), (x + w, y), (x + w, y + h)]
        pts += [(x + w, y + h), (x, y + h), (x, y + h), (x, y)]
        self._push("line-list", pts, colour)

    def line(self, x0, y0, x1, y1, colour) -> None:
        self._push("line-list", [(x0, y0), (x1, y1)], colour)


class SciFiWebGPU(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SEMIOTIC STANDARD // MU-TH-UR 6000 // WebGPU")
        self.msaa_sample_count = 4
        self.device = wgpu.utils.get_default_device()
        self.shader_dir = Path(__file__).parent / "shaders"
        self.terrain = TerrainData()
        self.ui = UIBatchData()
        self.tick = 0
        self.flight_t = 0.0
        self.speed = 1.0
        self.paused = False
        self.amber = False
        self.effects_on = True
        self.spin_x_face = 0.0
        self.spin_y_face = 0.0
        self.buttons: list[dict] = []
        self.hover_id: str | None = None
        self.flash: dict[str, int] = {}
        self.log: deque = deque(maxlen=64)
        self.log_msg_index = 0
        self.last_log_tick = 0
        self.rotating = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.view = look_at(Vec3(0.0, 4.5, 9.0), Vec3(0.0, 1.0, -14.0), Vec3(0, 1, 0))
        self.scene_texture_size = (0, 0)
        self.frame_bind_groups = []
        self._initialize_webgpu()
        self.log_line("INTERFACE 2037 READY FOR INQUIRY")
        self.log_line("TERRAIN SCAN COMMENCED")
        self.startTimer(16)
        self.update()

    def _initialize_webgpu(self) -> None:
        self._create_render_buffer()
        self._create_scene_texture()
        self._create_uniform_buffers()
        self._create_pipelines()
        self._create_bind_groups()

    def _create_scene_texture(self) -> None:
        self.scene_texture_size = self.texture_size
        self.scene_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.scene_texture_view = self.scene_texture.create_view()
        self.scene_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )

    def _create_uniform_buffers(self) -> None:
        self.ui_uniform_buffer = self.device.create_buffer(
            size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.crt_uniform_buffer = self.device.create_buffer(
            size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def _create_pipelines(self) -> None:
        ui_code = (self.shader_dir / "UIShader.wgsl").read_text()
        crt_code = (self.shader_dir / "CRTShader.wgsl").read_text()
        text_code = (self.shader_dir / "TextShader.wgsl").read_text()
        self.ui_shader = self.device.create_shader_module(code=ui_code)
        self.crt_shader = self.device.create_shader_module(code=crt_code)
        self.text_shader = self.device.create_shader_module(code=text_code)

        self.ui_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.crt_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )
        self.text_layout = self.device.create_bind_group_layout(
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
        self.ui_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.ui_layout]
        )
        self.crt_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.crt_layout]
        )
        self.text_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.text_layout]
        )
        vertex_layout = [
            {
                "array_stride": 12,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 0,
                    }
                ],
            }
        ]
        self.triangle_pipeline = self._make_ui_pipeline(
            wgpu.PrimitiveTopology.triangle_list, vertex_layout
        )
        self.triangle_strip_pipeline = self._make_ui_pipeline(
            wgpu.PrimitiveTopology.triangle_strip, vertex_layout
        )
        self.line_pipeline = self._make_ui_pipeline(
            wgpu.PrimitiveTopology.line_list, vertex_layout
        )
        self.line_strip_pipeline = self._make_ui_pipeline(
            wgpu.PrimitiveTopology.line_strip, vertex_layout
        )
        self.crt_pipeline = self.device.create_render_pipeline(
            label="scifi_crt_pipeline",
            layout=self.crt_pipeline_layout,
            vertex={
                "module": self.crt_shader,
                "entry_point": "vertex_main",
                "buffers": [],
            },
            fragment={
                "module": self.crt_shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            multisample={"count": self.msaa_sample_count},
        )
        self.text_pipeline = self.device.create_render_pipeline(
            label="scifi_text_pipeline",
            layout=self.text_pipeline_layout,
            vertex={
                "module": self.text_shader,
                "entry_point": "vertex_main",
                "buffers": [],
            },
            fragment={
                "module": self.text_shader,
                "entry_point": "fragment_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                    }
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            multisample={"count": 1},
        )

    def _make_ui_pipeline(self, topology, vertex_layout):
        return self.device.create_render_pipeline(
            label=f"scifi_ui_{topology}",
            layout=self.ui_pipeline_layout,
            vertex={
                "module": self.ui_shader,
                "entry_point": "vertex_main",
                "buffers": vertex_layout,
            },
            fragment={
                "module": self.ui_shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": topology},
            multisample={"count": 1},
        )

    def _create_bind_groups(self) -> None:
        self.ui_bind_group = self.device.create_bind_group(
            layout=self.ui_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.ui_uniform_buffer,
                        "offset": 0,
                        "size": 80,
                    },
                }
            ],
        )
        self.crt_bind_group = self.device.create_bind_group(
            layout=self.crt_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.crt_uniform_buffer,
                        "offset": 0,
                        "size": 48,
                    },
                },
                {"binding": 1, "resource": self.scene_texture_view},
                {"binding": 2, "resource": self.scene_sampler},
            ],
        )

    def _make_ui_bind_group(self, mvp: Mat4, colour: tuple[float, ...]):
        data = np.zeros(20, dtype=np.float32)
        data[:16] = matrix_uniform_values(mvp)
        data[16:20] = np.array(colour, dtype=np.float32)
        uniform_buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.UNIFORM
        )
        bind_group = self.device.create_bind_group(
            layout=self.ui_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": uniform_buffer, "offset": 0, "size": 80},
                }
            ],
        )
        self.frame_bind_groups.append(bind_group)
        return bind_group

    def _write_crt_uniform(self) -> None:
        phosphor = (1.0, 0.72, 0.22, 1.0) if self.amber else (0.35, 1.0, 0.45, 1.0)
        data = np.array(
            [
                self.tick * 0.016,
                1.0 if self.effects_on else 0.0,
                0.0,
                0.0,
                float(self.texture_size[0]),
                float(self.texture_size[1]),
                0.0,
                0.0,
                *phosphor,
            ],
            dtype=np.float32,
        )
        self.device.queue.write_buffer(self.crt_uniform_buffer, 0, data.tobytes())

    def _upload_vertices(self, vertices: np.ndarray):
        return self.device.create_buffer_with_data(
            data=np.ascontiguousarray(vertices, dtype=np.float32),
            usage=wgpu.BufferUsage.VERTEX,
        )

    def _layout(self) -> dict:
        w, h = self.texture_size
        s = h / 800.0
        m = int(14 * s)
        header_h = int(52 * s)
        footer_h = int(40 * s)
        left_w = int(220 * s)
        right_w = int(310 * s)
        top = header_h + m
        bottom = h - footer_h - m
        return {
            "s": s,
            "m": m,
            "header": (m, m, w - 2 * m, header_h - m),
            "footer": (m, h - footer_h, w - 2 * m, footer_h - m),
            "left": (m, top, left_w, bottom - top),
            "right": (w - m - right_w, top, right_w, bottom - top),
            "centre": (2 * m + left_w, top, w - 4 * m - left_w - right_w, bottom - top),
        }

    def _button_defs(self) -> list[tuple[str, str]]:
        return [
            ("hold", "RESUME" if self.paused else "HOLD"),
            ("vel+", "VEL +"),
            ("vel-", "VEL -"),
            ("phos", "PHOS: AMBER" if self.amber else "PHOS: GREEN"),
            ("scan", f"SCAN FX {'ON' if self.effects_on else 'OFF'}"),
            ("purge", "PURGE LOG"),
        ]

    def _draw_range(
        self, render_pass, vertices: np.ndarray, draw_range: DrawRange, mvp: Mat4
    ):
        pipeline = {
            "triangle-list": self.triangle_pipeline,
            "triangle-strip": self.triangle_strip_pipeline,
            "line-list": self.line_pipeline,
            "line-strip": self.line_strip_pipeline,
        }[draw_range.topology]
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(
            0, self._make_ui_bind_group(mvp, draw_range.colour), [], 0, 99
        )
        render_pass.draw(draw_range.count, 1, draw_range.first, 0)

    def _draw_terrain(self, render_pass, lay: dict) -> None:
        cx, cy, cw, ch = lay["centre"]
        pad = lay["m"]
        vx, vy = cx + pad, cy + pad
        vw, vh = cw - 2 * pad, ch - 2 * pad
        if vw <= 1 or vh <= 1:
            return
        render_pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0)
        render_pass.set_scissor_rect(vx, vy, vw, vh)
        project = perspective(40.0, float(vw) / float(vh), 0.1, 200.0, PerspMode.WebGPU)
        pivot_z = TERRAIN_NEAR_Z - (TERRAIN_ROWS / 2.0) * TERRAIN_ROW_STEP
        rot = Mat4.rotate_y(self.spin_y_face) @ Mat4.rotate_x(self.spin_x_face)
        model = (
            Mat4.translate(0.0, 0.0, pivot_z) @ rot @ Mat4.translate(0.0, 0.0, -pivot_z)
        )
        mvp = project @ self.view @ model
        self.terrain.update(self.flight_t)
        line_segments = self.terrain.line_segments()
        skirt_triangles = self.terrain.skirt_triangles()
        line_buffer = self._upload_vertices(line_segments.reshape(-1, 3))
        skirt_buffer = self._upload_vertices(skirt_triangles.reshape(-1, 3))
        skirt_row = (TERRAIN_COLS - 1) * 6
        line_row = (TERRAIN_COLS - 1) * 2
        for row in range(TERRAIN_ROWS - 1, -1, -1):
            render_pass.set_vertex_buffer(0, skirt_buffer)
            render_pass.set_pipeline(self.triangle_pipeline)
            render_pass.set_bind_group(
                0, self._make_ui_bind_group(mvp, (0.0, 0.0, 0.0, 1.0)), [], 0, 99
            )
            render_pass.draw(skirt_row, 1, row * skirt_row, 0)

            render_pass.set_vertex_buffer(0, line_buffer)
            fade = 0.35 + 0.65 * (1.0 - row / TERRAIN_ROWS)
            render_pass.set_pipeline(self.line_pipeline)
            render_pass.set_bind_group(
                0, self._make_ui_bind_group(mvp, (fade, fade, fade, 1.0)), [], 0, 99
            )
            render_pass.draw(line_row, 1, row * line_row, 0)
        render_pass.set_viewport(
            0, 0, self.texture_size[0], self.texture_size[1], 0.0, 1.0
        )
        render_pass.set_scissor_rect(0, 0, self.texture_size[0], self.texture_size[1])

    def _build_chrome(self, lay: dict) -> np.ndarray:
        s = lay["s"]
        self.ui.begin()
        dim = (0.10, 0.10, 0.10, 1.0)
        mid = (0.45, 0.45, 0.45, 1.0)
        hot = (1.0, 1.0, 1.0, 1.0)
        for key in ("header", "footer", "left", "right"):
            x, y, pw, ph = lay[key]
            self.ui.rect(x, y, pw, ph, dim)
            self.ui.outline(x, y, pw, ph, mid)
        self.ui.outline(*lay["centre"], mid)
        cx, cy, cw, ch = lay["centre"]
        b = int(26 * s)
        for px, py, dx, dy in (
            (cx, cy, 1, 1),
            (cx + cw, cy, -1, 1),
            (cx, cy + ch, 1, -1),
            (cx + cw, cy + ch, -1, -1),
        ):
            self.ui.line(px, py, px + dx * b, py, hot)
            self.ui.line(px, py, px, py + dy * b, hot)
        self.buttons = []
        lx, ly, lw, _lh = lay["left"]
        pad = int(12 * s)
        bh = int(52 * s)
        y = ly + pad
        for bid, label in self._button_defs():
            rect = (lx + pad, y, lw - 2 * pad, bh)
            self.buttons.append({"id": bid, "label": label, "rect": rect})
            flashing = self.tick < self.flash.get(bid, -1)
            hovered = self.hover_id == bid
            fill = (0.85, 0.85, 0.85, 1.0) if flashing else (0.30, 0.30, 0.30, 1.0)
            if not flashing and not hovered:
                fill = (0.16, 0.16, 0.16, 1.0)
            self.ui.rect(*rect, fill)
            self.ui.outline(*rect, hot if (hovered or flashing) else mid)
            y += bh + pad
        rx, ry, rw, _rh = lay["right"]
        self.ui.line(rx, ry + int(34 * s), rx + rw, ry + int(34 * s), mid)
        return self.ui.vertices

    def _draw_chrome(self, render_pass, lay: dict) -> None:
        vertices = self._build_chrome(lay)
        if vertices.size == 0:
            return
        buffer = self._upload_vertices(vertices)
        render_pass.set_vertex_buffer(0, buffer)
        mvp = ortho(
            0.0,
            float(self.texture_size[0]),
            float(self.texture_size[1]),
            0.0,
            -1.0,
            1.0,
        )
        for draw_range in self.ui.ranges:
            self._draw_range(render_pass, vertices, draw_range, mvp)

    def _render_text_image(self, lay: dict) -> np.ndarray:
        width, height = self.texture_size
        image = QImage(width, height, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        s = lay["s"]
        bright = QColor(245, 245, 245, 255)
        faint = QColor(140, 140, 140, 255)

        def text(font_size, x, y, message, colour=bright):
            painter.setPen(colour)
            painter.setFont(QFont("Arial", max(1, int(font_size))))
            painter.drawText(int(x), int(y), message)

        hx, hy, hw, hh = lay["header"]
        text(
            26 * s,
            hx + int(14 * s),
            hy + hh - int(10 * s),
            "WEYLAND-YUTANI CORP // SEMIOTIC STANDARD",
        )
        clock = f"MU-TH-UR 6000  T+{self.tick // 60:06d}"
        text(20 * s, hx + hw - int(320 * s), hy + hh - int(12 * s), clock, faint)

        cx, cy, cw, ch = lay["centre"]
        text(
            20 * s,
            cx + int(16 * s),
            cy + int(30 * s),
            "TERRAIN SCAN :: LV-426 APPROACH",
        )
        vel = 0.0 if self.paused else self.speed
        status = (
            f"VEL {vel:4.2f} :: ALT 3.7KM :: MODE {'HOLD' if self.paused else 'FLY'}"
            f" :: VIEW {self.spin_y_face:+04.0f}/{self.spin_x_face:+04.0f}"
        )
        text(16 * s, cx + int(16 * s), cy + ch - int(12 * s), status, faint)

        for btn in self.buttons:
            bx, by, _bw, bh = btn["rect"]
            flashing = self.tick < self.flash.get(btn["id"], -1)
            colour = QColor(10, 10, 10) if flashing else bright
            text(
                20 * s,
                bx + int(14 * s),
                by + bh // 2 + int(8 * s),
                btn["label"],
                colour,
            )
        lx, ly, lw, lh = lay["left"]
        text(
            16 * s, lx + int(12 * s), ly + lh - int(12 * s), "PANEL 04 // INPUT", faint
        )

        rx, ry, rw, rh = lay["right"]
        text(20 * s, rx + int(12 * s), ry + int(24 * s), "SYSTEM LOG")
        line_h = int(24 * s)
        y = ry + rh - int(14 * s)
        progress = min(1.0, (self.tick - self.last_log_tick) / 20.0)
        max_chars = int((rw - 24 * s) / (8.5 * s))
        painter.save()
        painter.setClipRect(rx, ry + int(40 * s), rw, rh - int(40 * s))
        for i, (stamp, msg) in enumerate(reversed(self.log)):
            line = f"{stamp} {msg}"[:max_chars]
            if i == 0:
                line = line[: max(1, int(len(line) * progress))]
            text(16 * s, rx + int(12 * s), y, line, bright if i == 0 else faint)
            y -= line_h
            if y < ry + int(60 * s):
                break
        painter.restore()

        fx, fy, _fw, fh = lay["footer"]
        cursor = "_" if (self.tick // 30) % 2 == 0 else " "
        text(
            20 * s,
            fx + int(14 * s),
            fy + fh - int(8 * s),
            f"INTERFACE 2037 READY FOR INQUIRY {cursor}",
        )
        painter.end()
        return (
            np.frombuffer(image.bits(), dtype=np.uint8)
            .reshape((height, width, 4))
            .copy()
        )

    def _draw_text(self, encoder, lay: dict) -> None:
        image_data = self._render_text_image(lay)
        texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            image_data.tobytes(),
            {
                "bytes_per_row": self.texture_size[0] * 4,
                "rows_per_image": self.texture_size[1],
            },
            self.texture_size,
        )
        bind_group = self.device.create_bind_group(
            layout=self.text_layout,
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": self.scene_sampler},
            ],
        )
        text_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_texture_view,
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        text_pass.set_pipeline(self.text_pipeline)
        text_pass.set_bind_group(0, bind_group, [], 0, 99)
        text_pass.draw(3)
        text_pass.end()
        self.frame_bind_groups.append(bind_group)

    def paintWebGPU(self) -> None:
        self.frame_bind_groups = []
        if self.scene_texture_size != self.texture_size:
            self._create_scene_texture()
            self._create_bind_groups()
        lay = self._layout()
        encoder = self.device.create_command_encoder()
        scene_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ]
        )
        self._draw_terrain(scene_pass, lay)
        self._draw_chrome(scene_pass, lay)
        scene_pass.end()
        self._draw_text(encoder, lay)

        self._write_crt_uniform()
        final_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ]
        )
        final_pass.set_pipeline(self.crt_pipeline)
        final_pass.set_bind_group(0, self.crt_bind_group, [], 0, 99)
        final_pass.draw(3)
        final_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.update()

    def timerEvent(self, event) -> None:
        self.advance()
        self.update()

    def advance(self) -> None:
        self.tick += 1
        if not self.paused:
            self.flight_t += 0.045 * self.speed
        if self.tick - self.last_log_tick > LOG_INTERVAL:
            self.log_line(LOG_MESSAGES[self.log_msg_index % len(LOG_MESSAGES)])
            self.log_msg_index += 1

    def log_line(self, msg: str) -> None:
        stamp = f"{self.tick // 3600:02d}:{(self.tick // 60) % 60:02d}"
        self.log.append((stamp, f"> {msg}"))
        self.last_log_tick = self.tick

    def button_at(self, mx: float, my: float) -> str | None:
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                return btn["id"]
        return None

    def in_centre(self, mx: float, my: float) -> bool:
        cx, cy, cw, ch = self._layout()["centre"]
        return cx <= mx <= cx + cw and cy <= my <= cy + ch

    def rotate_view(self, dx: float, dy: float) -> None:
        self.spin_y_face = max(-60.0, min(60.0, self.spin_y_face + 0.4 * dx))
        self.spin_x_face = max(-40.0, min(80.0, self.spin_x_face + 0.4 * dy))

    def reset_view(self) -> None:
        self.spin_x_face = 0.0
        self.spin_y_face = 0.0
        self.log_line("VIEW RESET")

    def press(self, bid: str) -> None:
        self.flash[bid] = self.tick + 8
        if bid == "hold":
            self.paused = not self.paused
            self.log_line("FLIGHT HOLD ENGAGED" if self.paused else "FLIGHT RESUMED")
        elif bid == "vel+":
            self.speed = min(4.0, self.speed + 0.25)
            self.log_line(f"VELOCITY SET {self.speed:4.2f}")
        elif bid == "vel-":
            self.speed = max(0.25, self.speed - 0.25)
            self.log_line(f"VELOCITY SET {self.speed:4.2f}")
        elif bid == "phos":
            self.amber = not self.amber
            self.log_line(f"PHOSPHOR {'AMBER' if self.amber else 'GREEN'}")
        elif bid == "scan":
            self.effects_on = not self.effects_on
            self.log_line(f"SCAN FX {'ENABLED' if self.effects_on else 'DISABLED'}")
        elif bid == "purge":
            self.log.clear()
            self.log_msg_index = 0
            self.log_line("LOG PURGED")

    def _scene_pos(self, event) -> tuple[float, float]:
        pos = event.position()
        return pos.x() * self.ratio, pos.y() * self.ratio

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            mx, my = self._scene_pos(event)
            bid = self.button_at(mx, my)
            if bid is not None:
                self.press(bid)
                self.update()
            elif self.in_centre(mx, my):
                self.rotating = True
                self.last_x, self.last_y = mx, my
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        mx, my = self._scene_pos(event)
        if self.rotating and event.buttons() == Qt.MouseButton.LeftButton:
            self.rotate_view(mx - self.last_x, my - self.last_y)
            self.last_x, self.last_y = mx, my
            self.update()
            return
        hover = self.button_at(mx, my)
        if hover != self.hover_id:
            self.hover_id = hover
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotating = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Space:
            self.press("hold")
        elif key == Qt.Key.Key_Up:
            self.press("vel+")
        elif key == Qt.Key.Key_Down:
            self.press("vel-")
        elif key == Qt.Key.Key_P:
            self.press("phos")
        elif key == Qt.Key.Key_S:
            self.press("scan")
        elif key == Qt.Key.Key_R:
            self.reset_view()
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
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
    window = SciFiWebGPU()
    window.resize(1280, 800)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
