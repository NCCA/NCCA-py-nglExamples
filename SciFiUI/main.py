#!/usr/bin/env -S uv run --script
"""
SciFiUI -- a retro sci-fi CRT terminal built entirely with py-ngl and OpenGL.

The central window shows a wireframe terrain fly-over drawn as stacked
ridge lines in the style of the pulsar plot on Joy Division's
"Unknown Pleasures" cover (hidden lines are removed with an opaque
"skirt" polygon under each ridge, painter's-algorithm back to front).

Around it is a Nostromo / MU-TH-UR 6000 style interface:

- clickable buttons on the left (hold, velocity, phosphor colour,
  scanline toggle, log purge) with hover highlighting
- a scrolling system log on the right with a typewriter reveal
- header / footer status bars with a blinking cursor

Everything is rendered in monochrome into an FBO, then a second
full-screen pass applies the CRT look: phosphor tint (green or amber),
barrel distortion, scanlines, a rolling bar, noise, flicker and
vignette. See shaders/CRTFragment.glsl.

All rendering lives in SciFiScene, which knows nothing about Qt --
MainWindow is a thin QOpenGLWindow shell that forwards events to it.

Run with:  uv run SciFiUI/main.py
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
import OpenGL.GL as gl
from ncca.ngl import Mat4, Vec3, logger, look_at, ortho, perspective
from ncca.ngl.opengl import ShaderLib, Text, VAOFactory, VAOType, VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

UI_SHADER = "SciFiUI"
CRT_SHADER = "SciFiCRT"

# terrain resolution -- rows are ridge lines receding into the screen
TERRAIN_ROWS = 42
TERRAIN_COLS = 130
TERRAIN_HALF_WIDTH = 7.0  # x extent of each ridge line
TERRAIN_NEAR_Z = -3.5  # z of the closest ridge
TERRAIN_ROW_STEP = 0.8  # z spacing between ridges
TERRAIN_FLOOR_Y = -2.0  # bottom of the opaque skirt polygons

# log messages, cycled forever (ASCII only -- the font atlas covers ' '..'~')
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

LOG_INTERVAL = 55  # ticks between new log lines


class ValueNoise:
    """
    Small tileable 2D value-noise generator (numpy, vectorised).

    Used to synthesise the terrain height field: smooth bilinear
    interpolation of a random lattice with a smoothstep fade, summed
    over a couple of octaves by the caller.
    """

    def __init__(self, size: int = 256, seed: int = 1979) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.grid = rng.random((size, size)).astype(np.float32)

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Sample noise at (x, z); arrays broadcast, returns values in [0,1]."""
        xi = np.floor(x).astype(np.int64)
        zi = np.floor(z).astype(np.int64)
        xf = (x - xi).astype(np.float32)
        zf = (z - zi).astype(np.float32)
        xi %= self.size
        zi %= self.size
        xj = (xi + 1) % self.size
        zj = (zi + 1) % self.size
        # smoothstep fade
        u = xf * xf * (3.0 - 2.0 * xf)
        v = zf * zf * (3.0 - 2.0 * zf)
        g = self.grid
        a = g[zi, xi]
        b = g[zi, xj]
        c = g[zj, xi]
        d = g[zj, xj]
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


class Terrain:
    """
    The Unknown-Pleasures style ridge line fly-over.

    Each frame the height field is re-sampled from value noise at a z
    offset that advances with flight time, the vertex buffers are
    re-uploaded, and the rows are drawn BACK TO FRONT. For each row an
    opaque black 'skirt' (triangle strip from the ridge down to the
    floor) is drawn first, then the bright ridge line on top -- nearer
    rows therefore hide the lines behind them without any depth buffer.
    """

    def __init__(self) -> None:
        self.noise = ValueNoise()
        self.xs = np.linspace(
            -TERRAIN_HALF_WIDTH, TERRAIN_HALF_WIDTH, TERRAIN_COLS, dtype=np.float32
        )
        # amplitude envelope: quiet at the edges, active in the middle,
        # exactly like the pulsar plot
        self.envelope = np.exp(-(np.abs(self.xs / (TERRAIN_HALF_WIDTH * 0.55)) ** 3.0))
        self.line_verts = np.zeros((TERRAIN_ROWS, TERRAIN_COLS, 3), dtype=np.float32)
        self.skirt_verts = np.zeros(
            (TERRAIN_ROWS, TERRAIN_COLS * 2, 3), dtype=np.float32
        )
        # x and z never change, only heights do
        for row in range(TERRAIN_ROWS):
            z = TERRAIN_NEAR_Z - row * TERRAIN_ROW_STEP
            self.line_verts[row, :, 0] = self.xs
            self.line_verts[row, :, 2] = z
            self.skirt_verts[row, 0::2, 0] = self.xs
            self.skirt_verts[row, 1::2, 0] = self.xs
            self.skirt_verts[row, 1::2, 1] = TERRAIN_FLOOR_Y
            self.skirt_verts[row, :, 2] = z
        self.line_vao = VAOFactory.create_vao(VAOType.MULTI_BUFFER, gl.GL_LINE_STRIP)
        self.skirt_vao = VAOFactory.create_vao(
            VAOType.MULTI_BUFFER, gl.GL_TRIANGLE_STRIP
        )

    def update(self, flight_t: float) -> None:
        """Re-sample heights for the current flight position."""
        row_idx = np.arange(TERRAIN_ROWS, dtype=np.float32)[:, None]
        # noise-space coordinates: x across the ridge, z advancing with flight
        nx = (self.xs[None, :] + TERRAIN_HALF_WIDTH) * 0.55
        nz = (row_idx * TERRAIN_ROW_STEP + flight_t) * 0.35
        n1 = self.noise.sample(nx, nz)
        n2 = self.noise.sample(nx * 2.7 + 11.3, nz * 2.3 + 7.7)
        # sharp occasional pulses for the classic spiky look
        n3 = self.noise.sample(nx * 1.3 + 51.0, nz * 1.1 + 23.0)
        heights = self.envelope[None, :] * (1.5 * n1 + 0.5 * n2 + 3.0 * np.power(n3, 6))
        self.line_verts[:, :, 1] = heights
        self.skirt_verts[:, 0::2, 1] = heights

    def draw(self) -> None:
        """Upload the current vertex data and draw all rows back to front."""
        line_flat = self.line_verts.reshape(-1)
        skirt_flat = self.skirt_verts.reshape(-1)
        skirt_row = TERRAIN_COLS * 2
        # upload once per frame into each VAO, then interleave the draws
        with self.skirt_vao as vao:
            vao.set_data(VertexData(data=skirt_flat, size=skirt_flat.nbytes), index=0)
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
        with self.line_vao as vao:
            vao.set_data(VertexData(data=line_flat, size=line_flat.nbytes), index=0)
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
        for row in range(TERRAIN_ROWS - 1, -1, -1):
            # opaque skirt first so it hides the rows behind ...
            with self.skirt_vao:
                ShaderLib.set_uniform("Colour", 0.0, 0.0, 0.0, 1.0)
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, row * skirt_row, skirt_row)
            # ... then the bright ridge line, faded slightly with distance
            with self.line_vao:
                fade = 0.35 + 0.65 * (1.0 - row / TERRAIN_ROWS)
                ShaderLib.set_uniform("Colour", fade, fade, fade, 1.0)
                gl.glDrawArrays(gl.GL_LINE_STRIP, row * TERRAIN_COLS, TERRAIN_COLS)


class UIBatch:
    """
    Tiny immediate-mode style helper for the 2D chrome.

    Rects and lines are accumulated in pixel coordinates each frame,
    uploaded into one dynamic VAO and drawn as ranges with a flat
    colour uniform per range. Simple and more than fast enough for a
    handful of panels and buttons.
    """

    def __init__(self) -> None:
        self.vao = VAOFactory.create_vao(VAOType.MULTI_BUFFER, gl.GL_TRIANGLES)
        self.verts: list[float] = []
        self.ranges: list[tuple] = []  # (mode, first, count, colour)

    def begin(self) -> None:
        self.verts = []
        self.ranges = []

    def _push(self, mode: int, pts: list[tuple], colour: tuple) -> None:
        first = len(self.verts) // 3
        for x, y in pts:
            self.verts.extend((float(x), float(y), 0.0))
        self.ranges.append((mode, first, len(pts), colour))

    def rect(self, x, y, w, h, colour) -> None:
        """Filled rectangle (two triangles), pixel coords, y down."""
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]
        self._push(gl.GL_TRIANGLES, pts, colour)

    def outline(self, x, y, w, h, colour) -> None:
        """Rectangle outline as a line loop."""
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        self._push(gl.GL_LINE_LOOP, pts, colour)

    def line(self, x0, y0, x1, y1, colour) -> None:
        self._push(gl.GL_LINES, [(x0, y0), (x1, y1)], colour)

    def draw(self) -> None:
        if not self.verts:
            return
        data = np.array(self.verts, dtype=np.float32)
        with self.vao as vao:
            vao.set_data(VertexData(data=data, size=data.nbytes), index=0)
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)
            for mode, first, count, colour in self.ranges:
                ShaderLib.set_uniform("Colour", *colour)
                gl.glDrawArrays(mode, first, count)


class SciFiScene:
    """
    The whole CRT terminal, independent of any windowing toolkit.

    The owner supplies a current OpenGL context, calls initialize()
    once, resize() when the framebuffer size changes, advance() per
    tick and render(target_fbo) per frame. Mouse positions passed to
    button_at() are in framebuffer pixels, y down.
    """

    def __init__(self) -> None:
        self.width = 1280
        self.height = 800

        # simulation state
        self.tick = 0
        self.flight_t = 0.0
        self.speed = 1.0
        self.paused = False
        self.amber = False
        self.effects_on = True

        # view rotation (arcball style, driven by dragging in the terrain panel)
        self.spin_x_face = 0.0  # pitch, degrees
        self.spin_y_face = 0.0  # yaw, degrees

        # UI state
        self.buttons: list[dict] = []  # rebuilt every frame with pixel rects
        self.hover_id: str | None = None
        self.flash: dict[str, int] = {}  # button id -> tick the flash ends
        self.log: deque = deque(maxlen=64)
        self.log_msg_index = 0
        self.last_log_tick = 0

        # GL objects, created in initialize()
        self.terrain: Terrain | None = None
        self.ui: UIBatch | None = None
        self.fbo_id = None
        self.fbo_texture = None
        self.fbo_size = (0, 0)
        self.screen_vao = None

    # ------------------------------------------------------------------ setup

    def initialize(self) -> None:
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # painter's algorithm everywhere -- no depth buffer needed
        gl.glDisable(gl.GL_DEPTH_TEST)

        shader_dir = Path(__file__).parent / "shaders"
        ok = ShaderLib.load_shader(
            UI_SHADER,
            str(shader_dir / "UIVertex.glsl"),
            str(shader_dir / "UIFragment.glsl"),
        )
        ok &= ShaderLib.load_shader(
            CRT_SHADER,
            str(shader_dir / "CRTVertex.glsl"),
            str(shader_dir / "CRTFragment.glsl"),
        )
        if not ok:
            raise RuntimeError("shader compilation failed")

        font = Path(__file__).parent.parent / "font" / "Arial.ttf"
        Text.add_font("log", str(font), 16)
        Text.add_font("ui", str(font), 20)
        Text.add_font("header", str(font), 26)

        self.terrain = Terrain()
        self.ui = UIBatch()
        # empty VAO for the full-screen CRT triangle (core profile needs one bound)
        self.screen_vao = gl.glGenVertexArrays(1)

        self.view = look_at(Vec3(0.0, 4.5, 9.0), Vec3(0.0, 1.0, -14.0), Vec3(0, 1, 0))

        self.log_line("INTERFACE 2037 READY FOR INQUIRY")
        self.log_line("TERRAIN SCAN COMMENCED")

    def resize(self, w: int, h: int) -> None:
        self.width, self.height = int(w), int(h)
        Text.set_screen_size(self.width, self.height)

    def _ensure_fbo(self) -> None:
        """(Re)create the offscreen render target at the current size."""
        w, h = self.width, self.height
        if self.fbo_size == (w, h) or w == 0 or h == 0:
            return
        if self.fbo_id is not None:
            gl.glDeleteFramebuffers(1, [self.fbo_id])
            gl.glDeleteTextures(1, [self.fbo_texture])
        self.fbo_texture = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo_texture)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.fbo_texture,
            0,
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.fbo_size = (w, h)

    # ------------------------------------------------------------------ layout

    def _layout(self) -> dict:
        """Compute all panel rectangles in framebuffer pixels (y down)."""
        w, h = self.width, self.height
        s = h / 800.0  # scale relative to design height
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
            "centre": (
                2 * m + left_w,
                top,
                w - 4 * m - left_w - right_w,
                bottom - top,
            ),
        }

    def _button_defs(self) -> list[tuple[str, str]]:
        """(id, current label) for each button, top to bottom."""
        return [
            ("hold", "RESUME" if self.paused else "HOLD"),
            ("vel+", "VEL +"),
            ("vel-", "VEL -"),
            ("phos", "PHOS: AMBER" if self.amber else "PHOS: GREEN"),
            ("scan", f"SCAN FX {'ON' if self.effects_on else 'OFF'}"),
            ("purge", "PURGE LOG"),
        ]

    # ------------------------------------------------------------------ drawing

    def render(self, target_fbo: int = 0) -> None:
        self._ensure_fbo()
        lay = self._layout()
        w, h = self.width, self.height

        # ---------- pass one: whole interface, monochrome, into the FBO
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self._draw_terrain(lay)
        self._draw_chrome(lay)
        self._draw_text(lay)

        # ---------- pass two: CRT post process to the screen
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target_fbo)
        gl.glViewport(0, 0, w, h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        ShaderLib.use(CRT_SHADER)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo_texture)
        ShaderLib.set_uniform("screenTex", 0)
        ShaderLib.set_uniform("iTime", self.tick * 0.016)
        ShaderLib.set_uniform("iResolution", float(w), float(h))
        if self.amber:
            ShaderLib.set_uniform("phosphor", 1.0, 0.72, 0.22)
        else:
            ShaderLib.set_uniform("phosphor", 0.35, 1.0, 0.45)
        ShaderLib.set_uniform("effectsOn", 1 if self.effects_on else 0)
        gl.glBindVertexArray(self.screen_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

    def _draw_terrain(self, lay: dict) -> None:
        """Perspective ridge-line pass, clipped to the centre panel."""
        cx, cy, cw, ch = lay["centre"]
        pad = lay["m"]
        vx, vy = cx + pad, cy + pad
        vw, vh = cw - 2 * pad, ch - 2 * pad
        # GL viewport origin is bottom-left; our layout is top-down
        gl.glViewport(vx, self.height - (vy + vh), vw, vh)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(vx, self.height - (vy + vh), vw, vh)

        ShaderLib.use(UI_SHADER)
        project = perspective(40.0, float(vw) / float(vh), 0.1, 200.0)
        # arcball-style rotation about the middle of the terrain strip so it
        # turns in place rather than swinging around the world origin
        pivot_z = TERRAIN_NEAR_Z - (TERRAIN_ROWS / 2.0) * TERRAIN_ROW_STEP
        rot = Mat4.rotate_y(self.spin_y_face) @ Mat4.rotate_x(self.spin_x_face)
        model = (
            Mat4.translate(0.0, 0.0, pivot_z) @ rot @ Mat4.translate(0.0, 0.0, -pivot_z)
        )
        ShaderLib.set_uniform("MVP", project @ self.view @ model)
        self.terrain.update(self.flight_t)
        self.terrain.draw()

        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, self.width, self.height)

    def _draw_chrome(self, lay: dict) -> None:
        """All the 2D panels, frames and buttons."""
        s = lay["s"]
        ui = self.ui
        ui.begin()

        dim = (0.10, 0.10, 0.10, 1.0)  # panel fill
        mid = (0.45, 0.45, 0.45, 1.0)  # frame lines
        hot = (1.0, 1.0, 1.0, 1.0)  # highlight

        # NOTE: the centre panel gets no fill -- the terrain has already
        # been drawn there and the chrome pass runs after it
        for key in ("header", "footer", "left", "right"):
            x, y, pw, ph = lay[key]
            ui.rect(x, y, pw, ph, dim)
            ui.outline(x, y, pw, ph, mid)
        ui.outline(*lay["centre"], mid)

        # corner brackets on the terrain frame, targeting-reticle style
        cx, cy, cw, ch = lay["centre"]
        b = int(26 * s)
        for px, py, dx, dy in (
            (cx, cy, 1, 1),
            (cx + cw, cy, -1, 1),
            (cx, cy + ch, 1, -1),
            (cx + cw, cy + ch, -1, -1),
        ):
            ui.line(px, py, px + dx * b, py, hot)
            ui.line(px, py, px, py + dy * b, hot)

        # buttons
        self.buttons = []
        lx, ly, lw, lh = lay["left"]
        pad = int(12 * s)
        bh = int(52 * s)
        y = ly + pad
        for bid, label in self._button_defs():
            rect = (lx + pad, y, lw - 2 * pad, bh)
            self.buttons.append({"id": bid, "label": label, "rect": rect})
            flashing = self.tick < self.flash.get(bid, -1)
            hovered = self.hover_id == bid
            if flashing:
                fill = (0.85, 0.85, 0.85, 1.0)
            elif hovered:
                fill = (0.30, 0.30, 0.30, 1.0)
            else:
                fill = (0.16, 0.16, 0.16, 1.0)
            ui.rect(*rect, fill)
            ui.outline(*rect, hot if (hovered or flashing) else mid)
            y += bh + pad

        # log panel title divider
        rx, ry, rw, rh = lay["right"]
        ui.line(rx, ry + int(34 * s), rx + rw, ry + int(34 * s), mid)

        ShaderLib.use(UI_SHADER)
        # pixel-space orthographic projection, y down to match the layout
        ShaderLib.set_uniform(
            "MVP", ortho(0.0, float(self.width), float(self.height), 0.0, -1.0, 1.0)
        )
        ui.draw()

    def _draw_text(self, lay: dict) -> None:
        """Header, footer, button labels and the scrolling log."""
        s = lay["s"]
        bright = Vec3(0.95, 0.95, 0.95)
        faint = Vec3(0.55, 0.55, 0.55)

        # header
        hx, hy, hw, hh = lay["header"]
        Text.render_text(
            "header",
            hx + int(14 * s),
            hy + hh - int(10 * s),
            "WEYLAND-YUTANI CORP // SEMIOTIC STANDARD",
            bright,
        )
        clock = f"MU-TH-UR 6000  T+{self.tick // 60:06d}"
        Text.render_text(
            "ui", hx + hw - int(320 * s), hy + hh - int(12 * s), clock, faint
        )

        # centre panel title and status line
        cx, cy, cw, ch = lay["centre"]
        Text.render_text(
            "ui",
            cx + int(16 * s),
            cy + int(30 * s),
            "TERRAIN SCAN :: LV-426 APPROACH",
            bright,
        )
        vel = 0.0 if self.paused else self.speed
        Text.render_text(
            "log",
            cx + int(16 * s),
            cy + ch - int(12 * s),
            f"VEL {vel:4.2f} :: ALT 3.7KM :: MODE {'HOLD' if self.paused else 'FLY'}"
            f" :: VIEW {self.spin_y_face:+04.0f}/{self.spin_x_face:+04.0f}",
            faint,
        )

        # button labels
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            flashing = self.tick < self.flash.get(btn["id"], -1)
            colour = Vec3(0.05, 0.05, 0.05) if flashing else bright
            Text.render_text(
                "ui", bx + int(14 * s), by + bh // 2 + int(8 * s), btn["label"], colour
            )
        lx, ly, lw, lh = lay["left"]
        Text.render_text(
            "log", lx + int(12 * s), ly + lh - int(12 * s), "PANEL 04 // INPUT", faint
        )

        # scrolling log, newest at the bottom, typewriter reveal on the
        # newest line, clipped to the panel with a scissor rect
        rx, ry, rw, rh = lay["right"]
        Text.render_text("ui", rx + int(12 * s), ry + int(24 * s), "SYSTEM LOG", bright)
        line_h = int(24 * s)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(rx, self.height - (ry + rh), rw, rh - int(40 * s))
        y = ry + rh - int(14 * s)
        progress = min(1.0, (self.tick - self.last_log_tick) / 20.0)
        # crude but effective: clamp lines to the panel width
        max_chars = int((rw - 24 * s) / (8.5 * s))
        for i, (stamp, msg) in enumerate(reversed(self.log)):
            text = f"{stamp} {msg}"[:max_chars]
            if i == 0:  # newest line types itself out
                text = text[: max(1, int(len(text) * progress))]
            Text.render_text(
                "log", rx + int(12 * s), y, text, bright if i == 0 else faint
            )
            y -= line_h
            if y < ry + int(60 * s):
                break
        gl.glDisable(gl.GL_SCISSOR_TEST)

        # footer with blinking cursor
        fx, fy, fw, fh = lay["footer"]
        cursor = "_" if (self.tick // 30) % 2 == 0 else " "
        Text.render_text(
            "ui",
            fx + int(14 * s),
            fy + fh - int(8 * s),
            f"INTERFACE 2037 READY FOR INQUIRY {cursor}",
            bright,
        )

    # ------------------------------------------------------------------ state

    def advance(self) -> None:
        """One simulation tick (~16ms)."""
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
        """Hit test a position in framebuffer pixels against the buttons."""
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                return btn["id"]
        return None

    def in_centre(self, mx: float, my: float) -> bool:
        """True if a framebuffer-pixel position is inside the terrain panel."""
        cx, cy, cw, ch = self._layout()["centre"]
        return cx <= mx <= cx + cw and cy <= my <= cy + ch

    def rotate_view(self, dx: float, dy: float) -> None:
        """
        Apply a mouse drag (pixel deltas) to the view rotation.

        Angles are clamped: the painter's-algorithm hidden-line trick
        relies on nearer rows staying nearer, so we keep the view well
        away from edge-on or underneath.
        """
        self.spin_y_face = max(-60.0, min(60.0, self.spin_y_face + 0.4 * dx))
        self.spin_x_face = max(-40.0, min(80.0, self.spin_x_face + 0.4 * dy))

    def reset_view(self) -> None:
        self.spin_x_face = 0.0
        self.spin_y_face = 0.0
        self.log_line("VIEW RESET")

    def press(self, bid: str) -> None:
        """Dispatch a button click and log it, MU-TH-UR style."""
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


class MainWindow(QOpenGLWindow):
    """Thin Qt shell around SciFiScene: context, timer and input events."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setTitle("SEMIOTIC STANDARD // MU-TH-UR 6000")
        self.scene = SciFiScene()
        # drag-to-rotate state (LMB inside the terrain panel)
        self.rotating = False
        self.last_x = 0.0
        self.last_y = 0.0

    def initializeGL(self) -> None:
        self.makeCurrent()
        self.scene.initialize()
        self.startTimer(16)

    def resizeGL(self, w: int, h: int) -> None:
        ratio = self.devicePixelRatio()
        self.scene.resize(int(w * ratio), int(h * ratio))

    def paintGL(self) -> None:
        self.makeCurrent()
        self.scene.render(self.defaultFramebufferObject())

    def timerEvent(self, event) -> None:
        self.scene.advance()
        self.update()

    def _scene_pos(self, event) -> tuple[float, float]:
        ratio = self.devicePixelRatio()
        pos = event.position()
        return pos.x() * ratio, pos.y() * ratio

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            mx, my = self._scene_pos(event)
            bid = self.scene.button_at(mx, my)
            if bid is not None:
                self.scene.press(bid)
                self.update()
            elif self.scene.in_centre(mx, my):
                # start rotating the terrain view
                self.rotating = True
                self.last_x, self.last_y = mx, my

    def mouseMoveEvent(self, event) -> None:
        mx, my = self._scene_pos(event)
        if self.rotating and event.buttons() == Qt.LeftButton:
            self.scene.rotate_view(mx - self.last_x, my - self.last_y)
            self.last_x, self.last_y = mx, my
            self.update()
            return
        hover = self.scene.button_at(mx, my)
        if hover != self.scene.hover_id:
            self.scene.hover_id = hover
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotating = False

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.scene.press("hold")
        elif key == Qt.Key_Up:
            self.scene.press("vel+")
        elif key == Qt.Key_Down:
            self.scene.press("vel-")
        elif key == Qt.Key_P:
            self.scene.press("phos")
        elif key == Qt.Key_S:
            self.scene.press("scan")
        elif key == Qt.Key_R:
            self.scene.reset_view()
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by the Qt event loop."""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
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

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1280, 800)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
