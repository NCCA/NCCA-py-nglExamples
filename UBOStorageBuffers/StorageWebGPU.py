#!/usr/bin/env -S uv run --script
"""
Uniform buffers vs. a runtime-sized storage buffer (WebGPU).

Same scene as main.py (diffuse teapot + checker grid), rebuilt on wgpu-py,
picking up where the OpenGL half's README leaves off: GL 4.1 has no SSBOs,
so the "many lights" half of this story can only be told here.

    - SceneBlock (VP, light) and MaterialBlock (albedo, specularColour,
      shininess) are ordinary uniform buffers, laid out with the SAME numpy
      dtypes as the OpenGL demo (layouts.py) -- WGSL's default
      uniform-address-space layout follows the same vec3-alignment rule as
      std140 (specularColour pushed to 16, shininess packing in at 28).
    - The point lights are a `var<storage, read> lights: array<PointLight>`
      -- a buffer whose element count is read back in the shader with
      `arrayLength(&lights)` rather than being a fixed max-size array
      baked into the shader at compile time, which is the one thing a
      uniform buffer cannot do. Growing/shrinking the light count with
      +/- recreates the storage buffer (and the bind group that points at
      it); the shader itself never changes.

    X            toggle MaterialBlock CPU layout: std140-correct / naive
                 (corrupts the teapot's specular highlight -- same trap,
                 same mechanism, as the OpenGL demo)
    +/-          grow/shrink the storage-buffer point-light count (8..64)
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from layouts import (
    MATERIAL_BLOCK_STD140_DTYPE,
    SCENE_BLOCK_DTYPE,
    naive_bytes_padded_to_std140,
)
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

MIN_LIGHTS = 8
MAX_LIGHTS = 64
LIGHT_STEP = 4

# PointLight { position: vec3<f32>, intensity: f32, colour: vec3<f32>, _pad: f32 }
# -- WGSL's storage-address-space layout for this struct (vec3 base
# alignment 16, so `intensity` fits in the gap at offset 12, `colour`
# starts fresh at 16, `_pad` fills the block out to a 16-byte multiple).
POINT_LIGHT_DTYPE = np.dtype(
    {
        "names": ["position", "intensity", "colour", "_pad"],
        "formats": [(np.float32, 3), np.float32, (np.float32, 3), np.float32],
        "offsets": [0, 12, 16, 28],
        "itemsize": 32,
    }
)

# ObjectUniforms for the teapot pipeline: M, normalMatrix (mat4 -- mat3
# uniform-buffer padding is fiddly, so this demo uses the mat4 workaround
# the design spec recommends), viewPos.
TEAPOT_OBJECT_DTYPE = np.dtype(
    {
        "names": ["M", "normalMatrix", "viewPos"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), (np.float32, 4)],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)

GRID_OBJECT_DTYPE = np.dtype(
    {"names": ["M"], "formats": [(np.float32, (4, 4))], "offsets": [0], "itemsize": 64}
)

GRID_CELLS = 8
GRID_SIZE = 10.0


def build_checker_grid(cells: int, size: float) -> np.ndarray:
    """Interleaved x,y,z,r,g,b for a flat, checker-coloured cells x cells
    grid of quads (2 triangles each) centred on the origin at y = -1."""
    half = size * 0.5
    step = size / cells
    verts = []
    colour_a = (0.85, 0.35, 0.2)
    colour_b = (0.2, 0.35, 0.85)
    for i in range(cells):
        for j in range(cells):
            x0, z0 = -half + i * step, -half + j * step
            x1, z1 = x0 + step, z0 + step
            colour = colour_a if (i + j) % 2 == 0 else colour_b
            corners = [(x0, -1.0, z0), (x1, -1.0, z0), (x1, -1.0, z1), (x0, -1.0, z1)]
            for idx in (0, 1, 2, 0, 2, 3):
                verts.append((*corners[idx], *colour))
    return np.array(verts, dtype=np.float32).reshape(-1)


def build_point_lights(count: int) -> np.ndarray:
    """`count` point lights arranged in a ring above the scene."""
    data = np.zeros(count, dtype=POINT_LIGHT_DTYPE)
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        data["position"][i] = (
            4.5 * math.cos(angle),
            2.5 + 1.5 * math.sin(2.0 * angle),
            4.5 * math.sin(angle),
        )
        hue = i / count
        data["colour"][i] = (
            0.5 + 0.5 * math.sin(2.0 * math.pi * hue),
            0.5 + 0.5 * math.sin(2.0 * math.pi * hue + 2.0),
            0.5 + 0.5 * math.sin(2.0 * math.pi * hue + 4.0),
        )
        data["intensity"][i] = 1.2
    return data


class WebGPUScene(WebGPUWidget):
    """Shared-SceneBlock teapot + grid, plus a runtime-sized storage-buffer
    light array the OpenGL half of this demo cannot express."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UBO vs Storage Buffer (WebGPU)")
        self.msaa_sample_count = 4

        self.correct_material_layout = True
        self.light_count = 16

        # --- camera / mouse state (same conventions as the other WebGPU demos) ---
        self.mouse_global_tx = Mat4()
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.eye = Vec3(0.0, 3.0, 9.0)
        self.view = look_at(self.eye, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 350.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_geometry()
        self._create_pipelines()
        self._create_static_buffers()
        self._rebuild_lights()
        self._create_render_buffer()
        self.update()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_geometry(self) -> None:
        teapot_data = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_vertex_buffer = self.device.create_buffer_with_data(
            data=teapot_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.teapot_count = teapot_data.size // 8

        grid_data = build_checker_grid(GRID_CELLS, GRID_SIZE)
        self.grid_vertex_buffer = self.device.create_buffer_with_data(
            data=grid_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.grid_count = grid_data.size // 6

    def _create_pipelines(self) -> None:
        teapot_src = (Path(__file__).parent / "SceneShader.wgsl").read_text()
        grid_src = (Path(__file__).parent / "GridShader.wgsl").read_text()
        teapot_module = self.device.create_shader_module(code=teapot_src)
        grid_module = self.device.create_shader_module(code=grid_src)

        target = {"format": wgpu.TextureFormat.rgba8unorm}
        depth_stencil = {
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        }
        multisample = {
            "count": self.msaa_sample_count,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        }

        self.teapot_pipeline = self.device.create_render_pipeline(
            label="teapot_pipeline",
            layout="auto",
            vertex={
                "module": teapot_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
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
                "module": teapot_module,
                "entry_point": "fragment_main",
                "targets": [target],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil=depth_stencil,
            multisample=multisample,
        )

        self.grid_pipeline = self.device.create_render_pipeline(
            label="grid_pipeline",
            layout="auto",
            vertex={
                "module": grid_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": grid_module,
                "entry_point": "fragment_main",
                "targets": [target],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil=depth_stencil,
            multisample=multisample,
        )

    def _create_static_buffers(self) -> None:
        self.scene_buffer = self.device.create_buffer(
            size=SCENE_BLOCK_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.material_buffer = self.device.create_buffer(
            size=MATERIAL_BLOCK_STD140_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.teapot_object_buffer = self.device.create_buffer(
            size=TEAPOT_OBJECT_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.grid_object_buffer = self.device.create_buffer(
            size=GRID_OBJECT_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.grid_bind_group = self.device.create_bind_group(
            layout=self.grid_pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.scene_buffer,
                        "offset": 0,
                        "size": self.scene_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.grid_object_buffer,
                        "offset": 0,
                        "size": self.grid_object_buffer.size,
                    },
                },
            ],
        )

    def _rebuild_lights(self) -> None:
        """Recreate the storage buffer (and the teapot bind group that
        references it) at the new size. This is the WebGPU-side cost of a
        runtime-sized array: growing it is a buffer + bind-group rebuild,
        not a sub-range write, because a bind group binds a fixed
        (buffer, size) pair."""
        data = build_point_lights(self.light_count)
        self.lights_buffer = self.device.create_buffer_with_data(
            data=data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.teapot_bind_group = self.device.create_bind_group(
            layout=self.teapot_pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.scene_buffer,
                        "offset": 0,
                        "size": self.scene_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.material_buffer,
                        "offset": 0,
                        "size": self.material_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.lights_buffer,
                        "offset": 0,
                        "size": self.lights_buffer.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.teapot_object_buffer,
                        "offset": 0,
                        "size": self.teapot_object_buffer.size,
                    },
                },
            ],
        )

    # ------------------------------------------------------------------
    # per-frame updates
    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _update_scene_block(self, vp: Mat4) -> None:
        data = np.zeros((), dtype=SCENE_BLOCK_DTYPE)
        data["VP"] = vp.to_numpy()
        data["lightPos"] = (4.0, 5.0, 4.0, 1.0)
        data["lightColour"] = (1.0, 0.95, 0.85, 1.0)
        self.device.queue.write_buffer(self.scene_buffer, 0, data.tobytes())

    def _update_material_block(self) -> None:
        albedo = (0.8, 0.65, 0.2)
        specular_colour = (1.0, 0.6, 0.3)
        shininess = 64.0
        if self.correct_material_layout:
            data = np.zeros((), dtype=MATERIAL_BLOCK_STD140_DTYPE)
            data["albedo"] = albedo
            data["specularColour"] = specular_colour
            data["shininess"] = shininess
            payload = data.tobytes()
        else:
            payload = naive_bytes_padded_to_std140(albedo, specular_colour, shininess)
        self.device.queue.write_buffer(self.material_buffer, 0, payload)

    def paintWebGPU(self) -> None:
        global_tx = self.scene_global_tx()
        vp = self.project @ self.view @ global_tx
        self._update_scene_block(vp)
        self._update_material_block()

        teapot_uniforms = np.zeros((), dtype=TEAPOT_OBJECT_DTYPE)
        model = Mat4()
        mv = self.view @ global_tx @ model
        teapot_uniforms["M"] = model.to_numpy()
        teapot_uniforms["normalMatrix"] = mv.inverse().transposed().to_numpy()
        teapot_uniforms["viewPos"] = (*self.eye, 1.0)
        self.device.queue.write_buffer(
            self.teapot_object_buffer, 0, teapot_uniforms.tobytes()
        )

        grid_uniforms = np.zeros((), dtype=GRID_OBJECT_DTYPE)
        grid_uniforms["M"] = Mat4().to_numpy()
        self.device.queue.write_buffer(
            self.grid_object_buffer, 0, grid_uniforms.tobytes()
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.12, 0.12, 0.14, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )

        render_pass.set_pipeline(self.grid_pipeline)
        render_pass.set_bind_group(0, self.grid_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.grid_vertex_buffer)
        render_pass.draw(self.grid_count)

        render_pass.set_pipeline(self.teapot_pipeline)
        render_pass.set_bind_group(0, self.teapot_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.teapot_vertex_buffer)
        render_pass.draw(self.teapot_count)

        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()
        self._draw_hud()

    def _draw_hud(self) -> None:
        white = QColor(255, 255, 255)
        bad = QColor(255, 100, 100)
        self.render_text(
            10,
            20,
            "SceneBlock (uniform) read by both the teapot and grid pipelines",
            14,
            "Arial",
            white,
        )
        layout_name = (
            "CORRECT std140 (specularColour@16, shininess@28)"
            if self.correct_material_layout
            else "NAIVE packed (specularColour@12, shininess@24) -- WRONG, "
            "colour scrambled + shininess reads 0"
        )
        self.render_text(
            10,
            42,
            f"[X] MaterialBlock layout: {layout_name}",
            14,
            "Arial",
            white if self.correct_material_layout else bad,
        )
        self.render_text(
            10,
            64,
            f"[+/-] point lights (storage buffer, runtime-sized): {self.light_count}",
            14,
            "Arial",
            white,
        )

    def resizeWebGPU(self, width, height) -> None:
        self.project = perspective(45.0, width / height, 0.1, 350.0, PerspMode.WebGPU)
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_X:
            self.correct_material_layout = not self.correct_material_layout
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            if self.light_count < MAX_LIGHTS:
                self.light_count = min(MAX_LIGHTS, self.light_count + LIGHT_STEP)
                self._rebuild_lights()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            if self.light_count > MIN_LIGHTS:
                self.light_count = max(MIN_LIGHTS, self.light_count - LIGHT_STEP)
                self._rebuild_lights()
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
