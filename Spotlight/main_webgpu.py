#!/usr/bin/env -S uv run --script
"""
Spotlight: 4 animated cone-attenuation spotlights over a grid of teapots (WebGPU).

Same scene as the OpenGL version (main.py) -- same light colours, orbit
centres/radii/phases and cone angles, all computed in world space -- but
built on the WebGPU stack. There's no runtime primitive generator here, so
the ground plane is a small hand-built quad rather than the OpenGL
version's Prims.TRIANGLE_PLANE.

This draws 26 objects (25 teapots + 1 ground plane) inside a single render
pass, each with its own uniform buffer pulled from a pool pre-allocated at
init and indexed by a per-frame draw counter -- see the pool comment below
for why a single shared buffer doesn't work here.

Controls:
    a : toggle light animation
    LMB rotate  RMB pan  wheel zoom  Space reset
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

_NUM_LIGHTS = 4
_LIGHT_COLOURS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.3, 0.4, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
]
_LIGHT_CENTRES = [(-4.0, -4.0), (4.0, -4.0), (-4.0, 4.0), (4.0, 4.0)]
_LIGHT_RADII = [2.5, 3.0, 2.0, 3.5]
_LIGHT_HEIGHT = 6.0
_DRAW_POOL_SIZE = 26  # 25 teapots + 1 ground plane

_LIGHT_DTYPE = np.dtype(
    [
        ("position", np.float32, 4),
        ("direction", np.float32, 4),
        ("diffuse", np.float32, 4),
        ("specular", np.float32, 4),
        ("params", np.float32, 4),
        ("attenuation", np.float32, 4),
    ]
)


def quad_floor(size: float) -> np.ndarray:
    """
    A flat, upward-facing quad, interleaved x,y,z,nx,ny,nz,u,v float32.

    Parameters
    ----------
        size : float
            edge length of the (square) quad
    """
    h = size / 2.0
    verts = [
        (-h, 0.0, -h, 0.0, 1.0, 0.0, 0.0, 0.0),
        (h, 0.0, -h, 0.0, 1.0, 0.0, 1.0, 0.0),
        (h, 0.0, h, 0.0, 1.0, 0.0, 1.0, 1.0),
        (-h, 0.0, -h, 0.0, 1.0, 0.0, 0.0, 0.0),
        (h, 0.0, h, 0.0, 1.0, 0.0, 1.0, 1.0),
        (-h, 0.0, h, 0.0, 1.0, 0.0, 0.0, 1.0),
    ]
    return np.array(verts, dtype=np.float32).flatten()


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.setWindowTitle("Spotlight (WebGPU)")
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.original_x = 0
        self.original_y = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.model_position = Vec3(0, 0, 0)
        self.animate = True
        self.time = 0.0
        self.grid_positions = [
            (float(x), float(z)) for x in range(-6, 7, 3) for z in range(-6, 7, 3)
        ]
        self.eye = Vec3(0, 8, 16)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.5, 150.0, PerspMode.WebGPU)

        self.device = get_default_device()
        self._create_pipeline()
        self._create_geometry()
        self._create_draw_buffer_pool()
        self._create_light_buffer()
        self._update_lights()
        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "SpotlightShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        vertex_layout = {
            "array_stride": 8 * 4,
            "attributes": [
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
            ],
        }
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_geometry(self) -> None:
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_buffer = self.device.create_buffer_with_data(
            data=teapot.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.teapot_count = teapot.size // 8

        ground = quad_floor(30.0)
        self.ground_buffer = self.device.create_buffer_with_data(
            data=ground.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.ground_count = ground.size // 8

    def _create_draw_buffer_pool(self) -> None:
        """Pre-allocate one uniform buffer + bind group per draw call.

        This demo does 26 draws (25 teapots + the ground plane) inside a
        single render pass. WebGPU's queue-timeline ordering only
        guarantees a submitted command buffer sees a resource's state as
        of immediately before that submit -- rewriting one shared uniform
        buffer 26 times before a single submit() would leave every draw
        seeing only the last write, collapsing the whole grid onto one
        teapot's transform (the exact bug fixed in MatrixStack/
        main_webgpu.py and LookAtDemos/main_webgpu.py). A pool sidesteps
        that: each draw gets its own buffer/bind-group slot, indexed by a
        counter that resets at the start of every frame.
        """
        # 208-byte Uniforms struct: 3 mat4x4 @ 64 bytes each + 1 vec4 @ 16 bytes.
        self._uniform_size = 4 * 4 * 4 * 3 + 16
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for index in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=self._uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"spotlight_draw_{index}",
            )
            self.draw_uniform_buffers.append(buf)

    def _create_light_buffer(self) -> None:
        self.light_data = np.zeros(_NUM_LIGHTS, dtype=_LIGHT_DTYPE)
        for i in range(_NUM_LIGHTS):
            r, g, b, a = _LIGHT_COLOURS[i]
            self.light_data[i]["direction"] = (0.0, -1.0, 0.0, 0.0)
            self.light_data[i]["diffuse"] = (r, g, b, a)
            self.light_data[i]["specular"] = (r, g, b, a)
            self.light_data[i]["params"] = (
                math.cos(math.radians(25.0)),
                math.cos(math.radians(15.0)),
                8.0,
                0.0,
            )
            self.light_data[i]["attenuation"] = (1.0, 0.02, 0.005, 0.0)
        self.light_buffer = self.device.create_buffer(
            size=self.light_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # The light buffer is written once per frame (in _update_lights)
        # and read by every draw's bind group -- unlike the per-draw
        # transforms, all 26 draws in a frame genuinely want the same
        # light state, so a single shared buffer is correct here, not
        # an aliasing bug.
        for buf in self.draw_uniform_buffers:
            self.draw_bind_groups.append(
                self.device.create_bind_group(
                    layout=self.bind_group_layout,
                    entries=[
                        {
                            "binding": 0,
                            "resource": {
                                "buffer": buf,
                                "offset": 0,
                                "size": self._uniform_size,
                            },
                        },
                        {
                            "binding": 1,
                            "resource": {
                                "buffer": self.light_buffer,
                                "offset": 0,
                                "size": self.light_data.nbytes,
                            },
                        },
                    ],
                )
            )

    def _update_lights(self) -> None:
        for i in range(_NUM_LIGHTS):
            cx, cz = _LIGHT_CENTRES[i]
            radius = _LIGHT_RADII[i]
            phase = self.time + i * 1.7
            x = cx + math.cos(phase) * radius
            z = cz + math.sin(phase) * radius
            self.light_data[i]["position"] = (x, _LIGHT_HEIGHT, z, 1.0)
        self.device.queue.write_buffer(self.light_buffer, 0, self.light_data.tobytes())

    def _on_tick(self) -> None:
        if self.animate:
            self.time += 0.03
            self._update_lights()
            self.update()

    def _global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _draw_object(
        self, render_pass, draw_index: int, model: Mat4, vertex_buffer, count: int
    ) -> None:
        mvp = self.project @ self.view @ model
        normal_matrix = model.inverse().transposed()
        data = np.zeros(self._uniform_size // 4, dtype=np.float32)
        data[0:16] = model.to_numpy().flatten()
        data[16:32] = mvp.to_numpy().flatten()
        data[32:48] = normal_matrix.to_numpy().flatten()
        data[48:51] = (self.eye.x, self.eye.y, self.eye.z)
        self.device.queue.write_buffer(
            self.draw_uniform_buffers[draw_index], 0, data.tobytes()
        )
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.draw(count)

    def paintWebGPU(self) -> None:
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "clear_value": (0.15, 0.15, 0.15, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.pipeline)

        global_tx = self._global_tx()
        draw_index = 0
        self._draw_object(
            render_pass, draw_index, global_tx, self.ground_buffer, self.ground_count
        )
        draw_index += 1
        for x, z in self.grid_positions:
            model = global_tx @ Mat4().translate(x, 0.5, z)
            self._draw_object(
                render_pass, draw_index, model, self.teapot_buffer, self.teapot_count
            )
            draw_index += 1

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.5, 150.0, PerspMode.WebGPU)
        self.update()

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x = position.x()
            self.original_y = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        if self.rotate and event.buttons() == Qt.LeftButton:
            diff_x = position.x() - self.original_x
            diff_y = position.y() - self.original_y
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x = position.x()
            self.original_y = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            diff_x = position.x() - self.original_x_pos
            diff_y = position.y() - self.original_y_pos
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += 0.01 * diff_x
            self.model_position.y -= 0.01 * diff_y
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.model_position.z += 0.1
        elif delta < 0:
            self.model_position.z -= 0.1
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position = Vec3(0, 0, 0)
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
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
