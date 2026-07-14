#!/usr/bin/env -S uv run --script
"""
GPU instancing vs a Python draw-call loop (WebGPU).

The WebGPU counterpart to main.py -- same cube field, same
``instance_layout.golden_spiral`` placement, same HUD frame-time readout.
The draw strategies differ slightly from the OpenGL version because WebGPU
has no per-draw uniform-push equivalent that's cheap to call N times:

    instanced -- one render_pass.draw(36, n) call. The per-instance vertex
                 buffer (step_mode "instance", locations 3/4) supplies
                 offset/scale/colour per cube via @builtin(instance_index).
    naive     -- a Python for loop of n render_pass.draw(36, 1, 0, i) calls,
                 each with first_instance=i. Same pipeline, same buffers --
                 wgpu-py's native backend passes first_instance straight
                 through to wgpuRenderPassEncoderDraw, and the WebGPU spec
                 defines it as also offsetting instance-step-mode vertex
                 fetches, so each call still reads instance i's own data.
                 The only thing that changes is one draw call per cube
                 instead of one draw call for the whole field -- exactly
                 the same lesson as the OpenGL naive loop, minus the
                 uniform-push overhead (which WebGPU pipelines don't have,
                 so the naive/instanced gap here is purely draw-call count).

Controls:
    I    toggle instanced / naive draw
    +/-  double / halve the instance count N (clamped 1..65536)
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import wgpu
from instance_layout import cube, golden_spiral
from ncca.ngl import Mat4, PerspMode, Vec3, logger, look_at, perspective
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device

DEFAULT_N = 4096
MIN_N = 1
MAX_N = 65536
FIELD_RADIUS = 8.0
FRAME_HISTORY = 30

# std140-style uniform block: MVP and the normal matrix, both full mat4s
# (mat3 has WGSL padding gotchas -- see the design spec).
UNIFORM_DTYPE = np.dtype(
    [
        ("MVP", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
    ]
)


class WebGPUScene(WebGPUWidget):
    """A field of cubes drawn either instanced or via a naive draw-call loop."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instancing: one draw call vs N (WebGPU)")
        self.msaa_sample_count = 4

        # --- demo state driven by the keyboard ---
        self.instanced = True
        self.n = DEFAULT_N
        self.field_rotation = 0.0
        self.frame_times: deque = deque(maxlen=FRAME_HISTORY)

        # --- camera / mouse state (same conventions as the GL demos) ---
        self.model_position = Vec3(0, 0, -12)
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

        self.view = look_at(
            Vec3(0.0, 6.0, 16.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(
            45.0, self.width() / self.height(), 0.05, 350.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()
        self.startTimer(16)
        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "InstanceShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.pipeline = self.device.create_render_pipeline(
            label="instance_pipeline",
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        # cube vertex data: one record per vertex
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {
                                "format": "float32x3",
                                "offset": 12,
                                "shader_location": 1,
                            },
                        ],
                    },
                    {
                        # per-instance offset+scale / colour: one record per
                        # instance -- the WebGPU equivalent of
                        # glVertexAttribDivisor(loc, 1).
                        "array_stride": 8 * 4,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x4", "offset": 0, "shader_location": 3},
                            {
                                "format": "float32x4",
                                "offset": 16,
                                "shader_location": 4,
                            },
                        ],
                    },
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _create_scene(self) -> None:
        cube_data = cube(1.0)
        self.cube_buffer = self.device.create_buffer_with_data(
            data=cube_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.cube_vertex_count = cube_data.size // 6

        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = self.device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
        )

        self._rebuild_instance_buffer()

    def _rebuild_instance_buffer(self) -> None:
        """(Re)generate the per-instance layout for the current N."""
        self.instance_data = golden_spiral(self.n, radius=FIELD_RADIUS)
        self.instance_buffer = self.device.create_buffer_with_data(
            data=self.instance_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.frame_times.clear()

    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face + self.field_rotation)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        start = time.perf_counter()
        global_tx = self.scene_global_tx()
        mv = self.view @ global_tx
        self.uniforms["MVP"] = (self.project @ mv).to_numpy()
        # a full mat4 inverse-transpose in the uniform block, not a mat3 --
        # WGSL mat3 uniforms carry padding gotchas (see the design spec).
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
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
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.cube_buffer)
        render_pass.set_vertex_buffer(1, self.instance_buffer)

        if self.instanced:
            render_pass.draw(self.cube_vertex_count, self.n)
        else:
            # Teaching point: identical pipeline and buffers, but one
            # draw() call per cube instead of one call for the whole
            # field -- first_instance=i selects instance i's record from
            # the same per-instance buffer the instanced path reads.
            for i in range(self.n):
                render_pass.draw(self.cube_vertex_count, 1, 0, i)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()
        self._draw_hud()
        self.frame_times.append(time.perf_counter() - start)

    def _draw_hud(self) -> None:
        white = QColor(255, 255, 255)
        avg_ms = (
            1000.0 * sum(self.frame_times) / len(self.frame_times)
            if self.frame_times
            else 0.0
        )
        mode = "INSTANCED (1 draw call)" if self.instanced else "NAIVE (N draw calls)"
        self.render_text(10, 20, f"[I] mode: {mode}", 14, "Arial", white)
        self.render_text(
            10,
            45,
            f"[+/-] N = {self.n}   avg frame time = {avg_ms:.2f} ms",
            14,
            "Arial",
            white,
        )

    def resizeWebGPU(self, width, height) -> None:
        self.project = perspective(45.0, width / height, 0.05, 350.0, PerspMode.WebGPU)
        self.update()

    def timerEvent(self, event) -> None:
        self.field_rotation += 0.3
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _set_n(self, n: int) -> None:
        self.n = max(MIN_N, min(MAX_N, n))
        self._rebuild_instance_buffer()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_I:
            self.instanced = not self.instanced
            self.frame_times.clear()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._set_n(self.n * 2)
        elif key == Qt.Key_Minus:
            self._set_n(self.n // 2)
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, -12)
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
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
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
