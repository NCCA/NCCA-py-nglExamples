#!/usr/bin/env -S uv run --script
"""Game-style held-key spaceship control (WebGPU).

The WebGPU counterpart to main.py -- same bitmask-indexed motion table, same
record/playback via `KeyRecorder`, same two-timer split between input
sampling and redraw rate. `game_controls.py` is shared unmodified between
both backends, so this file's only genuinely new code is drawing the ship:
`ship_mesh.py` replicates `Obj.create_vao()`'s interleave logic (GL-only, no
numpy accessor) to build a vertex buffer directly, and
`GameKeyControlShader.wgsl` is a hand-rolled flat/unlit shader standing in
for the OpenGL side's `nglColourShader`.

As with the OpenGL version there is no mouse camera control at all -- the
camera is a single fixed `look_at`, matching the C++ source, which never
wires up a mouse handler.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from game_controls import GameControls, KeyRecorder, move_ship, ship_transform
from ncca.ngl import PerspMode, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QFileDialog
from ship_mesh import load_ship_vertex_data
from wgpu.utils import get_default_device

MODELS_DIR = Path(__file__).parent / "models"

# std140-style uniform block: MVP and the flat draw colour. Both fields are
# already 16-byte aligned (64 + 16), so there's no padding gotcha here.
UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("colour", np.float32, 4),
    ]
)


class WebGPUScene(WebGPUWidget):
    """Held-key spaceship demo widget.

    Deliberately plain `WebGPUWidget` -- no mouse handlers of any kind. The
    C++ source has none, so neither does this port, matching Task 2's GL
    sibling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Game style key control in Qt (WebGPU)")
        self.msaa_sample_count = 4

        self.view = look_at(
            Vec3(0.0, 0.0, 80.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(
            45.0, self.width() / self.height(), 0.05, 350.0, PerspMode.WebGPU
        )

        self.ship_pos = Vec3(0.0, 0.0, 0.0)
        self.ship_rotation = 0.0

        self.keys_pressed = 0
        self.recording = False
        self.playback_active = False
        self.current_playback_frame = 0
        self.key_recorder = KeyRecorder()

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()

        # Two independent timers, matching the source's timerEvent split:
        # one drives ship movement/recording/playback at 15ms, the other
        # only triggers a redraw at 30ms.
        self.ship_timer = QTimer(self)
        self.ship_timer.timeout.connect(self._on_tick)
        self.ship_timer.start(15)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.timeout.connect(self.update)
        self.redraw_timer.start(30)

        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "GameKeyControlShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.pipeline = self.device.create_render_pipeline(
            label="game_key_control_pipeline",
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {
                                "format": "float32x3",
                                "offset": 12,
                                "shader_location": 1,
                            },
                            {
                                "format": "float32x2",
                                "offset": 24,
                                "shader_location": 2,
                            },
                        ],
                    }
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
        data, vertex_count = load_ship_vertex_data(MODELS_DIR / "SpaceShip.obj")
        self.ship_vertex_count = vertex_count
        self.ship_buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )

        # Single ship, single draw call per frame -- no per-draw
        # uniform-buffer pool needed here.
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = self.device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="game_key_control_uniforms",
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

        # The colour never changes, so it's written once here rather than
        # every frame in paintWebGPU().
        self.uniforms["colour"] = (1.0, 1.0, 0.0, 1.0)
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if self.playback_active:
            if self.current_playback_frame <= 0:
                self.ship_pos = self.key_recorder.get_start_position()
            if self.current_playback_frame < self.key_recorder.size():
                self.keys_pressed = self.key_recorder[self.current_playback_frame]
                self.current_playback_frame += 1
            else:
                self.playback_active = False
                self.current_playback_frame = 0
        elif self.recording:
            self.key_recorder.add_frame(self.keys_pressed)
        self.ship_pos, self.ship_rotation = move_ship(
            self.ship_pos, self.ship_rotation, self.keys_pressed
        )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        mvp = (
            self.project @ self.view @ ship_transform(self.ship_pos, self.ship_rotation)
        )
        self.uniforms["mvp"] = mvp.to_numpy()
        self.device.queue.write_buffer(
            self.uniform_buffer, 0, self.uniforms["mvp"].tobytes()
        )

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
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
        render_pass.set_vertex_buffer(0, self.ship_buffer)
        render_pass.draw(self.ship_vertex_count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        if self.recording:
            self.render_text(10, 18, "Recording", 14, "Arial", QColor(255, 0, 0))
        if self.playback_active:
            self.render_text(
                10,
                18,
                f"Playback doing frame {self.current_playback_frame}",
                14,
                "Arial",
                QColor(255, 255, 0),
            )

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.05, 350.0, PerspMode.WebGPU
        )
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.keys_pressed |= int(GameControls.UP)
        elif key == Qt.Key_Down:
            self.keys_pressed |= int(GameControls.DOWN)
        elif key == Qt.Key_Left:
            self.keys_pressed |= int(GameControls.LEFT)
        elif key == Qt.Key_Right:
            self.keys_pressed |= int(GameControls.RIGHT)
        elif key == Qt.Key_R:
            self.keys_pressed |= int(GameControls.ROTATE)
        elif key == Qt.Key_Space:
            self.key_recorder.set_start_position(self.ship_pos)
            self.recording = not self.recording
        elif key == Qt.Key_P:
            self.playback_active = not self.playback_active
        elif key == Qt.Key_S:
            path, _ = QFileDialog.getSaveFileName(None, "Save Keypresses", ".", "*.kp")
            if path:
                self.key_recorder.save(path)
        elif key == Qt.Key_L:
            path, _ = QFileDialog.getOpenFileName(None, "Load Keypresses", ".", "*.kp")
            if path:
                self.key_recorder.load(path)
        elif key == Qt.Key_Escape:
            self.close()

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.keys_pressed &= ~int(GameControls.UP)
        elif key == Qt.Key_Down:
            self.keys_pressed &= ~int(GameControls.DOWN)
        elif key == Qt.Key_Left:
            self.keys_pressed &= ~int(GameControls.LEFT)
        elif key == Qt.Key_Right:
            self.keys_pressed &= ~int(GameControls.RIGHT)
        elif key == Qt.Key_R:
            self.keys_pressed &= ~int(GameControls.ROTATE)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self.ship_timer.stop()
        self.redraw_timer.stop()
        super().closeEvent(event)


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
