#!/usr/bin/env -S uv run --script
"""WebGPU line-list version of the ResetLine demo."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from blade_field import animate_blades, create_blade_field, expand_line_list
from ncca.ngl import Mat4, PerspMode, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """The restarted strips expanded into one independent line list."""

    def __init__(self, rows: int = 120, cols: int = 120, seed: int = 7) -> None:
        super().__init__()
        self.setWindowTitle("ResetLine (WebGPU line list)")
        self.msaa_sample_count = 4
        self.field = create_blade_field(rows=rows, cols=cols, seed=seed)
        self.vertices = self.field.vertices.copy()
        self.phase = 0.0
        self.animate = False

        self.model_position = Vec3()
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.view = look_at(Vec3(0, 10, 10), Vec3(), Vec3(0, 1, 0))
        self.project = perspective(
            45.0,
            self.width() / max(self.height(), 1),
            0.001,
            100.0,
            PerspMode.WebGPU,
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(10)
        self.animation_timer.timeout.connect(self._tick)
        self.animation_timer.start()

    def _create_pipeline(self) -> None:
        shader = self.device.create_shader_module(
            code=(Path(__file__).parent / "ResetLineShader.wgsl").read_text()
        )
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )
        self.pipeline = self.device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
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
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_scene(self) -> None:
        self.line_vertices = expand_line_list(self.vertices, self.field.ranges)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.line_vertices,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.uniform_buffer = self.device.create_buffer(
            size=64,
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
                        "size": 64,
                    },
                }
            ],
        )

    def _global_transform(self) -> Mat4:
        transform = Mat4().rotate_y(self.spin_y_face) @ Mat4().rotate_x(
            self.spin_x_face
        )
        transform[3, 0] = self.model_position.x
        transform[3, 1] = self.model_position.y
        transform[3, 2] = self.model_position.z
        return transform

    def paintWebGPU(self) -> None:
        mvp = self.project @ self.view @ self._global_transform()
        self.device.queue.write_buffer(
            self.uniform_buffer, 0, mvp.to_numpy().astype(np.float32).tobytes()
        )
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.7, 0.7, 0.7, 1.0),
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
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(len(self.line_vertices))
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.001, 100.0, PerspMode.WebGPU
        )
        self.update()

    def _tick(self) -> None:
        if self.animate:
            self.vertices = animate_blades(self.vertices, self.field.ranges, self.phase)
            self.phase += 0.05
            self.line_vertices = expand_line_list(self.vertices, self.field.ranges)
            self.device.queue.write_buffer(
                self.vertex_buffer,
                0,
                np.ascontiguousarray(self.line_vertices).tobytes(),
            )
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()

    def mousePressEvent(self, event) -> None:
        self.last_x = event.position().x()
        self.last_y = event.position().y()
        self.rotate = event.button() == Qt.LeftButton
        self.translate = event.button() == Qt.RightButton

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()
        if self.rotate and event.buttons() == Qt.LeftButton:
            self.spin_y_face += int(0.5 * (x - self.last_x))
            self.spin_x_face += int(0.5 * (y - self.last_y))
        elif self.translate and event.buttons() == Qt.RightButton:
            self.model_position.x += 0.01 * (x - self.last_x)
            self.model_position.y -= 0.01 * (y - self.last_y)
        self.last_x, self.last_y = x, y
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        self.model_position.z += 0.1 if event.angleDelta().y() > 0 else -0.1
        self.update()

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoketest", nargs="?", const=300, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene(args.rows, args.cols, args.seed)
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
