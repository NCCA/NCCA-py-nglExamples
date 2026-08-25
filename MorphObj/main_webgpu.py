#!/usr/bin/env -S uv run --script
"""WebGPU three-pose OBJ morph demo."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from morph_mesh import adjust_weight, advance_punch, load_morph_mesh
from ncca.ngl import Mat4, PerspMode, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

MODEL_DIR = Path(__file__).parent / "models"
UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("model_view", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("light_position", np.float32, 4),
        ("weights", np.float32, 4),
    ]
)


class WebGPUScene(WebGPUWidget):
    """The same packed pose data blended in a WGSL vertex shader."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MorphObj (WebGPU)")
        self.msaa_sample_count = 4
        self.weight_one = 0.0
        self.weight_two = 0.0
        self.left_direction = 1
        self.right_direction = 1
        self.animation_enabled = True
        self.mesh_data = load_morph_mesh(
            MODEL_DIR / "BrucePose1.obj",
            MODEL_DIR / "BrucePose2.obj",
            MODEL_DIR / "BrucePose3.obj",
        )

        self.model_position = Vec3()
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.view = look_at(Vec3(0, 10, 40), Vec3(0, 10, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0,
            self.width() / max(self.height(), 1),
            0.05,
            350.0,
            PerspMode.WebGPU,
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()
        self.left_timer = QTimer(self)
        self.left_timer.setInterval(4)
        self.left_timer.timeout.connect(self._advance_left)
        self.right_timer = QTimer(self)
        self.right_timer.setInterval(4)
        self.right_timer.timeout.connect(self._advance_right)

    def _create_pipeline(self) -> None:
        shader = self.device.create_shader_module(
            code=(Path(__file__).parent / "MorphShader.wgsl").read_text()
        )
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )
        attributes = [
            {
                "format": "float32x3",
                "offset": location * 3 * 4,
                "shader_location": location,
            }
            for location in range(6)
        ]
        self.pipeline = self.device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 18 * 4,
                        "step_mode": "vertex",
                        "attributes": attributes,
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
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

    def _create_scene(self) -> None:
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.mesh_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniforms["light_position"] = (2.0, 20.0, 2.0, 1.0)
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
                        "size": self.uniforms.nbytes,
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
        model_view = self.view @ self._global_transform()
        self.uniforms["mvp"] = (self.project @ model_view).to_numpy()
        self.uniforms["model_view"] = model_view.to_numpy()
        self.uniforms["normal_matrix"] = model_view.inverse().transposed().to_numpy()
        self.uniforms["weights"] = (self.weight_one, self.weight_two, 0.0, 0.0)
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
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
        render_pass.draw(len(self.mesh_data))
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()
        white = QColor(255, 255, 255)
        self.render_text(
            10,
            25,
            f"Q/W pose one {self.weight_one:.2f}   A/S pose two {self.weight_two:.2f}",
            14,
            "Arial",
            white,
        )
        self.render_text(10, 50, "Z/X punch   Space pause", 14, "Arial", white)

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.05, 350.0, PerspMode.WebGPU
        )
        self.update()

    def _advance_left(self) -> None:
        if not self.animation_enabled:
            return
        self.weight_one, self.left_direction, active = advance_punch(
            self.weight_one, self.left_direction
        )
        if not active:
            self.left_timer.stop()
        self.update()

    def _advance_right(self) -> None:
        if not self.animation_enabled:
            return
        self.weight_two, self.right_direction, active = advance_punch(
            self.weight_two, self.right_direction
        )
        if not active:
            self.right_timer.stop()
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Q:
            self.weight_one = adjust_weight(self.weight_one, -0.1)
        elif key == Qt.Key_W:
            self.weight_one = adjust_weight(self.weight_one, 0.1)
        elif key == Qt.Key_A:
            self.weight_two = adjust_weight(self.weight_two, -0.1)
        elif key == Qt.Key_S:
            self.weight_two = adjust_weight(self.weight_two, 0.1)
        elif key == Qt.Key_Z and not self.left_timer.isActive():
            self.weight_one = 0.0
            self.left_direction = 1
            self.left_timer.start()
        elif key == Qt.Key_X and not self.right_timer.isActive():
            self.weight_two = 0.0
            self.right_direction = 1
            self.right_timer.start()
        elif key == Qt.Key_Space:
            self.animation_enabled = not self.animation_enabled
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
        self.left_timer.stop()
        self.right_timer.stop()
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
    parser.add_argument(
        "--smoketest", nargs="?", const=300, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
