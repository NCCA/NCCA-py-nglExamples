"""WebGPU instanced-particle version of the abstract octree demo."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from octree import ParticleSystem, bounds_lines
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
    ]
)


class WebGPUScene(WebGPUWidget):
    """CPU octree simulation with a single WebGPU instanced sphere draw."""

    def __init__(self, grid_size: int = 10, seed: int = 7) -> None:
        super().__init__()
        self.setWindowTitle("OctreeAbstract (WebGPU)")
        self.msaa_sample_count = 4
        self.grid_size = grid_size
        self.seed = seed
        self.system = ParticleSystem.grid(grid_size, seed)
        self.animate = True

        self.model_position = Vec3()
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.view = look_at(Vec3(0, 0, 35), Vec3(), Vec3(0, 1, 0))
        self.project = perspective(
            45.0,
            self.width() / max(self.height(), 1),
            0.05,
            350.0,
            PerspMode.WebGPU,
        )

        self.device = get_default_device()
        self._create_pipelines()
        self._create_scene()
        self._create_render_buffer()
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setInterval(50)
        self.simulation_timer.timeout.connect(self._tick)
        self.simulation_timer.start()

    def _create_pipelines(self) -> None:
        shader = self.device.create_shader_module(
            code=(Path(__file__).parent / "OctreeShader.wgsl").read_text()
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
        common = {
            "layout": layout,
            "fragment": {
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            "depth_stencil": {
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            "multisample": {"count": self.msaa_sample_count},
        }
        self.particle_pipeline = self.device.create_render_pipeline(
            **common,
            vertex={
                "module": shader,
                "entry_point": "particle_vertex",
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
                        ],
                    },
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x4", "offset": 0, "shader_location": 2},
                            {
                                "format": "float32x4",
                                "offset": 16,
                                "shader_location": 3,
                            },
                        ],
                    },
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
        self.line_pipeline = self.device.create_render_pipeline(
            **common,
            vertex={
                "module": shader,
                "entry_point": "line_vertex",
                "buffers": [
                    {
                        "array_stride": 3 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    }
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
        )

    def _instance_data(self) -> np.ndarray:
        data = np.ones((self.system.count, 8), dtype=np.float32)
        data[:, :3] = self.system.positions
        data[:, 3] = self.system.radii
        data[:, 4:7] = self.system.colours
        return data

    def _create_scene(self) -> None:
        sphere = PrimData.sphere(1.0, 12)
        self.sphere_vertex_count = len(sphere)
        self.sphere_buffer = self.device.create_buffer_with_data(
            data=sphere, usage=wgpu.BufferUsage.VERTEX
        )
        self.instance_data = self._instance_data()
        self.instance_buffer = self.device.create_buffer_with_data(
            data=self.instance_data,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.lines = bounds_lines()
        self.line_buffer = self.device.create_buffer_with_data(
            data=self.lines, usage=wgpu.BufferUsage.VERTEX
        )
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
        self.uniforms["normal_matrix"] = model_view.inverse().transposed().to_numpy()
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
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_pipeline(self.particle_pipeline)
        render_pass.set_vertex_buffer(0, self.sphere_buffer)
        render_pass.set_vertex_buffer(1, self.instance_buffer)
        render_pass.draw(self.sphere_vertex_count, self.system.count)
        render_pass.set_pipeline(self.line_pipeline)
        render_pass.set_vertex_buffer(0, self.line_buffer)
        render_pass.draw(len(self.lines))
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()
        self.render_text(
            10,
            25,
            f"particles {self.system.count}   Space reset   A pause",
            14,
            "Arial",
            QColor(255, 255, 255),
        )

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.05, 350.0, PerspMode.WebGPU
        )
        self.update()

    def _upload_instances(self) -> None:
        self.instance_data = self._instance_data()
        self.device.queue.write_buffer(
            self.instance_buffer, 0, self.instance_data.tobytes()
        )

    def _tick(self) -> None:
        if self.animate:
            self.system.step()
            self._upload_instances()
        self.update()

    def _reset_particles(self) -> None:
        self.system = ParticleSystem.grid(self.grid_size, self.seed)
        self._upload_instances()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self._reset_particles()
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
        self.simulation_timer.stop()
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
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoketest", nargs="?", const=300, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene(args.grid, args.seed)
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
