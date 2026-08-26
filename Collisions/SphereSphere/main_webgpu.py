#!/usr/bin/env -S uv run --script
"""SphereSphere (WebGPU): 4 fixed spheres, 2 moving and bouncing off 2
static ones -- independent WebGPU port of Collisions/SphereSphere/main.py,
same object count/positions/radii/colours/collision rules.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Vec3,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_sphere_collide

_SPHERES = [
    {
        "pos": Vec3(-10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(-7.0, 0.0, 0.0),
        "dir": Vec3(0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (1.0, 0.0, 0.0),
    },
    {
        "pos": Vec3(7.0, 0.0, 0.0),
        "dir": Vec3(-0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (0.0, 0.0, 1.0),
    },
]
_DRAW_POOL_SIZE = 4


def _v3(v: Vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.spheres = [dict(s) for s in _SPHERES]
        self.view = look_at(Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.05, 350.0, PerspMode.WebGPU)
        self.mouse_global_tx = Mat4()
        self.model_position = Vec3(0, 0, 0)
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.orig_x = 0
        self.orig_y = 0
        self.orig_x_pos = 0
        self.orig_y_pos = 0
        self.device = get_default_device()
        self._create_pipeline()
        self._create_geometry()
        self._create_draw_buffer_pool()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(20)

    def _create_pipeline(self) -> None:
        shader_path = Path(__file__).parent / "SphereSphereShader.wgsl"
        shader_module = self.device.create_shader_module(code=shader_path.read_text())
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
        vertex_buffer_layout = {
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
                {
                    "format": wgpu.VertexFormat.float32x2,
                    "offset": 6 * 4,
                    "shader_location": 2,
                },
            ],
        }
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_geometry(self) -> None:
        # A real generated sphere, not a baked-mesh substitute -- see
        # BoundingBox/main_webgpu.py's _create_geometry() for the full
        # rationale. Precision 40 matches main.py's
        # `Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)`.
        data = PrimData.sphere(1.0, 40)
        vertex_count = data.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = vertex_count

    def _create_draw_buffer_pool(self) -> None:
        # Only 4 draws/frame here, well below any real aliasing risk, but
        # the pool convention (see Spotlight/main_webgpu.py's comment) is
        # kept for consistency: each draw gets its own uniform
        # buffer/bind-group slot, indexed by a counter reset every frame,
        # rather than rewriting one shared buffer between draws.
        uniform_size = (16 + 16 + 4) * 4  # mvp mat4 + normal_matrix mat4 + colour vec4
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for _ in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": buf, "offset": 0, "size": uniform_size},
                    }
                ],
            )
            self.draw_uniform_buffers.append(buf)
            self.draw_bind_groups.append(bind_group)

    def _on_tick(self) -> None:
        self.spheres[2]["pos"] = self.spheres[2]["pos"] + self.spheres[2]["dir"]
        self.spheres[3]["pos"] = self.spheres[3]["pos"] + self.spheres[3]["dir"]
        self._check_collisions()
        self.update()

    def _check_collisions(self) -> None:
        s2, s3, s0, s1 = (
            self.spheres[2],
            self.spheres[3],
            self.spheres[0],
            self.spheres[1],
        )
        if sphere_sphere_collide(
            _v3(s2["pos"]), s2["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
            s3["dir"] = s3["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s0["pos"]), s0["radius"], _v3(s2["pos"]), s2["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s1["pos"]), s1["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s3["dir"] = s3["dir"] * -1.0

    def _draw_sphere(
        self, render_pass, draw_index: int, s: dict, global_tx: Mat4
    ) -> None:
        m = Mat4().translate(s["pos"].x, s["pos"].y, s["pos"].z) @ Mat4().scale(
            s["radius"], s["radius"], s["radius"]
        )
        m = global_tx @ m
        mv = self.view @ m
        mvp = self.project @ mv
        normal_matrix = m.inverse().transposed()
        data = np.zeros(16 + 16 + 4, dtype=np.float32)
        data[0:16] = mvp.to_numpy().flatten()
        data[16:32] = normal_matrix.to_numpy().flatten()
        data[32:36] = np.array([*s["colour"], 1.0], dtype=np.float32)
        self.device.queue.write_buffer(
            self.draw_uniform_buffers[draw_index], 0, data.tobytes()
        )
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)

    def paintWebGPU(self) -> None:
        if not hasattr(self, "device"):
            return
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
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
        for draw_index, s in enumerate(self.spheres):
            self._draw_sphere(render_pass, draw_index, s, self.mouse_global_tx)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, w: int, h: int) -> None:
        self.project = perspective(
            45.0, float(w) / max(h, 1), 0.05, 350.0, PerspMode.WebGPU
        )
        self.update()

    def closeEvent(self, event) -> None:
        # Stop the animation timer before the base class tears down the
        # wgpu surface/device -- otherwise a queued timer tick can fire a
        # GPU call (paintWebGPU via update()) after teardown and crash.
        # Same fix as main.py's closeEvent, mirrored from
        # ShadedGrid/main_webgpu.py and Spotlight/main_webgpu.py.
        self.animation_timer.stop()
        super().closeEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.orig_x, self.orig_y = position.x(), position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.orig_x_pos, self.orig_y_pos = position.x(), position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.orig_x
            diff_y = position.y() - self.orig_y
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.orig_x, self.orig_y = position.x(), position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.orig_x_pos)
            diff_y = int(position.y() - self.orig_y_pos)
            self.orig_x_pos, self.orig_y_pos = position.x(), position.y()
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
            self.model_position.z += 0.5
        elif delta < 0:
            self.model_position.z -= 0.5
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position = Vec3(0, 0, 0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.setWindowTitle("SphereSphere (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
