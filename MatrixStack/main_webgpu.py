#!/usr/bin/env -S uv run --script
"""
MatrixStack: an OpenGL-style push/pop matrix stack (WebGPU).

Same push/pop matrix-stack logic as the OpenGL version (main.py) — the stack
itself (matrix_stack.py) is pure CPU-side maths shared unchanged between
both entry points. All of the transforms are packed into one uniform buffer
before the instanced troll and sphere draws are encoded.

Controls:
    I/O  increase / decrease the sphere ring's vertical wave frequency
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import math
import sys
import traceback

import numpy as np
import wgpu
from matrix_stack import MatrixStack
from ncca.ngl import PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("colour", np.float32, 4),
        ("light_pos", np.float32, 4),
        ("light_diffuse", np.float32, 4),
    ]
)
TROLL_INSTANCE_COUNT = 3
SPHERE_INSTANCE_COUNT = len(np.arange(0.0, 2.0 * math.pi, 0.05))
FLOOR_INSTANCE_COUNT = 1
TOTAL_INSTANCE_COUNT = (
    TROLL_INSTANCE_COUNT + SPHERE_INSTANCE_COUNT + FLOOR_INSTANCE_COUNT
)


def quad_floor(size: float) -> np.ndarray:
    """Interleaved x,y,z,nx,ny,nz,u,v flat quad facing +y, centred at the origin."""
    h = size * 0.5
    corners = [(-h, 0, h), (h, 0, h), (h, 0, -h), (-h, 0, -h)]
    n = (0.0, 1.0, 0.0)
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    order = (0, 1, 2, 0, 2, 3)
    verts = [(*corners[i], *n, *uvs[i]) for i in order]
    return np.array(verts, dtype=np.float32).reshape(-1)


def build_scene_uniforms(
    stack: MatrixStack,
    rotation: float,
    freq: float,
    model_position: Vec3,
    spin_x_face: float,
    spin_y_face: float,
) -> np.ndarray:
    """
    Builds all of the per-instance uniforms before the render pass starts.

    Parameters
    ----------
        stack : MatrixStack
            the shared matrix stack with its current view and projection
        rotation : float
            current animation angle in degrees
        freq : float
            vertical wave frequency for the sphere ring
        model_position : Vec3
            mouse-controlled scene translation
        spin_x_face : float
            mouse-controlled X rotation in degrees
        spin_y_face : float
            mouse-controlled Y rotation in degrees
    """
    uniforms = np.zeros(TOTAL_INSTANCE_COUNT, dtype=UNIFORM_DTYPE)
    uniforms["light_pos"] = (1.0, 1.0, 1.0, 0.0)
    uniforms["light_diffuse"] = (1.0, 1.0, 1.0, 1.0)
    instance = 0

    def store_current(colour: tuple[float, float, float, float]) -> None:
        nonlocal instance
        uniforms[instance]["mvp"] = stack.mvp().to_numpy()
        uniforms[instance]["normal_matrix"] = (
            stack.mv().inverse().transposed().to_numpy()
        )
        uniforms[instance]["colour"] = colour
        instance += 1

    stack.push_matrix()
    stack.translate(0.0, 0.0, model_position.z)
    stack.translate(model_position.x, model_position.y, 0.0)
    stack.rotate_axis_angle(spin_x_face, 1.0, 0.0, 0.0)
    stack.rotate_axis_angle(spin_y_face, 0.0, 1.0, 0.0)

    stack.push_matrix()
    stack.translate(0.0, -0.65, 0.0)
    store_current((1.0, 1.0, 1.0, 1.0))
    stack.pop_matrix()

    stack.push_matrix()
    stack.scale(0.5, 0.5, 0.5)
    stack.translate(-1.0, -1.85, -1.0)
    stack.rotate_axis_angle(45.0, 0.0, 1.0, 0.0)
    store_current((1.0, 1.0, 1.0, 1.0))
    stack.pop_matrix()

    stack.push_matrix()
    stack.scale(0.5, 0.5, 0.5)
    stack.translate(1.0, -1.85, -1.0)
    store_current((1.0, 1.0, 1.0, 1.0))
    stack.pop_matrix()

    for angle in np.arange(0.0, 2.0 * math.pi, 0.05):
        x = math.cos(angle) * 2.0
        z = math.sin(angle) * 2.0
        y = math.sin(angle * freq) * 0.5
        stack.push_matrix()
        stack.rotate_axis_angle(rotation, 0.0, 1.0, 0.0)
        stack.translate(x, y, z)
        stack.push_matrix()
        stack.scale(0.04, 0.04, 0.04)
        stack.rotate_axis_angle(rotation * 2.0, 0.0, 1.0, 0.0)
        store_current((abs(x), abs(y), abs(z), 1.0))
        stack.pop_matrix()
        stack.pop_matrix()

    stack.push_matrix()
    stack.translate(0.0, -1.2, 0.0)
    store_current((1.0, 1.0, 1.0, 1.0))
    stack.pop_matrix()
    stack.pop_matrix()
    return uniforms


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MatrixStack (WebGPU)")
        self.msaa_sample_count = 4

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

        self.stack = MatrixStack()
        self.rotation = 0.0
        self.freq = 1.0

        self.stack.set_view(look_at(Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0)))
        self.stack.set_projection(
            perspective(
                45.0, self.width() / self.height(), 0.05, 350.0, PerspMode.WebGPU
            )
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_geometry()
        self._create_render_buffer()

        timer = QTimer(self)
        timer.timeout.connect(self._advance)
        timer.start(10)

    def _advance(self) -> None:
        self.rotation += 1.0
        self.update()

    def _create_pipeline(self) -> None:
        from pathlib import Path

        shader_src = (Path(__file__).parent / "MatrixStackShader.wgsl").read_text()
        self.shader_module = self.device.create_shader_module(code=shader_src)
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.uniform_buffer = self.device.create_buffer(
            size=TOTAL_INSTANCE_COUNT * UNIFORM_DTYPE.itemsize,
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
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.shader_module,
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
                "module": self.shader_module,
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

    def _make_object(self, data: np.ndarray):
        vertex_buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        return {
            "vertex_buffer": vertex_buffer,
            "count": data.size // 8,
        }

    def _create_geometry(self) -> None:
        self.troll = self._make_object(PrimData.primitive(Prims.TROLL.value))
        self.sphere = self._make_object(PrimData.sphere(1.0, 20))
        self.floor = self._make_object(quad_floor(10.0))

    def _draw_instances(
        self,
        render_pass,
        obj: dict,
        instance_count: int,
        first_instance: int,
    ) -> None:
        render_pass.set_vertex_buffer(0, obj["vertex_buffer"])
        render_pass.draw(obj["count"], instance_count, 0, first_instance)

    def _upload_and_draw_scene(self, render_pass) -> None:
        uniforms = build_scene_uniforms(
            self.stack,
            self.rotation,
            self.freq,
            self.model_position,
            self.spin_x_face,
            self.spin_y_face,
        )
        # Rewriting one uniform for every draw leaves each queued draw seeing
        # the last value. Upload the complete frame once and index it per instance.
        self.device.queue.write_buffer(self.uniform_buffer, 0, uniforms.tobytes())
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        self._draw_instances(render_pass, self.troll, TROLL_INSTANCE_COUNT, 0)
        self._draw_instances(
            render_pass,
            self.sphere,
            SPHERE_INSTANCE_COUNT,
            TROLL_INSTANCE_COUNT,
        )
        self._draw_instances(
            render_pass,
            self.floor,
            FLOOR_INSTANCE_COUNT,
            TOTAL_INSTANCE_COUNT - FLOOR_INSTANCE_COUNT,
        )

    def paintWebGPU(self) -> None:
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
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
        self._upload_and_draw_scene(render_pass)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.stack.set_projection(
            perspective(45.0, width / height, 0.05, 350.0, PerspMode.WebGPU)
        )
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_I:
            self.freq += 1.0
        elif key == Qt.Key_O:
            self.freq -= 1.0
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
