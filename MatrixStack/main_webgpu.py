#!/usr/bin/env -S uv run --script
"""
MatrixStack: an OpenGL-style push/pop matrix stack (WebGPU).

Same push/pop matrix-stack logic as the OpenGL version (main.py) — the stack
itself (matrix_stack.py) is pure CPU-side maths shared unchanged between
both entry points. The ring of small GL-generated spheres is replaced with
the baked octahedron mesh, and the reference grid with a flat quad, since
WebGPU has no equivalent runtime primitive generator.

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

# One uniform buffer + bind group per draw call in a frame: 3 trolls + 126
# ring octahedra (the `while i < 2*pi: ... i += 0.05` loop below runs 126
# times, independent of the freq/rotation state) + 1 floor = 130. WebGPU's
# queue-timeline ordering only guarantees a submitted command buffer sees a
# resource's state as of immediately before that submit -- a single buffer
# rewritten in a loop before one submit() would have every draw call in
# that render pass observe only the last write (the exact bug this pool
# fixes; same pattern as LookAtDemos/main_webgpu.py's per-pane buffers and
# BVHViewer/webgpu_renderer.py's per-camera buffers).
MAX_DRAWS_PER_FRAME = 130


def quad_floor(size: float) -> np.ndarray:
    """Interleaved x,y,z,nx,ny,nz,u,v flat quad facing +y, centred at the origin."""
    h = size * 0.5
    corners = [(-h, 0, h), (h, 0, h), (h, 0, -h), (-h, 0, -h)]
    n = (0.0, 1.0, 0.0)
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    order = (0, 1, 2, 0, 2, 3)
    verts = [(*corners[i], *n, *uvs[i]) for i in order]
    return np.array(verts, dtype=np.float32).reshape(-1)


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
        self._create_draw_buffer_pool()
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

    def _create_draw_buffer_pool(self) -> None:
        """Pre-allocate MAX_DRAWS_PER_FRAME uniform buffers + bind groups.

        _draw_current() hands out one pool slot per draw call, indexed by a
        counter that resets at the start of each frame, instead of the
        original one-buffer-per-mesh-type scheme (which aliased every draw
        sharing a mesh onto whatever transform was written last -- see
        MAX_DRAWS_PER_FRAME's comment above).
        """
        self._draw_uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self._draw_uniform_buffers = []
        self._draw_bind_groups = []
        for index in range(MAX_DRAWS_PER_FRAME):
            uniform_buffer = self.device.create_buffer(
                size=self._draw_uniforms.nbytes,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"matrix_stack_draw_{index}",
            )
            self._draw_uniform_buffers.append(uniform_buffer)
            self._draw_bind_groups.append(
                self.device.create_bind_group(
                    layout=self.bind_group_layout,
                    entries=[
                        {
                            "binding": 0,
                            "resource": {
                                "buffer": uniform_buffer,
                                "offset": 0,
                                "size": uniform_buffer.size,
                            },
                        }
                    ],
                )
            )
        self._draw_index = 0

    def _make_object(self, data: np.ndarray):
        vertex_buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        return {"vertex_buffer": vertex_buffer, "count": data.size // 8}

    def _create_geometry(self) -> None:
        self.troll = self._make_object(PrimData.primitive(Prims.TROLL.value))
        self.octahedron = self._make_object(PrimData.primitive(Prims.OCTAHEDRON.value))
        self.floor = self._make_object(quad_floor(10.0))

    def _draw_current(self, render_pass, obj: dict, colour: tuple) -> None:
        """Draw obj using the stack's *current* top-of-stack transform.

        Reuses MatrixStack.mvp()/mv() directly -- the exact same methods the
        OpenGL version calls -- so the only thing that differs between the
        two backends is how the result reaches the GPU (a GL uniform vs a
        WebGPU uniform buffer), not how it's computed. Each call claims the
        next buffer/bind-group slot from the pre-allocated pool (see
        MAX_DRAWS_PER_FRAME) so draws sharing a mesh don't alias onto the
        same transform.
        """
        uniform_buffer = self._draw_uniform_buffers[self._draw_index]
        bind_group = self._draw_bind_groups[self._draw_index]
        self._draw_index += 1

        self._draw_uniforms["mvp"] = self.stack.mvp().to_numpy()
        self._draw_uniforms["normal_matrix"] = (
            self.stack.mv().inverse().transposed().to_numpy()
        )
        self._draw_uniforms["colour"] = colour
        self._draw_uniforms["light_pos"] = (1.0, 1.0, 1.0, 0.0)
        self._draw_uniforms["light_diffuse"] = (1.0, 1.0, 1.0, 1.0)
        self.device.queue.write_buffer(uniform_buffer, 0, self._draw_uniforms.tobytes())
        render_pass.set_bind_group(0, bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, obj["vertex_buffer"])
        render_pass.draw(obj["count"])

    def paintWebGPU(self) -> None:
        self._draw_index = 0
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

        # Identical push/pop structure to the OpenGL version's paintGL --
        # this is the actual point of the demo: the same MatrixStack calls
        # drive both backends.
        self.stack.push_matrix()
        self.stack.translate(0, 0, self.model_position.z)
        self.stack.translate(self.model_position.x, self.model_position.y, 0)
        self.stack.rotate_axis_angle(self.spin_x_face, 1, 0, 0)
        self.stack.rotate_axis_angle(self.spin_y_face, 0, 1, 0)

        self.stack.push_matrix()
        self.stack.translate(0, -0.65, 0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(-1.0, -1.85, -1.0)
        self.stack.rotate_axis_angle(45, 0, 1, 0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(1.0, -1.85, -1.0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        i = 0.0
        while i < 2.0 * math.pi:
            self.stack.push_matrix()
            x = math.cos(i) * 2.0
            z = math.sin(i) * 2.0
            y = math.sin(i * self.freq) * 0.5
            self.stack.rotate_axis_angle(self.rotation, 0, 1, 0)
            self.stack.translate(x, y, z)
            self.stack.push_matrix()
            self.stack.scale(0.04, 0.04, 0.04)
            self.stack.rotate_axis_angle(self.rotation * 2, 0, 1, 0)
            self._draw_current(
                render_pass, self.octahedron, (abs(x), abs(y), abs(z), 1.0)
            )
            self.stack.pop_matrix()
            self.stack.pop_matrix()
            i += 0.05

        self.stack.push_matrix()
        self.stack.translate(0, -1.2, 0)
        self._draw_current(render_pass, self.floor, (1, 1, 1, 1))
        self.stack.pop_matrix()
        self.stack.pop_matrix()

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
