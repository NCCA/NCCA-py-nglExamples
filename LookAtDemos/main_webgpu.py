#!/usr/bin/env -S uv run --script
"""
LookAtDemos: ngl::lookAt and ngl::perspective/ortho (WebGPU).

Same simple/multi-view comparison as the OpenGL version (main.py), using
render_pass.set_viewport()/set_scissor_rect() to draw all four quadrants in
one WebGPU render pass (same technique as BVHViewer's four-view mode). The
reference grid is dropped (no baked WebGPU line-grid data); each viewport
shows the troll only.

Controls:
    Tab  toggle simple / multi-view mode
    LMB rotate  RMB pan  wheel zoom (perspective view only)  Space reset  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, ortho, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype(
    [("mvp", np.float32, (4, 4)), ("normal_matrix", np.float32, (4, 4))]
)

# One uniform buffer + bind group per on-screen pane (up to 4 in multi-view
# mode; simple mode just uses pane 0). WebGPU's queue-timeline ordering only
# guarantees a submitted command buffer sees a resource's state as of
# immediately before that submit - a single shared buffer rewritten in a
# loop before one submit() would have every draw call in that render pass
# observe only the last write. Per-pane buffers (the same pattern
# BVHViewer's webgpu_renderer.py uses for its four camera views) avoids that.
PANE_COUNT = 4


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LookAtDemos (WebGPU)")
        self.msaa_sample_count = 4
        self.multi_view = False

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

        self.device = get_default_device()
        self._create_pipeline()

        troll_data = PrimData.primitive(Prims.TROLL.value)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=troll_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = troll_data.size // 8
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.pane_uniform_buffers = []
        self.pane_bind_groups = []
        for pane_index in range(PANE_COUNT):
            uniform_buffer = self.device.create_buffer(
                size=self.uniforms.nbytes,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"look_at_pane_{pane_index}",
            )
            self.pane_uniform_buffers.append(uniform_buffer)
            self.pane_bind_groups.append(
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
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "LookAtShader.wgsl").read_text()
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
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
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
            multisample={"count": self.msaa_sample_count},
        )

    def _draw_pane(
        self,
        render_pass,
        view: Mat4,
        project: Mat4,
        model: Mat4,
        pane_index: int,
    ) -> None:
        mv = view @ model
        self.uniforms["mvp"] = (project @ mv).to_numpy()
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        uniform_buffer = self.pane_uniform_buffers[pane_index]
        self.device.queue.write_buffer(uniform_buffer, 0, self.uniforms.tobytes())
        render_pass.set_bind_group(0, self.pane_bind_groups[pane_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)

    def paintWebGPU(self) -> None:
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

        # Viewport/scissor rects must be in device pixels - self.texture_size
        # is the actual render-target size (self.width()/height() are Qt
        # logical pixels, which only match on a devicePixelRatio == 1
        # display and would otherwise draw into just the top-left corner of
        # the real framebuffer on HiDPI/Retina screens).
        w, h = self.texture_size
        if not self.multi_view:
            render_pass.set_viewport(0, 0, w, h, 0.0, 1.0)
            render_pass.set_scissor_rect(0, 0, w, h)
            view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
            project = perspective(45.0, w / h, 0.05, 350.0, PerspMode.WebGPU)
            self._draw_pane(render_pass, view, project, self.mouse_global_tx, 0)
        else:
            half_w, half_h = w // 2, h // 2
            # Per-pane near/far match the OpenGL sibling's (main.py):
            # (0.1, 100) for top/side, (0.01, 200) for front. The ortho
            # bounds are -aspect..aspect rather than a fixed -1..1 -- each
            # quadrant here is half_w x half_h, and a fixed box stretches X
            # on a non-square pane, which would falsely suggest orthographic
            # projection itself distorts shapes.
            panes = [
                (
                    (0, half_h, half_w, half_h),
                    Vec3(0, 2, 0),
                    Vec3(0, 0, -1),
                    True,
                    0.1,
                    100,
                ),
                (
                    (half_w, half_h, half_w, half_h),
                    Vec3(0, 1, 1),
                    Vec3(0, 1, 0),
                    False,
                    0.01,
                    100,
                ),
                ((0, 0, half_w, half_h), Vec3(0, 0, 2), Vec3(0, 1, 0), True, 0.01, 200),
                (
                    (half_w, 0, half_w, half_h),
                    Vec3(2, 0, 0),
                    Vec3(0, 1, 0),
                    True,
                    0.1,
                    100,
                ),
            ]
            for pane_index, (
                (x, y, pw, ph),
                eye,
                up,
                is_ortho,
                near,
                far,
            ) in enumerate(panes):
                render_pass.set_viewport(x, y, pw, ph, 0.0, 1.0)
                render_pass.set_scissor_rect(x, y, pw, ph)
                view = look_at(eye, Vec3(0, 0, 0), up)
                aspect = pw / max(ph, 1)
                if is_ortho:
                    project = ortho(-aspect, aspect, -1, 1, near, far, PerspMode.WebGPU)
                    model = Mat4()
                else:
                    project = perspective(45.0, aspect, near, far, PerspMode.WebGPU)
                    model = self.mouse_global_tx
                self._draw_pane(render_pass, view, project, model, pane_index)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Tab:
            self.multi_view = not self.multi_view
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
