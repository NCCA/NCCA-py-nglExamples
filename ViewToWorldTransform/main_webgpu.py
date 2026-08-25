#!/usr/bin/env -S uv run --script
"""
ViewToWorldTransform: click to place objects in world space (WebGPU).

Same shift-click-to-place behaviour as the OpenGL version (main.py), sharing
the identical view_to_world.unproject_point maths.

Controls:
    Shift+LMB  place a cube at the unprojected point
    Space  clear placed cubes
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from view_to_world import unproject_point
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype(
    [("mvp", np.float32, (4, 4)), ("normal_matrix", np.float32, (4, 4))]
)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ViewToWorldTransform (WebGPU)")
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

        self.click_positions: list[Vec3] = []
        self.last_click_screen: tuple[int, int] | None = None

        self.view = look_at(Vec3(0, 0, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.5, 50.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()

        cube_data = PrimData.primitive(Prims.CUBE.value)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=cube_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = cube_data.size // 8
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "ViewToWorldShader.wgsl").read_text()
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

    def scene_global_tx(self) -> Mat4:
        """The mouse orbit/pan transform shared by drawing and picking.

        `paintWebGPU` draws every placed cube through this, and
        `mousePressEvent` folds it into the unproject matrix too --
        otherwise a click only maps to the right world point when the
        camera hasn't been orbited or panned since load.
        """
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z
        return global_tx

    def paintWebGPU(self) -> None:
        global_tx = self.scene_global_tx()

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

        uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        for position in self.click_positions:
            model = Mat4().translate(position.x, position.y, position.z)
            mv = self.view @ global_tx @ model
            uniforms["mvp"] = (self.project @ mv).to_numpy()
            uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
            uniform_buffer = self.device.create_buffer_with_data(
                data=uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM
            )
            bind_group = self.device.create_bind_group(
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
            render_pass.set_bind_group(0, bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.vertex_count)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        if self.last_click_screen is not None and self.click_positions:
            last = self.click_positions[-1]
            sx, sy = self.last_click_screen
            text = f"Pos=({sx},{sy}) World=({last.x:.2f},{last.y:.2f},{last.z:.2f})"
            self.render_text(10, 20, text, 14, "Arial", QColor(255, 255, 255))

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.5, 50.0, PerspMode.WebGPU)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.click_positions.clear()
            self.last_click_screen = None
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            # Mouse events arrive in Qt logical pixels, but the render
            # target (self.texture_size, set in WebGPUWidget.resizeEvent)
            # is sized in device pixels. unproject_point needs x/y and
            # width/height in the SAME pixel space, so scale the click by
            # the device pixel ratio and pass texture_size -- the exact
            # dimensions the scene was rasterised at -- rather than the
            # logical self.width()/self.height().
            ratio = self.devicePixelRatioF()
            x, y = int(position.x() * ratio), int(position.y() * ratio)
            width, height = self.texture_size
            global_tx = self.scene_global_tx()
            view_projection = (self.project @ self.view @ global_tx).to_numpy()
            world = unproject_point(x, y, width, height, view_projection)
            self.click_positions.append(Vec3(*world))
            self.last_click_screen = (x, y)
            self.update()
        elif event.button() == Qt.LeftButton:
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
