#!/usr/bin/env -S uv run --script
"""ShadedGrid demo (WebGPU): an animated wave grid, Phong-shaded.

Same wave-height grid and animation as the OpenGL version in main.py --
both share wave_grid.py's build_wave_grid(), so see that module for the
derivation of the heightfield and its normals. This entry point
deliberately does not include the geometry-shader normal-visualization
pass main.py draws on top: WebGPU has no geometry-shader stage, and
reinterpreting it as a compute pass was ruled out of scope. What's left is
the shaded, animating grid on its own.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wave_grid import build_wave_grid
from wgpu.utils import get_default_device

_GRID_N = 40
_GRID_SIZE = 4.0

# std140-style uniform blocks -- see PhongGridShader.wgsl for the matching
# WGSL struct layouts (vec3 fields are padded to 16 bytes there too).
_TRANSFORM_DTYPE = np.dtype(
    [
        ("M", np.float32, (4, 4)),
        ("MVP", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("viewerPos", np.float32, 3),
        ("_pad0", np.float32, 1),
    ]
)
_MATERIAL_DTYPE = np.dtype(
    [
        ("ambient", np.float32, 4),
        ("diffuse", np.float32, 4),
        ("specular", np.float32, 4),
        ("shininess", np.float32, 1),
        ("_pad0", np.float32, 3),
    ]
)
_LIGHT_DTYPE = np.dtype(
    [
        ("position", np.float32, 3),
        ("_pad0", np.float32, 1),
        ("ambient", np.float32, 4),
        ("diffuse", np.float32, 4),
        ("specular", np.float32, 4),
    ]
)
_LIGHTING_DTYPE = np.dtype(
    [
        ("material", _MATERIAL_DTYPE),
        ("lights", _LIGHT_DTYPE, 3),
    ]
)


class WebGPUScene(WebGPUWidget):
    """The animating wave grid, Phong-shaded with three point lights."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ShadedGrid (WebGPU)")
        self.msaa_sample_count = 4

        # --- camera / mouse state (same conventions as the other WebGPU demos) ---
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

        # --- animation state ---
        self.animate = True
        self.offset = 0.0
        self.vertex_count = 0

        self.eye = Vec3(0.0, 3.0, 6.0)
        self.view = look_at(self.eye, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.5, 150.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "PhongGridShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.pipeline = self.device.create_render_pipeline(
            label="shaded_grid_pipeline",
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
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
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
        # Per-frame transform uniforms (M, MVP, normal_matrix, viewerPos).
        self.transform_uniforms = np.zeros((), dtype=_TRANSFORM_DTYPE)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="shaded_grid_transform_uniforms",
        )

        # Material + 3-point-light uniforms, matching main.py's values --
        # these never change, so the buffer is written once here.
        lighting = np.zeros((), dtype=_LIGHTING_DTYPE)
        lighting["material"]["ambient"] = (0.329412, 0.223529, 0.027451, 1.0)
        lighting["material"]["diffuse"] = (0.780392, 0.568627, 0.113725, 1.0)
        lighting["material"]["specular"] = (0.992157, 0.941176, 0.807843, 1.0)
        lighting["material"]["shininess"] = 57.8974
        lighting["lights"][0]["position"] = (3.0, 2.0, 2.0)
        lighting["lights"][0]["ambient"] = (0.1, 0.1, 0.1, 1.0)
        lighting["lights"][0]["diffuse"] = (1.0, 1.0, 1.0, 1.0)
        lighting["lights"][0]["specular"] = (1.0, 1.0, 1.0, 1.0)
        lighting["lights"][1]["position"] = (-3.0, 1.5, 2.0)
        lighting["lights"][1]["ambient"] = (0.05, 0.05, 0.05, 1.0)
        lighting["lights"][1]["diffuse"] = (0.6, 0.6, 0.6, 1.0)
        lighting["lights"][1]["specular"] = (0.6, 0.6, 0.6, 1.0)
        lighting["lights"][2]["position"] = (0.0, 1.0, -3.0)
        lighting["lights"][2]["ambient"] = (0.05, 0.05, 0.05, 1.0)
        lighting["lights"][2]["diffuse"] = (0.4, 0.4, 0.4, 1.0)
        lighting["lights"][2]["specular"] = (0.4, 0.4, 0.4, 1.0)
        self.lighting_buffer = self.device.create_buffer_with_data(
            data=lighting.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="shaded_grid_lighting_uniforms",
        )

        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.transform_buffer,
                        "offset": 0,
                        "size": self.transform_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.lighting_buffer,
                        "offset": 0,
                        "size": self.lighting_buffer.size,
                    },
                },
            ],
        )

        # Vertex buffer sized once -- the grid's vertex count never changes
        # frame to frame, only the positions/normals do, so a single
        # COPY_DST buffer rewritten with queue.write_buffer() each tick is
        # the right tool (see BVHViewer/webgpu_renderer.py for the same
        # pattern on its line buffer).
        data = build_wave_grid(_GRID_N, _GRID_SIZE, self.offset)
        self.vertex_count = data.size // 8
        self.vertex_buffer = self.device.create_buffer(
            size=data.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="shaded_grid_vertices",
        )
        self.device.queue.write_buffer(self.vertex_buffer, 0, data.tobytes())

    def _upload_grid(self) -> None:
        """Rebuild the grid at the current offset and re-upload it.

        Mirrors main.py's _upload_grid(): called every animation tick with
        a full CPU-side rebuild, then pushed to the GPU with
        queue.write_buffer() rather than recreating the buffer.
        """
        data = build_wave_grid(_GRID_N, _GRID_SIZE, self.offset)
        self.vertex_count = data.size // 8
        self.device.queue.write_buffer(self.vertex_buffer, 0, data.tobytes())

    def _on_tick(self) -> None:
        if self.animate:
            self.offset += 0.02
            self._upload_grid()
            self.update()

    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        global_tx = self.scene_global_tx()
        mvp = self.project @ self.view @ global_tx
        normal_matrix = global_tx.inverse().transposed()

        self.transform_uniforms["M"] = global_tx.to_numpy()
        self.transform_uniforms["MVP"] = mvp.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.transform_uniforms["viewerPos"] = self.eye.to_numpy()
        self.device.queue.write_buffer(
            self.transform_buffer, 0, self.transform_uniforms.tobytes()
        )

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.2, 0.2, 0.2, 1.0),
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
        render_pass.draw(self.vertex_count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.5, 150.0, PerspMode.WebGPU
        )
        self.update()

    # ------------------------------------------------------------------
    # input (hand-copied from Blending/BlendingWebGPU.py -- no shared
    # mixin exists for QWidget-based WebGPU demos)
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_U:
            self.animate = not self.animate
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

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # Stop the animation timer before the base class tears down the
        # wgpu surface/device -- otherwise a queued timer tick can fire a
        # GPU call (queue.write_buffer / paintWebGPU) after teardown and
        # crash. Same fix as main.py's closeEvent, ported here from the
        # start rather than waiting for it to bite.
        self.animation_timer.stop()
        super().closeEvent(event)


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
