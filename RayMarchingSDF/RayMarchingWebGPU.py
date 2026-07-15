#!/usr/bin/env -S uv run --script
"""Sphere-traced signed distance fields (WebGPU).

The same scene as main.py (OpenGL): a ground plane, a sphere, a box and a
torus melted together with a smooth minimum, plus one sphere orbiting
overhead, all described inside a single fragment shader (RayMarch.wgsl).
No geometry is uploaded at all -- a full-screen triangle generated from
@builtin(vertex_index) (the same trick as OITransparency's composite pass)
is enough, because every pixel just marches its own ray through the
distance field.

RayMarch.wgsl is a line-for-line transcription of
shaders/RayMarchFragment.glsl -- same function names, same scene
constants, same march loop -- so the README can show them side by side.
The maths itself (sd_sphere, sd_box, sd_torus, sd_plane, smooth_min,
scene) is unit tested once, in numpy, in sdf_maths.py.

Controls:
    S      toggle soft shadows
    O      toggle ambient occlusion
    N      visualise surface normals
    I      visualise the iteration count as a heat map
    +/-    widen/narrow the smooth-min blend radius
    Space  pause/resume the orbiting sphere
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import time
import traceback
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Vec3
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

BASE_DISTANCE = 6.0
TARGET_HEIGHT = 0.8
FOV_DEGREES = 45.0

# Mirrors the WGSL `Params` uniform struct in RayMarch.wgsl: each vec3 is
# immediately followed by the f32 that fills its alignment pad, exactly as
# std140 lays it out, so this can be written straight into the buffer.
PARAMS_DTYPE = np.dtype(
    [
        ("cam_pos", np.float32, 3),
        ("fov_scale", np.float32),
        ("cam_forward", np.float32, 3),
        ("aspect", np.float32),
        ("cam_right", np.float32, 3),
        ("time", np.float32),
        ("cam_up", np.float32, 3),
        ("smooth_k", np.float32),
        ("shadows_on", np.uint32),
        ("ao_on", np.uint32),
        ("show_normals", np.uint32),
        ("show_iterations", np.uint32),
    ]
)


class WebGPUScene(WebGPUWidget):
    """Fullscreen ray-marched SDF scene with the standard PyNGL mouse orbit."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ray Marching SDF (WebGPU)")
        self.msaa_sample_count = 1

        # --- demo state driven by the keyboard ---
        self.shadows_on = True
        self.ao_on = True
        self.show_normals = False
        self.show_iterations = False
        self.smooth_k = 0.3
        self.paused = False
        self._clock_start = time.perf_counter()
        self._paused_time = 0.0

        # --- camera / mouse state (same conventions as the GL demo) ---
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
        self.ZOOM = 0.3

        self.device = get_default_device()
        self._create_pipeline()
        self._create_render_buffer()
        self.startTimer(16)
        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "RayMarch.wgsl").read_text()
        self.shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.params = np.zeros((), dtype=PARAMS_DTYPE)
        self.params_buffer = self.device.create_buffer(
            size=self.params.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.params_buffer,
                        "offset": 0,
                        "size": self.params_buffer.size,
                    },
                }
            ],
        )

        self.pipeline = self.device.create_render_pipeline(
            label="raymarch_pipeline",
            layout=pipeline_layout,
            vertex={"module": self.shader_module, "entry_point": "vertex_main"},
            fragment={
                "module": self.shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    # ------------------------------------------------------------------
    # required override: this demo has no MSAA / depth target, just the
    # single colour texture the widget reads back into its numpy buffer
    # ------------------------------------------------------------------
    def _create_render_buffer(self) -> None:
        if not hasattr(self, "device"):
            return
        colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture = colour_buffer_texture
        self.colour_buffer_texture_view = colour_buffer_texture.create_view()
        buffer_size = self._calculate_aligned_buffer_size()
        self.readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def _elapsed_time(self) -> float:
        if self.paused:
            return self._paused_time
        return time.perf_counter() - self._clock_start

    def _camera_basis(self):
        """Same orbit camera as the OpenGL demo's _camera_basis(): pitch/yaw
        from spin_x/y_face, zoom from model_position.z, pan from
        model_position.x/y. Returns (pos, forward, right, up) as Vec3.
        """
        yaw = radians(self.spin_y_face)
        pitch = radians(self.spin_x_face)
        distance = max(1.0, BASE_DISTANCE - self.model_position.z)
        target = Vec3(self.model_position.x, TARGET_HEIGHT + self.model_position.y, 0.0)
        offset = Vec3(
            distance * cos(pitch) * sin(yaw),
            distance * sin(pitch),
            distance * cos(pitch) * cos(yaw),
        )
        pos = target + offset
        forward = (target - pos).normalized()
        world_up = Vec3(0.0, 1.0, 0.0)
        right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()
        return pos, forward, right, up

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        pos, forward, right, up = self._camera_basis()
        fov_scale = sin(radians(FOV_DEGREES) * 0.5) / cos(radians(FOV_DEGREES) * 0.5)
        width, height = self.texture_size

        self.params["cam_pos"] = pos.to_list()
        self.params["fov_scale"] = fov_scale
        self.params["cam_forward"] = forward.to_list()
        self.params["aspect"] = float(width) / float(height)
        self.params["cam_right"] = right.to_list()
        self.params["time"] = self._elapsed_time()
        self.params["cam_up"] = up.to_list()
        self.params["smooth_k"] = self.smooth_k
        self.params["shadows_on"] = 1 if self.shadows_on else 0
        self.params["ao_on"] = 1 if self.ao_on else 0
        self.params["show_normals"] = 1 if self.show_normals else 0
        self.params["show_iterations"] = 1 if self.show_iterations else 0
        self.device.queue.write_buffer(self.params_buffer, 0, self.params.tobytes())

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.colour_buffer_texture_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.1, 0.1, 0.1, 1.0),
                }
            ]
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.draw(3)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()
        self._draw_hud()

    def _draw_hud(self) -> None:
        white = QColor(255, 255, 255)
        state = (
            f"[S]hadows {'ON' if self.shadows_on else 'OFF'}  "
            f"[O] AO {'ON' if self.ao_on else 'OFF'}  "
            f"[N]ormals {'ON' if self.show_normals else 'OFF'}  "
            f"[I]terations {'ON' if self.show_iterations else 'OFF'}"
        )
        self.render_text(10, 20, state, 14, "Arial", white)
        self.render_text(
            10,
            45,
            f"[+/-] smooth-min k = {self.smooth_k:.2f}   [Space] {'paused' if self.paused else 'running'}",
            14,
            "Arial",
            white,
        )

    def resizeWebGPU(self, width, height) -> None:
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_S:
            self.shadows_on = not self.shadows_on
        elif key == Qt.Key_O:
            self.ao_on = not self.ao_on
        elif key == Qt.Key_N:
            self.show_normals = not self.show_normals
        elif key == Qt.Key_I:
            self.show_iterations = not self.show_iterations
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.smooth_k = min(1.5, self.smooth_k + 0.05)
        elif key == Qt.Key_Minus:
            self.smooth_k = max(0.0, self.smooth_k - 0.05)
        elif key == Qt.Key_Space:
            if not self.paused:
                self._paused_time = self._elapsed_time()
            else:
                self._clock_start = time.perf_counter() - self._paused_time
            self.paused = not self.paused
        self.update()

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

    def timerEvent(self, event) -> None:
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
