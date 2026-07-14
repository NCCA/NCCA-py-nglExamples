#!/usr/bin/env -S uv run --script
"""
Reynolds flocking on the GPU: compute shaders + instanced rendering (WebGPU).

2048 boids by default, each one thread in a compute pass that reads every
other boid's position and velocity (the naive O(N^2) approach - see the
README for the spatial-hash follow-on) and applies the three classic
Reynolds rules: separation, alignment, cohesion, plus a soft push back from
the walls of a bounding box. State lives in two storage buffers per
quantity (position, velocity), ping-ponged A -> B each frame so a boid
never reads a value another thread in the same dispatch has already
overwritten. The maths is developed and tested CPU-side first in
boid_maths.py - BoidsCompute.wgsl is a transcription of that file, not an
independent implementation, and the render pass (BoidsRender.wgsl) fetches
each instance's position/velocity straight out of the same storage buffers
in the vertex shader, with no per-instance vertex buffer at all.

Controls:
    1/2/3  select separation / alignment / cohesion weight
    +/-    adjust the selected weight
    R      re-seed (new random positions and velocities)
    Space  pause / resume the simulation (rendering keeps running)
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from boid_maths import BoidParams
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device

DEFAULT_N = 2048
WEIGHT_STEP = 0.1
BOID_SCALE = 0.18
MAX_DT = 1.0 / 30.0  # clamp the simulation step so a stalled frame doesn't fling boids

# Mirrors SimParams in BoidsCompute.wgsl field for field - see that struct's
# comment for why everything is a scalar float/uint rather than vec3 (no
# WGSL uniform-block padding to worry about).
SIM_PARAMS_DTYPE = np.dtype(
    [
        ("dt", "float32"),
        ("rs", "float32"),
        ("ra", "float32"),
        ("rc", "float32"),
        ("w_sep", "float32"),
        ("w_align", "float32"),
        ("w_coh", "float32"),
        ("v_min", "float32"),
        ("v_max", "float32"),
        ("bound_x", "float32"),
        ("bound_y", "float32"),
        ("bound_z", "float32"),
        ("wall_margin", "float32"),
        ("w_wall", "float32"),
        ("n", "uint32"),
        ("_pad0", "uint32"),
    ]
)

RENDER_UNIFORM_DTYPE = np.dtype(
    [
        ("view_proj", np.float32, (4, 4)),
        ("boid_scale", "float32"),
        ("speed_min", "float32"),
        ("speed_max", "float32"),
        ("_pad0", "float32"),
    ]
)

# The boid arrow shape hardcoded from VertexArrayObject/Boid: 4 triangles
# (12 verts), local +z is the nose, y is up, x is wingspan.
BOID_VERTS = np.array(
    [
        (0.0, 1.0, 1.0),
        (0.0, 0.0, -1.0),
        (-0.5, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.5),
        (-0.5, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.5),
        (0.5, 0.0, 1.0),
    ],
    dtype=np.float32,
).reshape(-1)


def make_initial_state(n: int, params: BoidParams) -> tuple[np.ndarray, np.ndarray]:
    """Random positions inside the bounding box and random headings within
    the speed clamp - vec4 with w=0 throughout (see the padding note in
    BoidsCompute.wgsl)."""
    bounds = np.asarray(params.bounds, dtype=np.float32)
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = np.random.uniform(-bounds, bounds, size=(n, 3)).astype(np.float32)

    directions = np.random.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    directions /= norms
    speeds = np.random.uniform(params.v_min, params.v_max, size=(n, 1)).astype(
        np.float32
    )
    vel = np.zeros((n, 4), dtype=np.float32)
    vel[:, :3] = directions * speeds
    return pos, vel


class WebGPUScene(WebGPUWidget):
    """Boid flock: compute pass updates position/velocity storage buffers,
    render pass draws one instanced arrow per boid straight out of them."""

    def __init__(self, n: int = DEFAULT_N):
        super().__init__()
        self.setWindowTitle("BoidsCompute: Reynolds flocking (WebGPU compute)")
        self.msaa_sample_count = 4
        self.n = n
        self.paused = False
        self.selected_rule = 1  # 1=separation 2=alignment 3=cohesion

        self.params = BoidParams(
            rs=0.6,
            ra=1.5,
            rc=2.0,
            w_sep=1.5,
            w_align=1.0,
            w_coh=1.0,
            v_min=1.0,
            v_max=4.0,
            dt=0.0,
            bounds=(8.0, 5.0, 8.0),
            wall_margin=2.0,
            w_wall=4.0,
        )

        # --- camera / mouse state (same conventions as the other WebGPU demos) ---
        self.model_position = Vec3(0, 0, -20)
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

        self.view = look_at(
            Vec3(0.0, 6.0, 20.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 200.0, PerspMode.WebGPU
        )

        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_time = self.timer.elapsed() / 1000.0

        self.device = get_default_device()
        self._create_buffers()
        self._create_compute_pipeline()
        self._create_render_pipeline()
        self._create_render_buffer()
        self.startTimer(16)
        self.update()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_buffers(self) -> None:
        pos_init, vel_init = make_initial_state(self.n, self.params)

        storage_rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        self.pos_bufs = [
            self.device.create_buffer_with_data(data=pos_init, usage=storage_rw),
            self.device.create_buffer(size=pos_init.nbytes, usage=storage_rw),
        ]
        self.vel_bufs = [
            self.device.create_buffer_with_data(data=vel_init, usage=storage_rw),
            self.device.create_buffer(size=vel_init.nbytes, usage=storage_rw),
        ]
        # `current` indexes which buffer set holds valid simulation state
        # (the other one is scratch, about to be overwritten by the next
        # compute pass). Ping-pongs 0 -> 1 -> 0... every frame.
        self.current = 0

        self.sim_params = np.zeros((), dtype=SIM_PARAMS_DTYPE)
        self.sim_params_buffer = self.device.create_buffer(
            size=self.sim_params.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.boid_vertex_buffer = self.device.create_buffer_with_data(
            data=BOID_VERTS, usage=wgpu.BufferUsage.VERTEX
        )
        self.boid_vertex_count = BOID_VERTS.size // 3

        self.render_uniforms = np.zeros((), dtype=RENDER_UNIFORM_DTYPE)
        self.render_uniform_buffer = self.device.create_buffer(
            size=self.render_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _create_compute_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "BoidsCompute.wgsl").read_text()
        module = self.device.create_shader_module(code=shader_src)

        self.compute_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.compute_bind_group_layout]
        )
        self.compute_pipeline = self.device.create_compute_pipeline(
            label="boids_compute",
            layout=pipeline_layout,
            compute={"module": module, "entry_point": "cs_main"},
        )

        # Two bind groups, one per ping-pong direction: index 0 reads set 0
        # and writes set 1, index 1 reads set 1 and writes set 0. Which one
        # runs this frame is selected purely by self.current.
        def bind_group(read_idx: int, write_idx: int):
            return self.device.create_bind_group(
                layout=self.compute_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.pos_bufs[read_idx]}},
                    {"binding": 1, "resource": {"buffer": self.vel_bufs[read_idx]}},
                    {"binding": 2, "resource": {"buffer": self.pos_bufs[write_idx]}},
                    {"binding": 3, "resource": {"buffer": self.vel_bufs[write_idx]}},
                    {"binding": 4, "resource": {"buffer": self.sim_params_buffer}},
                ],
            )

        self.compute_bind_groups = [bind_group(0, 1), bind_group(1, 0)]

    def _create_render_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "BoidsRender.wgsl").read_text()
        module = self.device.create_shader_module(code=shader_src)

        self.render_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.render_bind_group_layout]
        )
        self.render_pipeline = self.device.create_render_pipeline(
            label="boids_render",
            layout=pipeline_layout,
            vertex={
                "module": module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 3 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": "float32x3",
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            fragment={
                "module": module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
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

        def bind_group(idx: int):
            return self.device.create_bind_group(
                layout=self.render_bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": self.render_uniform_buffer},
                    },
                    {"binding": 1, "resource": {"buffer": self.pos_bufs[idx]}},
                    {"binding": 2, "resource": {"buffer": self.vel_bufs[idx]}},
                ],
            )

        # render_bind_groups[i] reads whichever buffer set holds valid state
        # when self.current == i - rebuilt is unnecessary since the buffer
        # *objects* never change, only which one currently holds "now".
        self.render_bind_groups = [bind_group(0), bind_group(1)]

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _reseed(self) -> None:
        pos_init, vel_init = make_initial_state(self.n, self.params)
        self.device.queue.write_buffer(self.pos_bufs[self.current], 0, pos_init)
        self.device.queue.write_buffer(self.vel_bufs[self.current], 0, vel_init)

    def _write_sim_params(self, dt: float) -> None:
        p = self.params
        self.sim_params["dt"] = dt
        self.sim_params["rs"] = p.rs
        self.sim_params["ra"] = p.ra
        self.sim_params["rc"] = p.rc
        self.sim_params["w_sep"] = p.w_sep
        self.sim_params["w_align"] = p.w_align
        self.sim_params["w_coh"] = p.w_coh
        self.sim_params["v_min"] = p.v_min
        self.sim_params["v_max"] = p.v_max
        self.sim_params["bound_x"] = p.bounds[0]
        self.sim_params["bound_y"] = p.bounds[1]
        self.sim_params["bound_z"] = p.bounds[2]
        self.sim_params["wall_margin"] = p.wall_margin
        self.sim_params["w_wall"] = p.w_wall
        self.sim_params["n"] = self.n
        self.device.queue.write_buffer(self.sim_params_buffer, 0, self.sim_params)

    def _compute_pass(self, dt: float) -> None:
        self._write_sim_params(dt)
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.compute_pipeline)
        compute_pass.set_bind_group(
            0, self.compute_bind_groups[self.current], [], 0, 999999
        )
        workgroups = (self.n + 63) // 64
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        # the write target of this pass now holds the current state
        self.current = 1 - self.current

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        current_time = self.timer.elapsed() / 1000.0
        dt = min(current_time - self.last_time, MAX_DT)
        self.last_time = current_time

        if not self.paused:
            self._compute_pass(dt)

        scene_tx = self._scene_global_tx()
        view_proj = self.project @ self.view @ scene_tx
        self.render_uniforms["view_proj"] = view_proj.to_numpy()
        self.render_uniforms["boid_scale"] = BOID_SCALE
        self.render_uniforms["speed_min"] = self.params.v_min
        self.render_uniforms["speed_max"] = self.params.v_max
        self.device.queue.write_buffer(
            self.render_uniform_buffer, 0, self.render_uniforms
        )

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.05, 0.05, 0.08, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(
            0, self.render_bind_groups[self.current], [], 0, 999999
        )
        render_pass.set_vertex_buffer(0, self.boid_vertex_buffer)
        render_pass.draw(self.boid_vertex_count, self.n)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()
        self._draw_hud()

    def _scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _draw_hud(self) -> None:
        white = QColor(255, 255, 255)
        yellow = QColor(255, 255, 0)
        rule_names = {1: "separation", 2: "alignment", 3: "cohesion"}
        weight = {1: self.params.w_sep, 2: self.params.w_align, 3: self.params.w_coh}[
            self.selected_rule
        ]
        self.render_text(10, 20, f"boids: {self.n}", 14, "Arial", white)
        self.render_text(
            10,
            45,
            f"[1/2/3] rule: {rule_names[self.selected_rule]}   [+/-] weight: {weight:.2f}",
            14,
            "Arial",
            yellow,
        )
        self.render_text(
            10,
            70,
            f"[R] reseed   [Space] {'RESUME' if self.paused else 'PAUSE'}",
            14,
            "Arial",
            white,
        )

    def resizeWebGPU(self, width, height) -> None:
        self.project = perspective(
            45.0, width / height if height else 1.0, 0.1, 200.0, PerspMode.WebGPU
        )
        self.update()

    def timerEvent(self, event) -> None:
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_1:
            self.selected_rule = 1
        elif key == Qt.Key_2:
            self.selected_rule = 2
        elif key == Qt.Key_3:
            self.selected_rule = 3
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._adjust_weight(WEIGHT_STEP)
        elif key == Qt.Key_Minus:
            self._adjust_weight(-WEIGHT_STEP)
        elif key == Qt.Key_R:
            self._reseed()
        elif key == Qt.Key_Space:
            self.paused = not self.paused
        self.update()
        super().keyPressEvent(event)

    def _adjust_weight(self, delta: float) -> None:
        if self.selected_rule == 1:
            self.params.w_sep = max(0.0, self.params.w_sep + delta)
        elif self.selected_rule == 2:
            self.params.w_align = max(0.0, self.params.w_align + delta)
        else:
            self.params.w_coh = max(0.0, self.params.w_coh + delta)

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
        "-n",
        "--num-boids",
        type=int,
        default=DEFAULT_N,
        help=f"number of boids to simulate (default {DEFAULT_N})",
    )
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

    win = WebGPUScene(n=args.num_boids)
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
