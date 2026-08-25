#!/usr/bin/env -S uv run --script
"""RaySphere (WebGPU): 50 spheres tested each tick against 2 sweeping rays
-- independent WebGPU port of Collisions/RaySphere/main.py, same spawn
ranges and sweep behaviour, adapted for wgpu's rendering model.

A hit sphere is tinted red rather than drawn wireframe (wgpu has no
practical per-draw polygon-mode toggle against a pooled pipeline), and
the ray lines plus their near/far hit-point markers are each a single
rebuilt-every-frame draw rather than individually pooled objects -- see
the module docstring notes inline for why.
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Prims,
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
from collision_maths import ray_sphere_intersect

_NUM_SPHERES = 50
# 50 spheres + 2 ray-start cube markers, one pool slot per draw call/frame.
_DRAW_POOL_SIZE = _NUM_SPHERES + 2


def _v3(v: Vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _hit_points(ray_start: Vec3, ray_dir: Vec3, sphere_pos: Vec3, radius: float):
    """Quadratic-root near/far hit points, for drawing only -- duplicated
    from main.py's helper of the same name (rendering-only geometry, not
    part of the shared collision_maths.py API). Returns (near, far) Vec3
    or (None, None)."""
    d = _v3(ray_dir)
    d = d / np.linalg.norm(d)
    p = _v3(ray_start) - _v3(sphere_pos)
    a = float(d @ d)
    b = 2.0 * float(d @ p)
    c = float(p @ p) - radius * radius
    discrim = b * b - 4.0 * a * c
    if discrim < 0.0:
        return None, None
    root = discrim**0.5
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    o = _v3(ray_start)
    h1 = o + d * t1
    h2 = o + d * t2
    return Vec3(*h1), Vec3(*h2)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.spheres = [
            {
                "pos": Vec3(random.uniform(0, 10), random.uniform(0, 8), 0.0),
                "radius": random.uniform(0, 1) + 0.2,
                "hit": False,
            }
            for _ in range(_NUM_SPHERES)
        ]
        self.ray1_start = Vec3(0, 10, 0)
        self.ray1_end = Vec3(0, -5, 0)
        self.ray2_start = Vec3(0, 0, 20)
        self.ray2_end = Vec3(0, 0, -5)
        self._sweep_forward = True
        self.animate = True
        self.view = look_at(Vec3(0, 0, -25), Vec3(0, 0, 0), Vec3(0, 1, 0))
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
        self._create_pipelines()
        self._create_geometry()
        self._create_draw_buffer_pool()
        self._create_marker_buffers()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(50)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_pipelines(self) -> None:
        shader_path = Path(__file__).parent / "RaySphereShader.wgsl"
        shader_module = self.device.create_shader_module(code=shader_path.read_text())

        self.mesh_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        mesh_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.mesh_bind_group_layout]
        )
        mesh_vertex_buffer_layout = {
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
        self.mesh_pipeline = self.device.create_render_pipeline(
            label="ray_sphere_mesh",
            layout=mesh_pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [mesh_vertex_buffer_layout],
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

        # The ray lines and hit-point markers share one bind group layout
        # and buffer: both draws use vertex positions already in world
        # space, so the same project @ view @ global_tx MVP transforms
        # either -- no per-object model matrix needed, unlike the mesh
        # pool above.
        self.line_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        line_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.line_bind_group_layout]
        )
        line_vertex_buffer_layout = {
            "array_stride": 6 * 4,
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
            ],
        }
        common_line_kwargs = dict(
            layout=line_pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_line",
                "buffers": [line_vertex_buffer_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_line",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )
        self.line_pipeline = self.device.create_render_pipeline(
            label="ray_sphere_lines",
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            **common_line_kwargs,
        )
        self.point_pipeline = self.device.create_render_pipeline(
            label="ray_sphere_hit_points",
            primitive={"topology": wgpu.PrimitiveTopology.point_list},
            **common_line_kwargs,
        )

    def _create_geometry(self) -> None:
        # A real generated sphere, not a baked-mesh substitute -- see
        # BoundingBox/main_webgpu.py's _create_geometry() for the full
        # rationale. Precision 20 matches main.py's
        # `Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)`.
        sphere_data = PrimData.sphere(1.0, 20)
        self.sphere_vertex_buffer = self.device.create_buffer_with_data(
            data=sphere_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.sphere_vertex_count = sphere_data.size // 8

        cube_data = PrimData.primitive(Prims.CUBE.value)
        self.cube_vertex_buffer = self.device.create_buffer_with_data(
            data=cube_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.cube_vertex_count = cube_data.size // 8

    def _create_draw_buffer_pool(self) -> None:
        # One uniform buffer/bind-group slot per draw (52: 50 spheres + 2
        # ray-start cube markers), indexed by a counter reset every frame
        # -- see Spotlight/main_webgpu.py and SphereSphere/main_webgpu.py
        # for the established rationale (a single shared buffer rewritten
        # mid-frame would alias between draws issued before the GPU has
        # consumed the earlier write).
        uniform_size = (16 + 16 + 4) * 4  # mvp mat4 + normal_matrix mat4 + colour vec4
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for _ in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bind_group = self.device.create_bind_group(
                layout=self.mesh_bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": buf, "offset": 0, "size": uniform_size},
                    }
                ],
            )
            self.draw_uniform_buffers.append(buf)
            self.draw_bind_groups.append(bind_group)

    def _create_marker_buffers(self) -> None:
        # Ray lines: one combined 4-vertex line-list draw (2 separate
        # segments), rebuilt every tick since the endpoints animate --
        # not pooled, per the Dynamic-marker pattern.
        self.line_vertex_buffer = self.device.create_buffer(
            size=4 * 6 * 4,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="ray_lines",
        )
        self.line_uniform_buffer = self.device.create_buffer(
            size=16 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="ray_sphere_line_mvp",
        )
        self.line_bind_group = self.device.create_bind_group(
            layout=self.line_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.line_uniform_buffer,
                        "offset": 0,
                        "size": self.line_uniform_buffer.size,
                    },
                }
            ],
        )

        # Hit-point markers: worst case every sphere is hit by both rays
        # at both its near and far point -- 2 rays x 2 points x N spheres.
        # Sized generously up front; only the first point_vertex_count
        # vertices are drawn each frame.
        self.point_vertex_capacity = 4 * len(self.spheres)
        self.point_vertex_buffer = self.device.create_buffer(
            size=self.point_vertex_capacity * 6 * 4,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="ray_sphere_hit_points",
        )
        self.point_vertex_count = 0

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if not self.animate:
            return
        for s in self.spheres:
            hit1 = ray_sphere_intersect(
                _v3(self.ray1_start),
                _v3(self.ray1_end) - _v3(self.ray1_start),
                _v3(s["pos"]),
                s["radius"],
            )
            hit2 = ray_sphere_intersect(
                _v3(self.ray2_start),
                _v3(self.ray2_end) - _v3(self.ray2_start),
                _v3(s["pos"]),
                s["radius"],
            )
            s["hit"] = hit1 or hit2

        step = 0.5 if self._sweep_forward else -0.5
        self.ray1_end.x += step
        self.ray2_end.x -= step
        if self.ray1_end.x > 22.0:
            self._sweep_forward = False
        elif self.ray1_end.x <= -22.0:
            self._sweep_forward = True
        self.update()

    # ------------------------------------------------------------------
    # per-frame geometry rebuilds
    # ------------------------------------------------------------------
    def _update_line_buffer(self) -> None:
        data = np.array(
            [
                self.ray1_start.x,
                self.ray1_start.y,
                self.ray1_start.z,
                1.0,
                1.0,
                1.0,
                self.ray1_end.x,
                self.ray1_end.y,
                self.ray1_end.z,
                1.0,
                1.0,
                1.0,
                self.ray2_start.x,
                self.ray2_start.y,
                self.ray2_start.z,
                1.0,
                1.0,
                1.0,
                self.ray2_end.x,
                self.ray2_end.y,
                self.ray2_end.z,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.device.queue.write_buffer(self.line_vertex_buffer, 0, data.tobytes())

    def _update_point_buffer(self) -> None:
        verts: list[float] = []
        for s in self.spheres:
            if not s["hit"]:
                continue
            for ray_start, ray_end in (
                (self.ray1_start, self.ray1_end),
                (self.ray2_start, self.ray2_end),
            ):
                ray_dir = Vec3(
                    ray_end.x - ray_start.x,
                    ray_end.y - ray_start.y,
                    ray_end.z - ray_start.z,
                )
                near, far = _hit_points(ray_start, ray_dir, s["pos"], s["radius"])
                if near is None:
                    continue
                verts.extend([near.x, near.y, near.z, 1.0, 0.0, 0.0])
                verts.extend([far.x, far.y, far.z, 0.0, 1.0, 0.0])

        self.point_vertex_count = len(verts) // 6
        if verts:
            data = np.array(verts, dtype=np.float32)
            self.device.queue.write_buffer(self.point_vertex_buffer, 0, data.tobytes())

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _draw_instance(
        self,
        render_pass,
        draw_index: int,
        vertex_buffer,
        vertex_count: int,
        m: Mat4,
        colour: tuple,
        global_tx: Mat4,
    ) -> None:
        m = global_tx @ m
        mv = self.view @ m
        mvp = self.project @ mv
        normal_matrix = m.inverse().transposed()
        data = np.zeros(16 + 16 + 4, dtype=np.float32)
        data[0:16] = mvp.to_numpy().flatten()
        data[16:32] = normal_matrix.to_numpy().flatten()
        data[32:36] = np.array([*colour, 1.0], dtype=np.float32)
        self.device.queue.write_buffer(
            self.draw_uniform_buffers[draw_index], 0, data.tobytes()
        )
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.draw(vertex_count)

    def paintWebGPU(self) -> None:
        if not hasattr(self, "device"):
            return
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        self._update_line_buffer()
        self._update_point_buffer()
        line_mvp = self.project @ self.view @ self.mouse_global_tx
        self.device.queue.write_buffer(
            self.line_uniform_buffer,
            0,
            line_mvp.to_numpy().flatten().astype(np.float32).tobytes(),
        )

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
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

        render_pass.set_pipeline(self.mesh_pipeline)
        draw_index = 0
        for start in (self.ray1_start, self.ray2_start):
            m = Mat4().translate(start.x, start.y, start.z)
            self._draw_instance(
                render_pass,
                draw_index,
                self.cube_vertex_buffer,
                self.cube_vertex_count,
                m,
                (1.0, 1.0, 1.0),
                self.mouse_global_tx,
            )
            draw_index += 1

        for s in self.spheres:
            m = Mat4().translate(s["pos"].x, s["pos"].y, s["pos"].z) @ Mat4().scale(
                s["radius"], s["radius"], s["radius"]
            )
            colour = (1.0, 1.0, 0.0)
            if s["hit"]:
                colour = (colour[0] * 1.6, colour[1] * 0.6, colour[2] * 0.6)
            self._draw_instance(
                render_pass,
                draw_index,
                self.sphere_vertex_buffer,
                self.sphere_vertex_count,
                m,
                colour,
                self.mouse_global_tx,
            )
            draw_index += 1

        render_pass.set_pipeline(self.line_pipeline)
        render_pass.set_bind_group(0, self.line_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.line_vertex_buffer)
        render_pass.draw(4)

        if self.point_vertex_count:
            render_pass.set_pipeline(self.point_pipeline)
            render_pass.set_bind_group(0, self.line_bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.point_vertex_buffer)
            render_pass.draw(self.point_vertex_count)

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
        # GPU call (queue.write_buffer / paintWebGPU) after teardown and
        # crash. Same fix as main.py's closeEvent, ported here from the
        # start rather than waiting for it to bite.
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
            self.animate = not self.animate
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
    window.setWindowTitle("RaySphere (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
