#!/usr/bin/env -S uv run --script
"""RayTriangle (WebGPU): 50 static triangles tested every frame against one
keyboard-moved ray -- independent WebGPU port of Collisions/RayTriangle/main.py,
same triangle-spawn ranges and ray-control keys, adapted for wgpu's
rendering model.

A hit triangle is tinted red rather than drawn wireframe (wgpu has no
practical per-draw polygon-mode toggle against a pooled pipeline), and the
hit-point markers are rebuilt-every-frame GPU points -- one per
simultaneously-hit triangle -- rather than mesh spheres; see the module
docstring notes inline for why. There's no animation timer here, same as
main.py -- the scene only changes on key input, so every triangle is
re-tested against the ray on every paintWebGPU call rather than gated
behind a tick.
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
from collision_maths import ray_triangle_intersect

_STEP = 0.5
_NUM_TRIANGLES = 50
# 50 triangle draws + 50 v0-cube-marker draws, one pool slot per draw call/frame.
_DRAW_POOL_SIZE = _NUM_TRIANGLES * 2


def _random_triangle() -> tuple[Vec3, Vec3, Vec3]:
    """One triangle: a centre 10 units out along a random unit vector, then
    3 verts independently jittered around it -- duplicated from main.py's
    helper of the same name (scene-setup only, not part of the shared
    collision_maths.py API)."""
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    c = Vec3(*(axis * 10.0))
    verts = []
    for _ in range(3):
        verts.append(
            Vec3(
                c.x + random.uniform(-2, 2) + 0.1,
                c.y + random.uniform(-2, 2) + 0.1,
                c.z - random.uniform(0, 2) + 0.1,
            )
        )
    return tuple(verts)


def _calc_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    e1 = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
    e2 = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
    n = np.cross(e1, e2)
    length = np.linalg.norm(n)
    if length > 0:
        n = n / length
    return Vec3(*n)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.ray_start = Vec3(0, 0, 0.2)
        self.ray_end = Vec3(0, 0, -20)
        self.hit_points: list = []
        self.view = look_at(Vec3(0, 1, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
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
        self._create_triangles()
        self._create_draw_buffer_pool()
        self._create_marker_buffers()
        self._create_render_buffer()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_pipelines(self) -> None:
        shader_path = Path(__file__).parent / "RayTriangleShader.wgsl"
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
            label="ray_triangle_mesh",
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
                # Unlike a closed mesh (a sphere, say), a triangle here is
                # a single open-faced polygon -- main.py never enables
                # GL_CULL_FACE either, so both sides need to stay visible
                # as the camera orbits. The v0 cube markers share this
                # pipeline too; culling "none" costs a few overdrawn back
                # faces on those and nothing else.
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        # The ray line and hit-point marker share one bind group layout
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
        common_line_kwargs = {
            "layout": line_pipeline_layout,
            "vertex": {
                "module": shader_module,
                "entry_point": "vs_line",
                "buffers": [line_vertex_buffer_layout],
            },
            "fragment": {
                "module": shader_module,
                "entry_point": "fs_line",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            "depth_stencil": {
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            "multisample": {"count": self.msaa_sample_count},
        }
        self.line_pipeline = self.device.create_render_pipeline(
            label="ray_triangle_line",
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            **common_line_kwargs,
        )
        self.point_pipeline = self.device.create_render_pipeline(
            label="ray_triangle_hit_points",
            primitive={"topology": wgpu.PrimitiveTopology.point_list},
            **common_line_kwargs,
        )

    def _create_geometry(self) -> None:
        cube_data = PrimData.primitive(Prims.CUBE.value)
        self.cube_vertex_buffer = self.device.create_buffer_with_data(
            data=cube_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.cube_vertex_count = cube_data.size // 8

    def _create_triangles(self) -> None:
        # Each triangle gets its own tiny position+normal vertex buffer,
        # built from its own 3 vertices rather than instancing one shared
        # mesh (padded with a dummy 0,0 UV pair so the layout matches the
        # other meshes' pos+normal+uv stride and both can share
        # mesh_pipeline). Built once here, since triangles are static once
        # spawned -- only the ray moves every frame.
        self.triangles = []
        for _ in range(_NUM_TRIANGLES):
            v0, v1, v2 = _random_triangle()
            normal = _calc_normal(v0, v1, v2)
            data = np.array(
                [
                    v0.x,
                    v0.y,
                    v0.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                    v1.x,
                    v1.y,
                    v1.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                    v2.x,
                    v2.y,
                    v2.z,
                    normal.x,
                    normal.y,
                    normal.z,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            vertex_buffer = self.device.create_buffer_with_data(
                data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
            )
            self.triangles.append(
                {
                    "v0": v0,
                    "v1": v1,
                    "v2": v2,
                    "vertex_buffer": vertex_buffer,
                    "hit": False,
                }
            )

    def _create_draw_buffer_pool(self) -> None:
        # One uniform buffer/bind-group slot per draw (100: 50 triangles +
        # 50 v0 cube markers), indexed by a counter reset every frame --
        # see RaySphere/main_webgpu.py for the established rationale (a
        # single shared buffer rewritten mid-frame would alias between
        # draws issued before the GPU has consumed the earlier write).
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
        # Ray line: one 2-vertex line-list draw, rebuilt every frame since
        # the endpoints move under keyboard control -- not pooled, per the
        # Dynamic-marker pattern.
        self.line_vertex_buffer = self.device.create_buffer(
            size=2 * 6 * 4,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="ray_line",
        )
        self.line_uniform_buffer = self.device.create_buffer(
            size=16 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="ray_triangle_line_mvp",
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

        # Hit-point markers: worst case every triangle is simultaneously
        # pierced by the ray -- size for the full worst case, one point
        # per triangle (_NUM_TRIANGLES), mirroring
        # RaySphere/main_webgpu.py's point_vertex_capacity pattern (sized
        # there for its own worst case of 4 points per hit sphere).
        # Rebuilt from the current per-triangle test results each
        # paintWebGPU call; only the first point_vertex_count vertices
        # are drawn. When no triangle is hit that frame, the draw call is
        # skipped entirely (see paintWebGPU) rather than issuing a
        # 0-vertex draw.
        self.point_vertex_capacity = _NUM_TRIANGLES
        self.point_vertex_buffer = self.device.create_buffer(
            size=self.point_vertex_capacity * 6 * 4,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="ray_triangle_hit_points",
        )
        self.point_vertex_count = 0

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _test_ray(self) -> None:
        # Re-tested every paintWebGPU call rather than gated behind a
        # timer -- this demo has no animation timer (matches main.py),
        # the scene only changes on key input, so re-testing every frame
        # is cheap and always current.
        ray_start_np = np.array([self.ray_start.x, self.ray_start.y, self.ray_start.z])
        ray_end_np = np.array([self.ray_end.x, self.ray_end.y, self.ray_end.z])
        # Accumulate every hit point this frame, not just the last one --
        # 50 triangle centres scattered on a radius-10 sphere plausibly
        # produce close pairs, and the ray endpoints are under full
        # manual (8-key) control, so more than one simultaneous hit is
        # genuinely reachable, not just a theoretical edge case.
        self.hit_points = []
        for tri in self.triangles:
            hit, hit_point = ray_triangle_intersect(
                ray_start_np,
                ray_end_np,
                np.array([tri["v0"].x, tri["v0"].y, tri["v0"].z]),
                np.array([tri["v1"].x, tri["v1"].y, tri["v1"].z]),
                np.array([tri["v2"].x, tri["v2"].y, tri["v2"].z]),
            )
            tri["hit"] = hit
            if hit:
                self.hit_points.append(hit_point)

    # ------------------------------------------------------------------
    # per-frame geometry rebuilds
    # ------------------------------------------------------------------
    def _update_line_buffer(self) -> None:
        data = np.array(
            [
                self.ray_start.x,
                self.ray_start.y,
                self.ray_start.z,
                1.0,
                1.0,
                1.0,
                self.ray_end.x,
                self.ray_end.y,
                self.ray_end.z,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.device.queue.write_buffer(self.line_vertex_buffer, 0, data.tobytes())

    def _update_point_buffer(self) -> None:
        # Mirrors RaySphere/main_webgpu.py's _update_point_buffer: a flat
        # accumulate-all-hits pass over the current test results, written
        # in one go, drawing only point_vertex_count of the fixed-capacity
        # buffer this frame.
        verts: list[float] = []
        for p in self.hit_points:
            verts.extend([float(p[0]), float(p[1]), float(p[2]), 1.0, 0.0, 0.0])

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

        self._test_ray()
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
        for tri in self.triangles:
            colour = (1.0, 1.0, 0.0)
            if tri["hit"]:
                colour = (colour[0] * 1.6, colour[1] * 0.6, colour[2] * 0.6)
            self._draw_instance(
                render_pass,
                draw_index,
                tri["vertex_buffer"],
                3,
                Mat4(),
                colour,
                self.mouse_global_tx,
            )
            draw_index += 1

        for tri in self.triangles:
            m = Mat4().translate(tri["v0"].x, tri["v0"].y, tri["v0"].z) @ Mat4().scale(
                0.06, 0.06, 0.06
            )
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

        render_pass.set_pipeline(self.line_pipeline)
        render_pass.set_bind_group(0, self.line_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.line_vertex_buffer)
        render_pass.draw(2)

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
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Up:
            self.ray_end.y += _STEP
        elif key == Qt.Key_Down:
            self.ray_end.y -= _STEP
        elif key == Qt.Key_Left:
            self.ray_end.x -= _STEP
        elif key == Qt.Key_Right:
            self.ray_end.x += _STEP
        elif key == Qt.Key_W:
            self.ray_start.y += _STEP
        elif key == Qt.Key_Z:
            self.ray_start.y -= _STEP
        elif key == Qt.Key_A:
            self.ray_start.x -= _STEP
        elif key == Qt.Key_S:
            self.ray_start.x += _STEP
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
    window.setWindowTitle("RayTriangle (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
