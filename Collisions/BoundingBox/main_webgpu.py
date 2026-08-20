#!/usr/bin/env -S uv run --script
"""BoundingBox (WebGPU): N spheres bounce inside a cubic bounding box, with
an optional all-pairs sphere/sphere check -- independent WebGPU port of
Collisions/BoundingBox/main.py, same spawn ranges, wall-reflection and
sphere/sphere rules, adapted for wgpu's rendering model.

A colliding sphere is tinted red rather than drawn wireframe (wgpu has no
practical per-draw polygon-mode toggle against a pooled pipeline), same
approach as the other Collisions WebGPU siblings.

The C++'s `+` key has no upper limit on sphere count, but a WebGPU buffer
pool needs a hard ceiling -- pre-allocating one uniform buffer/bind group
per possible sphere. `_POOL_CAP = 200` (4x the default 50) gives generous
headroom for interactive experimentation; `+` becomes a no-op once the
pool is full rather than silently dropping a sphere or crashing. The
default behaviour (50 spheres, add/remove by 1) is unaffected -- the cap
is a WebGPU-architecture necessity, not a simplification.
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
from collision_maths import sphere_bbox_reflect, sphere_sphere_collide  # noqa: E402

_NUM_SPHERES = 50
_HALF_EXTENT = 40.0
_POOL_CAP = 200

_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom face (y = -h)
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top face (y = +h)
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # connecting edges
)


def _box_corners(h: float) -> list[Vec3]:
    return [
        Vec3(-h, -h, -h),
        Vec3(h, -h, -h),
        Vec3(h, -h, h),
        Vec3(-h, -h, h),
        Vec3(-h, h, -h),
        Vec3(h, h, -h),
        Vec3(h, h, h),
        Vec3(-h, h, h),
    ]


def _v3(v: Vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _random_unit_vec3() -> Vec3:
    v = np.random.normal(size=3)
    v = v / np.linalg.norm(v)
    return Vec3(*v)


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(
            random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)
        ),
        "dir": _random_unit_vec3(),
        "radius": random.uniform(0.5, 2.5),
        "hit": False,
    }


class WebGPUScene(WebGPUWidget):
    def __init__(self, num_spheres: int = _NUM_SPHERES) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        # Clamp at construction time too, not just in the `+` key handler
        # -- the buffer pool only ever has _POOL_CAP slots, so a caller
        # passing --spheres above the cap must be capped here as well, or
        # _draw_instance's first paintWebGPU() call IndexErrors straight
        # into the pool.
        num_spheres = min(num_spheres, _POOL_CAP)
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.animate = True
        self.check_sphere_sphere = False
        self.view = look_at(Vec3(0, 80, 80), Vec3(0, 0, 0), Vec3(0, 1, 0))
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
        self._create_box_buffers()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(40)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_pipelines(self) -> None:
        shader_path = Path(__file__).parent / "BoundingBoxShader.wgsl"
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
            label="bounding_box_spheres",
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

        # The box wireframe carries its colour baked into each vertex
        # (always white), so its bind group only needs the MVP -- one
        # shared buffer is enough since the box is a single draw, unlike
        # the pooled per-sphere mesh uniforms above.
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
        self.line_pipeline = self.device.create_render_pipeline(
            label="bounding_box_wireframe",
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
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_geometry(self) -> None:
        # A real generated sphere, not a baked-mesh substitute: PrimData
        # .sphere() is a pure-numpy port of Paul Bourke's classic sphere
        # algorithm (zero GL/Qt/wgpu dependency), producing the same
        # interleaved [x,y,z,nx,ny,nz,u,v] float32 layout that every other
        # buffer here expects. Generated once at unit radius and scaled
        # per-instance via the model matrix (each sphere's actual radius
        # varies, uniform(0.5, 2.5) -- see _draw_instance/paintWebGPU's
        # existing `Mat4().scale(radius, radius, radius)`), so one shared
        # vertex buffer covers every sphere in the pool regardless of its
        # radius. Precision 40 matches main.py's
        # `Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)` for a
        # consistent look between the two backends.
        sphere_data = PrimData.sphere(1.0, 40)
        self.sphere_vertex_buffer = self.device.create_buffer_with_data(
            data=sphere_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.sphere_vertex_count = sphere_data.size // 8

    def _create_draw_buffer_pool(self) -> None:
        # One uniform buffer/bind-group slot per possible sphere
        # (_POOL_CAP = 200), indexed by a counter reset every frame -- see
        # RaySphere/main_webgpu.py and SphereSphere/main_webgpu.py for the
        # established rationale (a single shared buffer rewritten
        # mid-frame would alias between draws issued before the GPU has
        # consumed the earlier write). Only the first len(self.spheres)
        # slots are ever bound/drawn in any given frame -- the rest of the
        # pool just sits idle, never touched, so there is no aliasing risk
        # from having more slots pre-allocated than are currently in use.
        uniform_size = (16 + 16 + 4) * 4  # mvp mat4 + normal_matrix mat4 + colour vec4
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for _ in range(_POOL_CAP):
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

    def _create_box_buffers(self) -> None:
        # The 12-edge wireframe box never changes shape (fixed
        # half-extent-40 cube), so its vertex data is uploaded once here
        # rather than every paintWebGPU call -- only its MVP (which tracks
        # the mouse-orbit camera, same as every other object) is rewritten
        # per frame. Not part of the sphere pool: a single static
        # line-list draw doesn't need per-draw uniform isolation.
        corners = _box_corners(_HALF_EXTENT)
        verts: list[float] = []
        for a, b in _BOX_EDGES:
            verts.extend((*corners[a].to_numpy(), 1.0, 1.0, 1.0))
            verts.extend((*corners[b].to_numpy(), 1.0, 1.0, 1.0))
        data = np.array(verts, dtype=np.float32)
        self.box_vertex_buffer = self.device.create_buffer_with_data(
            data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.box_vertex_count = len(_BOX_EDGES) * 2

        self.box_uniform_buffer = self.device.create_buffer(
            size=16 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="bounding_box_mvp",
        )
        self.box_bind_group = self.device.create_bind_group(
            layout=self.line_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.box_uniform_buffer,
                        "offset": 0,
                        "size": self.box_uniform_buffer.size,
                    },
                }
            ],
        )

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if not self.animate:
            return
        # Reset every sphere's hit flag before re-testing -- forgetting
        # this leaves a sphere permanently tinted after its one and only
        # collision, a bug class already caught in SpherePlane.
        for s in self.spheres:
            s["hit"] = False
            s["pos"] = s["pos"] + s["dir"]

        if self.check_sphere_sphere:
            for current in self.spheres:
                for other in self.spheres:
                    if current is other:
                        continue
                    if sphere_sphere_collide(
                        _v3(other["pos"]),
                        other["radius"],
                        _v3(current["pos"]),
                        current["radius"],
                    ):
                        # Asymmetric by design, straight from the C++: only
                        # the outer/"current" sphere reverses and flags --
                        # the "other" sphere is untouched here (it gets its
                        # own turn as the outer sphere later in this pass).
                        current["dir"] = current["dir"] * -1.0
                        current["hit"] = True

        # Wall reflection always runs, regardless of the S toggle.
        for s in self.spheres:
            hit, new_dir = sphere_bbox_reflect(
                _v3(s["pos"]), _v3(s["dir"]), s["radius"], _HALF_EXTENT
            )
            if hit:
                s["dir"] = Vec3(*new_dir)
                s["hit"] = True
        self.update()

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

        box_mvp = self.project @ self.view @ self.mouse_global_tx
        self.device.queue.write_buffer(
            self.box_uniform_buffer,
            0,
            box_mvp.to_numpy().flatten().astype(np.float32).tobytes(),
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

        render_pass.set_pipeline(self.line_pipeline)
        render_pass.set_bind_group(0, self.box_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.box_vertex_buffer)
        render_pass.draw(self.box_vertex_count)

        # Only len(self.spheres) of the _POOL_CAP pre-allocated slots are
        # drawn this frame -- the rest of the pool sits idle, untouched,
        # not aliased.
        render_pass.set_pipeline(self.mesh_pipeline)
        for draw_index, s in enumerate(self.spheres):
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

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, w: int, h: int) -> None:
        self.project = perspective(
            45.0, float(w) / max(h, 1), 0.05, 350.0, PerspMode.WebGPU
        )

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
        key = event.key()
        if key == Qt.Key_Space:
            self.animate = not self.animate
        elif key == Qt.Key_S:
            self.check_sphere_sphere = not self.check_sphere_sphere
        elif key == Qt.Key_R:
            self.spheres = [_spawn_sphere() for _ in range(len(self.spheres))]
        elif key == Qt.Key_Minus:
            if len(self.spheres) > 1:
                self.spheres.pop()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            if len(self.spheres) < _POOL_CAP:
                self.spheres.append(_spawn_sphere())
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
    parser.add_argument("--spheres", type=int, default=_NUM_SPHERES)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene(num_spheres=args.spheres)
    window.setWindowTitle("BoundingBox (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
