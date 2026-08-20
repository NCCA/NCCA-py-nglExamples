#!/usr/bin/env -S uv run --script
"""SpherePlane (WebGPU): N falling spheres collide with a tiltable plane
-- independent WebGPU port of Collisions/SpherePlane/main.py, same spawn
ranges, respawn cadence and tilt controls, adapted for wgpu's rendering
model.

A colliding sphere is tinted red rather than drawn wireframe (wgpu has
no practical per-draw polygon-mode toggle against a pooled pipeline),
same approach as the RaySphere/RayTriangle WebGPU siblings. The plane
is a small hand-built quad rather than an `ncca.ngl` primitive, tilted
at draw time exactly as main.py's OpenGL version does.
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import (
    Mat3,
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
from collision_maths import sphere_plane_collide  # noqa: E402

_NUM_SPHERES = 50
_PLANE_WIDTH = 5.0
_PLANE_DEPTH = 5.0
_RESPAWN_EVERY = 20
# 50 spheres + 1 plane, one pool slot per draw call/frame -- the plane is
# always object index 0.
_DRAW_POOL_SIZE = _NUM_SPHERES + 1


def _v3(v: Vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(random.uniform(-6, 6), 8.0, random.uniform(-6, 6)),
        "dir": Vec3(0.0, -1.0, 0.0),
        "radius": 0.2,
        "hit": False,
    }


def _quad_plane(width: float, depth: float) -> np.ndarray:
    """Interleaved x,y,z,nx,ny,nz,u,v flat quad facing +y, centred at the
    origin -- same pattern as Blending/BlendingWebGPU.py's quad() and
    MatrixStack/main_webgpu.py's quad_floor()."""
    w, d = width * 0.5, depth * 0.5
    corners = [(-w, 0, d), (w, 0, d), (w, 0, -d), (-w, 0, -d)]
    n = (0.0, 1.0, 0.0)
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    order = (0, 1, 2, 0, 2, 3)
    verts = [(*corners[i], *n, *uvs[i]) for i in order]
    return np.array(verts, dtype=np.float32).reshape(-1)


class WebGPUScene(WebGPUWidget):
    def __init__(self, num_spheres: int = _NUM_SPHERES) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.plane_xrot = 0.0
        self.plane_zrot = 0.0
        self.tick_count = 0
        self.view = look_at(Vec3(0, 0, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
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
        self._create_pipeline()
        self._create_geometry()
        self._create_draw_buffer_pool()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(130)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_path = Path(__file__).parent / "SpherePlaneShader.wgsl"
        shader_module = self.device.create_shader_module(code=shader_path.read_text())
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
        vertex_buffer_layout = {
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
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
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
            multisample={"count": self.msaa_sample_count},
        )

    def _create_geometry(self) -> None:
        sphere_data = PrimData.primitive(Prims.OCTAHEDRON.value)
        self.sphere_vertex_buffer = self.device.create_buffer_with_data(
            data=sphere_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.sphere_vertex_count = sphere_data.size // 8

        plane_data = _quad_plane(_PLANE_WIDTH, _PLANE_DEPTH)
        self.plane_vertex_buffer = self.device.create_buffer_with_data(
            data=plane_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.plane_vertex_count = plane_data.size // 8

    def _create_draw_buffer_pool(self) -> None:
        # One uniform buffer/bind-group slot per draw (51: 1 plane + 50
        # spheres), indexed by a counter reset every frame -- see
        # Spotlight/main_webgpu.py and SphereSphere/main_webgpu.py for the
        # established rationale (a single shared buffer rewritten mid-frame
        # would alias between draws issued before the GPU has consumed the
        # earlier write).
        uniform_size = (16 + 16 + 4) * 4  # mvp mat4 + normal_matrix mat4 + colour vec4
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for _ in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": buf, "offset": 0, "size": uniform_size},
                    }
                ],
            )
            self.draw_uniform_buffers.append(buf)
            self.draw_bind_groups.append(bind_group)

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _plane_normal(self) -> Vec3:
        # Mat4 only multiplies with Vec4, so the rotation is extracted
        # into a Mat3 which does support Mat3 @ Vec3 -- same pattern as
        # main.py's _plane_normal() (Task 8) and Camera/uvn_camera.py's
        # _rotate_vec3().
        rot = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        return Mat3.from_mat4(rot) @ Vec3(0, 1, 0)

    def _on_tick(self) -> None:
        normal = self._plane_normal()
        normal_np = _v3(normal)
        for s in self.spheres:
            s["hit"] = False
            s["pos"] = s["pos"] + s["dir"]
            hit = sphere_plane_collide(
                _v3(s["pos"]),
                s["radius"],
                np.array([0.0, 0.0, 0.0]),
                normal_np,
                _PLANE_WIDTH,
                _PLANE_DEPTH,
            )
            if hit:
                s["dir"] = normal
                s["hit"] = True

        self.tick_count += 1
        if self.tick_count >= _RESPAWN_EVERY:
            self.tick_count = 0
            for s in self.spheres:
                fresh = _spawn_sphere()
                s["pos"], s["dir"], s["radius"], s["hit"] = (
                    fresh["pos"],
                    fresh["dir"],
                    fresh["radius"],
                    fresh["hit"],
                )
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
        render_pass.set_pipeline(self.pipeline)

        # draw_index 0 is always the plane, tilted exactly as main.py's
        # paintGL() folds `rotate_z(zrot) @ rotate_x(xrot)` into its model
        # matrix.
        plane_tilt = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        self._draw_instance(
            render_pass,
            0,
            self.plane_vertex_buffer,
            self.plane_vertex_count,
            plane_tilt,
            (1.0, 1.0, 0.0),
            self.mouse_global_tx,
        )

        draw_index = 1
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
        if key == Qt.Key_Up:
            self.plane_xrot += 1.0
        elif key == Qt.Key_Down:
            self.plane_xrot -= 1.0
        elif key == Qt.Key_Left:
            self.plane_zrot -= 1.0
        elif key == Qt.Key_Right:
            self.plane_zrot += 1.0
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
    window.setWindowTitle("SpherePlane (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
