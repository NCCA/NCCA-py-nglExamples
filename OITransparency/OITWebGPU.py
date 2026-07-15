#!/usr/bin/env -S uv run --script
"""
Weighted blended order-independent transparency (WebGPU).

The same scene and modes as the OpenGL version (main.py) built on wgpu-py:

    1  naive alpha blending in scene order
    2  per-object back-to-front sorting (still wrong for the intersecting X)
    3  weighted blended OIT (McGuire & Bavoil 2013), order independent

    A/Z increase / decrease panel alpha, Space reset camera, Esc quit

The WebGPU-specific points to notice compared to the OpenGL version:
    - GL's glBlendFunci (a different blend function per attachment) becomes
      a per-target "blend" entry in the pipeline's fragment targets list.
    - GL's FBOs become plain textures used as render pass colour
      attachments; the accumulation pass attaches accum + reveal and *loads*
      the depth texture from the opaque pass without writing it.
    - The composite is a full-screen triangle generated from
      @builtin(vertex_index) -- no vertex buffer at all.

Reference: http://jcgt.org/published/0002/02/09/
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from oit_common import DEFAULT_ALPHA, PANEL_SIZE, PANELS, back_to_front
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

MODE_NAMES = {
    1: "1: naive alpha blend (scene order)",
    2: "2: sorted back-to-front (per object)",
    3: "3: weighted blended OIT (no sorting)",
}

UNIFORM_DTYPE = np.dtype(
    [
        ("MVP", np.float32, (4, 4)),
        ("MV", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("colour", np.float32, 4),
    ]
)


def quad(width: float, height: float, normal_axis: str = "z") -> np.ndarray:
    """Interleaved x,y,z,nx,ny,nz,u,v quad (two triangles), centred at the
    origin, facing +z (an upright panel) or +y (a floor)."""
    w, h = width * 0.5, height * 0.5
    if normal_axis == "z":
        corners = [(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)]
        n = (0.0, 0.0, 1.0)
    else:
        corners = [(-w, 0, h), (w, 0, h), (w, 0, -h), (-w, 0, -h)]
        n = (0.0, 1.0, 0.0)
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    order = (0, 1, 2, 0, 2, 3)
    verts = [(*corners[i], *n, *uvs[i]) for i in order]
    return np.array(verts, dtype=np.float32).reshape(-1)


def panel_matrix(panel) -> Mat4:
    """Model matrix rotating the +z quad by rotate_y and translating it."""
    model = Mat4.rotate_y(panel.rotate_y)
    model[3, 0], model[3, 1], model[3, 2] = panel.position
    return model


class SceneObject:
    """A mesh with its own uniform buffer, bind group and colour."""

    def __init__(self, device, layout, data: np.ndarray, model: Mat4, colour):
        self.vertex_buffer = device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        self.count = data.size // 8
        self.model = model
        self.colour = colour
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = device.create_bind_group(
            layout=layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
        )

    def update(self, device, view: Mat4, project: Mat4, global_tx: Mat4, alpha: float):
        mv = view @ global_tx @ self.model
        self.uniforms["MVP"] = (project @ mv).to_numpy()
        self.uniforms["MV"] = mv.to_numpy()
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        self.uniforms["colour"] = (*self.colour[:3], alpha)
        device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

    def draw(self, render_pass):
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.count)


class WebGPUScene(WebGPUWidget):
    """Three-pass weighted blended OIT with naive/sorted comparison modes."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weighted Blended OIT (WebGPU)")
        # OIT resolves a weighted average per pixel; keep every pass single
        # sampled so the accumulation and composite textures line up 1:1
        self.msaa_sample_count = 1

        # --- demo state driven by the keyboard ---
        self.mode = 3
        self.alpha = DEFAULT_ALPHA

        # --- camera / mouse state (same conventions as the GL demos) ---
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

        self.view = look_at(
            Vec3(0.0, 1.8, 8.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 350.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipelines()
        self._create_scene()
        self._create_render_buffer()
        self.update()

    # ------------------------------------------------------------------
    # pipelines
    # ------------------------------------------------------------------
    def _create_pipelines(self) -> None:
        here = Path(__file__).parent
        object_module = self.device.create_shader_module(
            code=(here / "OITShaders.wgsl").read_text()
        )
        composite_module = self.device.create_shader_module(
            code=(here / "CompositeShader.wgsl").read_text()
        )

        # explicit layout shared by the three object pipelines so one bind
        # group per object works with all of them
        self.object_bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        object_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.object_bgl]
        )
        vertex_state = {
            "module": object_module,
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
        }

        def object_pipeline(targets, depth_write, entry_point):
            return self.device.create_render_pipeline(
                layout=object_layout,
                vertex=vertex_state,
                fragment={
                    "module": object_module,
                    "entry_point": entry_point,
                    "targets": targets,
                },
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                depth_stencil={
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": depth_write,
                    "depth_compare": wgpu.CompareFunction.less,
                },
                multisample={
                    "count": 1,
                    "mask": 0xFFFFFFFF,
                    "alpha_to_coverage_enabled": False,
                },
            )

        # 1. opaque geometry: plain colour, depth write on
        self.opaque_pipeline = object_pipeline(
            [{"format": wgpu.TextureFormat.rgba8unorm}], True, "fragment_colour"
        )

        # 2. naive / sorted transparency: classic OVER blend into the opaque
        #    buffer, depth test on, depth write off
        over = {
            "src_factor": "src-alpha",
            "dst_factor": "one-minus-src-alpha",
            "operation": "add",
        }
        self.blend_pipeline = object_pipeline(
            [
                {
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "blend": {"color": dict(over), "alpha": dict(over)},
                }
            ],
            False,
            "fragment_colour",
        )

        # 3. OIT accumulation: two targets with *different* blend states,
        #    the WebGPU equivalent of glBlendFunci
        add = {"src_factor": "one", "dst_factor": "one", "operation": "add"}
        product = {
            "src_factor": "zero",
            "dst_factor": "one-minus-src",
            "operation": "add",
        }
        self.oit_pipeline = object_pipeline(
            [
                {
                    "format": wgpu.TextureFormat.rgba16float,
                    "blend": {"color": dict(add), "alpha": dict(add)},
                },
                {
                    "format": wgpu.TextureFormat.r16float,
                    "blend": {"color": dict(product), "alpha": dict(product)},
                },
            ],
            False,
            "fragment_oit",
        )

        # composite full-screen triangle (auto layout: single pipeline, and
        # its bind group is rebuilt with the textures on every resize)
        self.composite_pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": composite_module, "entry_point": "vertex_main"},
            fragment={
                "module": composite_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )
        self.composite_sampler = self.device.create_sampler(
            mag_filter="nearest", min_filter="nearest"
        )
        self.composite_params = np.zeros(4, dtype=np.uint32)
        self.composite_params_buffer = self.device.create_buffer(
            size=self.composite_params.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _create_scene(self) -> None:
        layout = self.object_bgl
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot = SceneObject(
            self.device, layout, teapot, Mat4(), (0.85, 0.8, 0.75)
        )
        self.floor = SceneObject(
            self.device, layout, quad(12.0, 12.0, "y"), Mat4(), (0.5, 0.5, 0.5)
        )
        panel_data = quad(PANEL_SIZE, PANEL_SIZE, "z")
        self.panels = [
            SceneObject(self.device, layout, panel_data, panel_matrix(p), p.colour)
            for p in PANELS
        ]

    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # render targets (called by the widget on every resize)
    # ------------------------------------------------------------------
    def _create_render_buffer(self):
        if not hasattr(self, "device"):
            return  # too early: resize during construction

        def texture(fmt, usage):
            return self.device.create_texture(
                size=self.texture_size, sample_count=1, format=fmt, usage=usage
            )

        render_and_sample = (
            wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        )
        # final image the widget copies to the QWidget surface
        self.colour_buffer_texture = texture(
            wgpu.TextureFormat.rgba8unorm,
            wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()
        # opaque scene colour + shared depth
        self.scene_colour_view = texture(
            wgpu.TextureFormat.rgba8unorm, render_and_sample
        ).create_view()
        self.depth_buffer_view = texture(
            wgpu.TextureFormat.depth24plus, wgpu.TextureUsage.RENDER_ATTACHMENT
        ).create_view()
        # the two OIT accumulation targets
        self.accum_view = texture(
            wgpu.TextureFormat.rgba16float, render_and_sample
        ).create_view()
        self.reveal_view = texture(
            wgpu.TextureFormat.r16float, render_and_sample
        ).create_view()

        self.readback_buffer = self.device.create_buffer(
            size=self._calculate_aligned_buffer_size(),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        # the composite bind group references the views, so rebuild it too
        self.composite_bind_group = self.device.create_bind_group(
            layout=self.composite_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.composite_sampler},
                {"binding": 1, "resource": self.scene_colour_view},
                {"binding": 2, "resource": self.accum_view},
                {"binding": 3, "resource": self.reveal_view},
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self.composite_params_buffer,
                        "offset": 0,
                        "size": self.composite_params_buffer.size,
                    },
                },
            ],
        )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        global_tx = self.scene_global_tx()
        self.teapot.update(self.device, self.view, self.project, global_tx, 1.0)
        tx = Mat4().translate(0.0, -0.5, 0.0)
        self.floor.update(self.device, self.view, self.project, global_tx @ tx, 1.0)
        for panel in self.panels:
            panel.update(self.device, self.view, self.project, global_tx, self.alpha)
        self.composite_params[0] = 1 if self.mode == 3 else 0
        self.device.queue.write_buffer(
            self.composite_params_buffer, 0, self.composite_params.tobytes()
        )

        command_encoder = self.device.create_command_encoder()

        # ---- pass 1: opaque geometry -> scene colour + depth ----
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_colour_view,
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
        render_pass.set_pipeline(self.opaque_pipeline)
        self.floor.draw(render_pass)
        self.teapot.draw(render_pass)
        render_pass.end()

        # ---- pass 2: the transparent panels ----
        if self.mode == 3:
            self._oit_accum_pass(command_encoder)
        else:
            self._blended_pass(command_encoder, global_tx, sort=self.mode == 2)

        # ---- pass 3: composite to the widget's colour buffer ----
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ],
        )
        render_pass.set_pipeline(self.composite_pipeline)
        render_pass.set_bind_group(0, self.composite_bind_group, [], 0, 999999)
        render_pass.draw(3)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        white = QColor(255, 255, 255)
        self.render_text(10, 20, MODE_NAMES[self.mode], 14, "Arial", white)
        self.render_text(
            10,
            45,
            f"keys 1/2/3 change mode   alpha [A/Z] {self.alpha:.2f}",
            14,
            "Arial",
            white,
        )

    def _blended_pass(self, command_encoder, global_tx: Mat4, sort: bool) -> None:
        """Modes 1 and 2: blend the panels straight into the opaque buffer,
        loading (not clearing) both the colour and the depth."""
        order = list(range(len(self.panels)))
        if sort:
            model_views = [
                (self.view @ global_tx @ p.model).to_numpy() for p in self.panels
            ]
            order = back_to_front(model_views)

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.scene_colour_view,
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.blend_pipeline)
        for i in order:
            self.panels[i].draw(render_pass)
        render_pass.end()

    def _oit_accum_pass(self, command_encoder) -> None:
        """Mode 3: accumulate all panels in ARBITRARY order into the accum
        (cleared to 0) and reveal (cleared to 1) targets, testing against
        but not writing the opaque depth."""
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.accum_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                },
                {
                    "view": self.reveal_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
                },
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.oit_pipeline)
        for panel in self.panels:  # note: scene order, no sorting anywhere
            panel.draw(render_pass)
        render_pass.end()

    def resizeWebGPU(self, width, height) -> None:
        self.project = perspective(45.0, width / height, 0.1, 350.0, PerspMode.WebGPU)
        self.update()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3):
            self.mode = int(event.text())
        elif key == Qt.Key_A:
            self.alpha = min(1.0, self.alpha + 0.05)
        elif key == Qt.Key_Z:
            self.alpha = max(0.05, self.alpha - 0.05)
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
