#!/usr/bin/env -S uv run --script
"""
Alpha blending and transparency (WebGPU).

The same scene and keyboard toggles as the OpenGL version (main.py) built on
wgpu-py. The important difference to notice: OpenGL blending is *dynamic*
state (glEnable / glBlendFunc can change per draw call), but in WebGPU the
blend state is baked into the render pipeline at creation time. Toggling a
blend mode here means switching to a different pre-built pipeline, which is
why this demo builds a small cache of pipeline variants.

Controls:
    B  toggle blending          D  toggle depth write for the panels
    O  toggle back-to-front sorting
    F  cycle blend preset (over / additive / premultiplied / multiply)
    A/Z increase / decrease panel alpha
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from blend_scene import DEFAULT_ALPHA, PANEL_SIZE, PANELS, back_to_front
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

# Blend presets cycled with F, expressed as WebGPU blend components.
# (label, colour blend, alpha blend)
BLEND_PRESETS = (
    (
        "OVER: src-alpha, one-minus-src-alpha",
        {
            "src_factor": "src-alpha",
            "dst_factor": "one-minus-src-alpha",
            "operation": "add",
        },
    ),
    (
        "ADDITIVE: src-alpha, one",
        {"src_factor": "src-alpha", "dst_factor": "one", "operation": "add"},
    ),
    (
        "PREMULTIPLIED: one, one-minus-src-alpha",
        {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
    ),
    (
        "MULTIPLY: dst, zero",
        {"src_factor": "dst", "dst_factor": "zero", "operation": "add"},
    ),
)

# std140-style uniform block: MVP, normal matrix (as mat4), colour
UNIFORM_DTYPE = np.dtype(
    [
        ("MVP", np.float32, (4, 4)),
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
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        self.uniforms["colour"] = (*self.colour[:3], alpha)
        device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

    def draw(self, render_pass):
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.count)


class WebGPUScene(WebGPUWidget):
    """Transparent panel scene with togglable blend / depth / sort state."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blending and Transparency (WebGPU)")
        self.msaa_sample_count = 4

        # --- demo state driven by the keyboard ---
        self.blend_enabled = True
        self.depth_write = False
        self.sort_panels = True
        self.preset = 0
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
            Vec3(0.0, 1.5, 7.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0)
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
        shader_src = (Path(__file__).parent / "BlendShader.wgsl").read_text()
        self.shader_module = self.device.create_shader_module(code=shader_src)
        # An explicit bind group layout shared by every pipeline variant, so
        # one bind group per object works with all of them ("auto" layouts
        # are unique per pipeline and would not be interchangeable).
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )
        # cache of pipeline variants keyed on (blend preset | None, depth_write)
        self._pipelines: dict = {}

    def _pipeline(self, preset: int | None, depth_write: bool):
        """Get (or lazily build) the pipeline variant for this blend state.

        preset None means blending disabled. This cache is the WebGPU
        equivalent of the glEnable(GL_BLEND)/glBlendFunc/glDepthMask calls in
        the OpenGL version of the demo.
        """
        key = (preset, depth_write)
        if key in self._pipelines:
            return self._pipelines[key]

        target: dict = {"format": wgpu.TextureFormat.rgba8unorm}
        if preset is not None:
            blend = BLEND_PRESETS[preset][1]
            target["blend"] = {"color": dict(blend), "alpha": dict(blend)}

        pipeline = self.device.create_render_pipeline(
            label=f"blend_pipeline_{key}",
            layout=self.pipeline_layout,
            vertex={
                "module": self.shader_module,
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
                "module": self.shader_module,
                "entry_point": "fragment_main",
                "targets": [target],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_write,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )
        self._pipelines[key] = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _create_scene(self) -> None:
        layout = self.bind_group_layout
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot = SceneObject(
            self.device, layout, teapot, Mat4(), (0.85, 0.8, 0.75)
        )
        self.floor = SceneObject(
            self.device, layout, quad(12.0, 12.0, "y"), Mat4(), (0.5, 0.5, 0.5)
        )
        panel_data = quad(PANEL_SIZE, PANEL_SIZE, "z")
        self.panels = []
        for panel in PANELS:
            model = Mat4()
            model[3, 0], model[3, 1], model[3, 2] = panel.position
            self.panels.append(
                SceneObject(self.device, layout, panel_data, model, panel.colour)
            )

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
        # for obj in (self.teapot, self.floor):
        self.teapot.update(self.device, self.view, self.project, global_tx, 1.0)
        tx = Mat4().translate(0, -0.5, 0.0)
        self.floor.update(self.device, self.view, self.project, global_tx @ tx, 1.0)
        for panel in self.panels:
            panel.update(self.device, self.view, self.project, global_tx, self.alpha)

        # panel draw order: scene order, or sorted back to front
        order = list(range(len(self.panels)))
        if self.sort_panels:
            model_views = [
                (self.view @ global_tx @ p.model).to_numpy() for p in self.panels
            ]
            order = back_to_front(model_views)

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
        # 1. opaque pass: depth write on, no blending
        render_pass.set_pipeline(self._pipeline(None, True))
        self.floor.draw(render_pass)
        self.teapot.draw(render_pass)

        # 2. transparent pass: switch pipeline variant to change blend state
        preset = self.preset if self.blend_enabled else None
        render_pass.set_pipeline(self._pipeline(preset, self.depth_write))
        for i in order:
            self.panels[i].draw(render_pass)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()
        self._draw_hud()

    def _draw_hud(self) -> None:
        white = QColor(255, 255, 255)
        state = (
            f"[B]lend {'ON' if self.blend_enabled else 'OFF'}  "
            f"[D]epth write {'ON' if self.depth_write else 'OFF'}  "
            f"[O]rder {'sorted back-to-front' if self.sort_panels else 'scene order'}  "
            f"alpha [A/Z] {self.alpha:.2f}"
        )
        self.render_text(10, 20, state, 14, "Arial", white)
        self.render_text(
            10, 45, f"[F] blend {BLEND_PRESETS[self.preset][0]}", 14, "Arial", white
        )

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
        elif key == Qt.Key_B:
            self.blend_enabled = not self.blend_enabled
        elif key == Qt.Key_D:
            self.depth_write = not self.depth_write
        elif key == Qt.Key_O:
            self.sort_panels = not self.sort_panels
        elif key == Qt.Key_F:
            self.preset = (self.preset + 1) % len(BLEND_PRESETS)
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
