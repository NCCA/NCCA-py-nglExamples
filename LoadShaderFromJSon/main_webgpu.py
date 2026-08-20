#!/usr/bin/env -S uv run --script
"""LoadShaderFromJSon (WebGPU): assemble a shader module at runtime from a JSON manifest.

wgpu-py has no notion of a ShaderLib-style "one file per stage" program --
a WGSL module is a single source string carrying both the `@vertex` and
`@fragment` entry points, handed straight to `device.create_shader_module`.
So the JSON-driven-assembly lesson from main.py's OpenGL version gets
reinterpreted for that shape: `shaders_webgpu/shaders.json` lists WGSL
fragments in order, `load_wgsl_from_json()` concatenates them into one
source string, and that string becomes the module. `common.wgsl` (the
`Material`/`Light`/`Transform`/`Params` uniform structs), `noise3D.wgsl`
(a syntax-only port of Ashima Arts' `snoise`) and `phong_noise.wgsl` (the
Phong vertex/fragment pass) are shared source fragments assembled the same
way `common.glsl` and `noise3D.glsl` were shared between the OpenGL
version's vertex and fragment stages.

The teapot itself is lit the same way as the OpenGL sibling: a Phong pass
whose surface colour comes from six octaves of 3D simplex noise, giving
the gold material a mottled, marble-like look that shifts as `time`
advances.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

# std140-style uniform blocks, one buffer per WGSL struct in common.wgsl --
# field names/order mirror the struct layouts there exactly, including the
# explicit padding fields WGSL's own alignment rules already imply but
# numpy doesn't insert on its own.
_TRANSFORM_DTYPE = np.dtype(
    [
        ("m", np.float32, (4, 4)),
        ("mvp", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("viewer_pos", np.float32, 3),
        ("_pad0", np.float32, 1),
    ]
)
# Material's WGSL layout isn't fully sequential: `_pad0: vec3<f32>` after
# `shininess: f32` has a 16-byte alignment requirement of its own, which
# pushes it from offset 52 to offset 64 -- a 12-byte gap numpy won't add
# unless told to, hence the explicit offsets/itemsize below (struct size
# 80, not the naively-packed 64).
_MATERIAL_DTYPE = np.dtype(
    {
        "names": ["ambient", "diffuse", "specular", "shininess", "_pad0"],
        "formats": [
            (np.float32, 4),
            (np.float32, 4),
            (np.float32, 4),
            np.float32,
            (np.float32, 3),
        ],
        "offsets": [0, 16, 32, 48, 64],
        "itemsize": 80,
    }
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
_PARAMS_DTYPE = np.dtype(
    [
        ("time", np.float32, 1),
        ("repeat", np.float32, 1),
        ("_pad0", np.float32, 2),
    ]
)


def load_wgsl_from_json(json_path: Path, device) -> "wgpu.GPUShaderModule":
    """Build one WGSL shader module from a JSON manifest of source fragments.

    Parameters
    ----------
        json_path : Path
            Path to the manifest, e.g. `shaders_webgpu/shaders.json`.
        device : wgpu.GPUDevice
            The device the module is created against.

    Returns
    -------
        wgpu.GPUShaderModule
            The compiled module, ready to hand to `create_render_pipeline`.
    """
    data = json.loads(json_path.read_text())
    module = data["ShaderModule"]
    source = "\n".join((json_path.parent / p).read_text() for p in module["path"])
    return device.create_shader_module(code=source)


class WebGPUScene(WebGPUWidget):
    """A Phong-lit teapot whose surface colour is perturbed by layered simplex noise."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LoadShaderFromJSon (WebGPU)")
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

        # --- animation / noise state ---
        self.time = 0.0
        # Starts clean at 0.1 -- the OpenGL sibling's first 1/2 keypress
        # jumps repeat from 0.01 to ~0.1 because the C++ source sets its
        # initial value in two disconnected places (see main.py's comment
        # and the README). There's no equivalent second init site here, so
        # there's nothing to reproduce.
        self.repeat = 0.1

        self.eye = Vec3(0.0, 1.0, 2.0)
        self.view = look_at(self.eye, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.05, 350.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(20)

        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_module = load_wgsl_from_json(
            Path(__file__).parent / "shaders_webgpu" / "shaders.json", self.device
        )

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
                for i in range(4)
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.pipeline = self.device.create_render_pipeline(
            label="noise_shader_pipeline",
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
        # Per-frame uniforms (transform, params) get their own COPY_DST
        # buffers, rewritten every paint/tick. Material and light are set
        # once here and never change.
        self.transform_uniforms = np.zeros((), dtype=_TRANSFORM_DTYPE)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="noise_shader_transform_uniforms",
        )

        material = np.zeros((), dtype=_MATERIAL_DTYPE)
        material["ambient"] = (0.274725, 0.1995, 0.0745, 0.0)
        material["diffuse"] = (0.75164, 0.60648, 0.22648, 0.0)
        material["specular"] = (0.628281, 0.555802, 0.3666065, 0.0)
        material["shininess"] = 51.2
        self.material_buffer = self.device.create_buffer_with_data(
            data=material.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="noise_shader_material_uniforms",
        )

        light = np.zeros((), dtype=_LIGHT_DTYPE)
        light["position"] = (-2.0, 5.0, 2.0)
        light["ambient"] = (0.0, 0.0, 0.0, 1.0)
        light["diffuse"] = (1.0, 1.0, 1.0, 1.0)
        light["specular"] = (0.8, 0.8, 0.8, 1.0)
        self.light_buffer = self.device.create_buffer_with_data(
            data=light.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
            label="noise_shader_light_uniforms",
        )

        self.params_uniforms = np.zeros((), dtype=_PARAMS_DTYPE)
        self.params_buffer = self.device.create_buffer(
            size=self.params_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="noise_shader_params_uniforms",
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
                        "buffer": self.material_buffer,
                        "offset": 0,
                        "size": self.material_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.light_buffer,
                        "offset": 0,
                        "size": self.light_buffer.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.params_buffer,
                        "offset": 0,
                        "size": self.params_buffer.size,
                    },
                },
            ],
        )

        # PrimData.primitive() gives a flat float32 array, 8 floats per
        # vertex (pos3, normal3, uv2) -- one draw call, no per-draw pool.
        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.vertex_count = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

    def _on_tick(self) -> None:
        self.time += 0.01
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

        self.transform_uniforms["m"] = global_tx.to_numpy()
        self.transform_uniforms["mvp"] = mvp.to_numpy()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.transform_uniforms["viewer_pos"] = self.eye.to_numpy()
        self.device.queue.write_buffer(
            self.transform_buffer, 0, self.transform_uniforms.tobytes()
        )

        self.params_uniforms["time"] = self.time
        self.params_uniforms["repeat"] = self.repeat
        self.device.queue.write_buffer(
            self.params_buffer, 0, self.params_uniforms.tobytes()
        )

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
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.05, 350.0, PerspMode.WebGPU
        )
        self.update()

    # ------------------------------------------------------------------
    # input (hand-copied from ShadedGrid/main_webgpu.py -- no shared mixin
    # exists for QWidget-based WebGPU demos)
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        elif key == Qt.Key_1:
            self.repeat -= 0.01
        elif key == Qt.Key_2:
            self.repeat += 0.01
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
        # Stop the timer before the base class tears down the wgpu
        # surface/device -- otherwise a queued tick can fire a GPU call
        # after teardown and crash. Same fix as ShadedGrid's closeEvent.
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions from Qt event handlers instead of swallowing them."""

    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
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

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)

    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
