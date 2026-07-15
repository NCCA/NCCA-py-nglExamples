#!/usr/bin/env -S uv run --script
"""A WebGPU version of the PBR texture demo.

This renders the same scene as the OpenGL ``main.py`` - a grid of teapots with
random metallic/roughness texture sets, a textured floor and four toggleable
lights - but built on wgpu-py. The interesting part is ``texture_pack_webgpu``,
which replaces the OpenGL texture-unit model with a bind group per material.

Controls: arrow keys move the camera, left mouse rotates, the wheel zooms,
1-4 toggle the lights, L toggles the light spheres, R reseeds the layout,
Space resets the camera and Escape quits.
"""

import argparse
import math
import random
import sys
import traceback

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import (
    FirstPersonCamera,
    PerspMode,
    PrimData,
    Prims,
    Random,
    Transform,
    Vec3,
)
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from texture_pack_webgpu import TexturePack

_FLOAT = np.dtype(np.float32).itemsize
_VERTEX_STRIDE = 8 * _FLOAT  # pos3, normal3, uv2
# Dynamic uniform offsets must be a multiple of the device alignment; 256 is the
# maximum any device requires, so it is always a safe stride to use.
_DYNAMIC_STRIDE = 256
_NUM_LIGHTS = 4

# One padded slot of per-object transforms (matches the Transforms struct).
_TRANSFORM_DTYPE = np.dtype(
    {
        "names": ["MVP", "M", "normalMatrix", "texRot"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
        ],
        "offsets": [0, 64, 128, 192],
        "itemsize": _DYNAMIC_STRIDE,
    }
)

# Lights + camera, shared by every teapot in a frame (matches the Lights struct).
_SCENE_DTYPE = np.dtype(
    {
        "names": ["lightPositions", "lightColors", "camPos"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), (np.float32, 4)],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)

# One padded slot for a light-indicator sphere (matches the LightSphere struct).
_LIGHT_SPHERE_DTYPE = np.dtype(
    {
        "names": ["MVP", "colour"],
        "formats": [(np.float32, (4, 4)), (np.float32, 4)],
        "offsets": [0, 64],
        "itemsize": _DYNAMIC_STRIDE,
    }
)

_LIGHT_POSITIONS = [
    (-5.0, 4.0, -5.0),
    (5.0, 4.0, -5.0),
    (-5.0, 4.0, 5.0),
    (5.0, 4.0, 5.0),
]
_LIGHT_COLOUR = (250.0, 250.0, 250.0)
_MOVE_KEYS = {Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down}


class PBRTextureScene(WebGPUWidget):
    """WebGPU widget rendering the textured PBR teapot grid."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WebGPU PBR Texture Demo")
        self.msaa_sample_count = 4

        self.seed = 12345
        self.light_on = [True, True, True, True]
        self.show_lights = True
        self.keys_pressed: set = set()

        self.rotate = False
        self.original_x_rotation = 0
        self.original_y_rotation = 0

        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 0.0

        self.camera = FirstPersonCamera(
            Vec3(0, 5, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
        )
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        self.camera.set_projection(45.0, width / height, 0.05, 350.0, PerspMode.WebGPU)

        self._initialize_web_gpu()
        self.startTimer(16)
        self.update()

    # ------------------------------------------------------------------ setup
    def _initialize_web_gpu(self) -> None:
        try:
            self.device = wgpu.utils.get_default_device()
            self._create_render_buffer()
            self._load_geometry()
            self._create_pbr_pipeline()
            self._create_light_pipeline()
            self.texture_pack = TexturePack(self.device, self.material_bgl)
            self.texture_pack.load_json("textures/textures.json")
            self._build_scene()
            self._create_transform_buffer()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            traceback.print_exc()
            raise

    def _load_geometry(self) -> None:
        """Upload the teapot, floor and light-sphere meshes as vertex buffers."""
        teapot = PrimData.primitive(Prims.TEAPOT.value).astype(np.float32)
        floor = (
            PrimData.triangle_plane(30, 30, 10, 10, Vec3(0, 1, 0))
            .astype(np.float32)
            .flatten()
        )
        sphere = PrimData.sphere(0.5, 20).astype(np.float32).flatten()

        self.geometry = {
            "teapot": self._make_vertex_buffer(teapot),
            "floor": self._make_vertex_buffer(floor),
            "sphere": self._make_vertex_buffer(sphere),
        }

    def _make_vertex_buffer(self, data: np.ndarray) -> dict:
        buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        return {"buffer": buffer, "count": data.size // 8}

    def _create_pbr_pipeline(self) -> None:
        """Create the textured PBR pipeline and its bind group layouts."""
        with open("PBRTexture.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        # @group(0) per-object transforms, addressed with a dynamic offset.
        self.transform_bgl = self.device.create_bind_group_layout(
            label="transform_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "has_dynamic_offset": True,
                    },
                }
            ],
        )
        # @group(1) lights + camera, shared across the frame.
        self.scene_bgl = self.device.create_bind_group_layout(
            label="scene_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        # @group(2) the active material's five maps + a sampler.
        material_entries = [
            {
                "binding": i,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            }
            for i in range(5)
        ]
        material_entries.append(
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            }
        )
        self.material_bgl = self.device.create_bind_group_layout(
            label="material_bgl", entries=material_entries
        )

        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.transform_bgl, self.scene_bgl, self.material_bgl]
        )
        self.pbr_pipeline = self.device.create_render_pipeline(
            label="pbr_texture_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": _VERTEX_STRIDE,
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
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        # Scene (lights + camera) uniform buffer and its bind group.
        self.scene_uniforms = np.zeros((), dtype=_SCENE_DTYPE)
        for i, pos in enumerate(_LIGHT_POSITIONS):
            self.scene_uniforms["lightPositions"][i] = (*pos, 1.0)
        self.scene_buffer = self.device.create_buffer(
            size=self.scene_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="scene_uniform_buffer",
        )
        self.scene_bind_group = self.device.create_bind_group(
            layout=self.scene_bgl,
            entries=[{"binding": 0, "resource": {"buffer": self.scene_buffer}}],
        )

    def _create_light_pipeline(self) -> None:
        """Create the unlit solid-colour pipeline for the light spheres."""
        with open("LightSphere.wgsl", "r") as f:
            shader = self.device.create_shader_module(code=f.read())

        self.light_bgl = self.device.create_bind_group_layout(
            label="light_sphere_bgl",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                        "has_dynamic_offset": True,
                    },
                }
            ],
        )
        layout = self.device.create_pipeline_layout(bind_group_layouts=[self.light_bgl])
        self.light_pipeline = self.device.create_render_pipeline(
            label="light_sphere_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": _VERTEX_STRIDE,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0}
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        self.light_sphere_uniforms = np.zeros(_NUM_LIGHTS, dtype=_LIGHT_SPHERE_DTYPE)
        self.light_sphere_buffer = self.device.create_buffer(
            size=self.light_sphere_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="light_sphere_buffer",
        )
        self.light_sphere_bind_group = self.device.create_bind_group(
            layout=self.light_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.light_sphere_buffer,
                        "offset": 0,
                        "size": _DYNAMIC_STRIDE,
                    },
                }
            ],
        )

    def _build_scene(self) -> None:
        """Lay out the teapot grid and floor for the current seed.

        Mirrors the OpenGL demo: a seeded RNG chooses a material, Y-rotation and
        texture rotation per teapot so the layout is stable within a seed but
        reshuffles when R is pressed.
        """
        Random.set_seed_value(self.seed)
        random.seed(self.seed)
        materials = self.texture_pack.materials or [
            "copper",
            "greasy",
            "panel",
            "rusty",
            "wood",
        ]

        tx = Transform()
        self.scene_objects = []  # (geometry key, model Mat4, texRot vec4, material)
        for row in np.arange(-10, 10, 1.6):
            for col in np.arange(-10, 10, 1.6):
                material = random.choice(materials)
                angle = Random.random_positive_number() * 360.0
                tx.reset()
                tx.set_position(float(col), 0.0, float(row))
                tx.set_rotation(0.0, angle, 0.0)
                model = tx.matrix()
                tex_rot = self._texture_rotation()
                self.scene_objects.append(("teapot", model, tex_rot, material))

        tx.reset()
        tx.set_position(0.0, -0.5, 0.0)
        self.scene_objects.append(
            ("floor", tx.matrix(), self._texture_rotation(), "greasy")
        )

    @staticmethod
    def _texture_rotation() -> np.ndarray:
        """A column-major mat2 rotation packed as (c, s, -s, c)."""
        theta = math.radians(Random.random_number(180.0))
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c, s, -s, c], dtype=np.float32)

    def _create_transform_buffer(self) -> None:
        """Allocate the per-object transforms buffer (one padded slot each)."""
        count = len(self.scene_objects)
        self.transform_uniforms = np.zeros(count, dtype=_TRANSFORM_DTYPE)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transform_buffer",
        )
        self.transform_bind_group = self.device.create_bind_group(
            layout=self.transform_bgl,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.transform_buffer,
                        "offset": 0,
                        "size": _DYNAMIC_STRIDE,
                    },
                }
            ],
        )

    # ------------------------------------------------------------------- paint
    def paintWebGPU(self) -> None:
        if not hasattr(self, "pbr_pipeline"):
            return
        current = self.timer.elapsed() * 0.001
        delta_time = min(current - self.last_frame, 0.05)
        self.last_frame = current
        self._update_camera_movement(delta_time)

        self._write_scene_uniforms()
        self._write_transform_uniforms()

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

        self._draw_scene(render_pass)
        if self.show_lights:
            self._draw_lights(render_pass)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        if self.keys_pressed & _MOVE_KEYS:
            self.update()

    def _write_scene_uniforms(self) -> None:
        colour_on = np.array((*_LIGHT_COLOUR, 1.0), dtype=np.float32)
        colour_off = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        for i in range(_NUM_LIGHTS):
            self.scene_uniforms["lightColors"][i] = (
                colour_on if self.light_on[i] else colour_off
            )
        self.scene_uniforms["camPos"] = (
            self.camera.eye.x,
            self.camera.eye.y,
            self.camera.eye.z,
            1.0,
        )
        self.device.queue.write_buffer(
            self.scene_buffer, 0, self.scene_uniforms.tobytes()
        )

    def _write_transform_uniforms(self) -> None:
        view = self.camera.view
        projection = self.camera.projection
        for i, (_, model, tex_rot, _) in enumerate(self.scene_objects):
            model_view = view @ model
            mvp = projection @ model_view
            normal_matrix = model_view.copy().inverse().transposed()
            slot = self.transform_uniforms[i]
            slot["MVP"] = mvp.to_numpy()
            slot["M"] = model.to_numpy()
            slot["normalMatrix"] = normal_matrix.to_numpy()
            slot["texRot"] = tex_rot
        self.device.queue.write_buffer(
            self.transform_buffer, 0, self.transform_uniforms.tobytes()
        )

    def _draw_scene(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
        render_pass.set_pipeline(self.pbr_pipeline)
        render_pass.set_bind_group(1, self.scene_bind_group)
        for i, (geom_key, _, _, material) in enumerate(self.scene_objects):
            render_pass.set_bind_group(
                0, self.transform_bind_group, [i * _DYNAMIC_STRIDE]
            )
            self.texture_pack.activate(render_pass, material)
            geometry = self.geometry[geom_key]
            render_pass.set_vertex_buffer(0, geometry["buffer"])
            render_pass.draw(geometry["count"])

    def _draw_lights(self, render_pass: wgpu.GPURenderPassEncoder) -> None:
        white = np.array((1.0, 1.0, 1.0, 1.0), dtype=np.float32)
        tx = Transform()
        for i, pos in enumerate(_LIGHT_POSITIONS):
            tx.reset()
            tx.set_position(*pos)
            mvp = self.camera.projection @ self.camera.view @ tx.matrix()
            self.light_sphere_uniforms[i]["MVP"] = mvp.to_numpy()
            self.light_sphere_uniforms[i]["colour"] = white
        self.device.queue.write_buffer(
            self.light_sphere_buffer, 0, self.light_sphere_uniforms.tobytes()
        )

        sphere = self.geometry["sphere"]
        render_pass.set_pipeline(self.light_pipeline)
        for i in range(_NUM_LIGHTS):
            render_pass.set_bind_group(
                0, self.light_sphere_bind_group, [i * _DYNAMIC_STRIDE]
            )
            render_pass.set_vertex_buffer(0, sphere["buffer"])
            render_pass.draw(sphere["count"])

    # ------------------------------------------------------------------- input
    def resizeWebGPU(self, width: int, height: int) -> None:
        ratio = self.devicePixelRatio()
        w = max(width * ratio, 1)
        h = max(height * ratio, 1)
        self.camera.set_projection(45.0, w / h, 0.05, 350.0, PerspMode.WebGPU)
        self.update()

    def _update_camera_movement(self, delta_time: float) -> None:
        x_direction = 0.0
        y_direction = 0.0
        for key in self.keys_pressed:
            if key == Qt.Key_Left:
                y_direction = -1.0
            elif key == Qt.Key_Right:
                y_direction = 1.0
            elif key == Qt.Key_Up:
                x_direction = 1.0
            elif key == Qt.Key_Down:
                x_direction = -1.0
        if x_direction != 0.0 or y_direction != 0.0:
            self.camera.move(x_direction, y_direction, delta_time)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        self.keys_pressed.add(key)
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_R:
            self.seed = random.randint(0, 1000000)
            self._build_scene()
        elif key == Qt.Key_Space:
            self.camera = FirstPersonCamera(
                Vec3(0, 5, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
            )
            width = max(self.width(), 1)
            height = max(self.height(), 1)
            self.camera.set_projection(
                45.0, width / height, 0.05, 350.0, PerspMode.WebGPU
            )
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            self.light_on[key - Qt.Key_1] ^= True
        elif key == Qt.Key_L:
            self.show_lights ^= True
        self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self.keys_pressed.discard(event.key())
        self.update()
        super().keyReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.camera.process_mouse_movement(diff_x, -diff_y)
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            position = event.position()
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        num_pixels = event.angleDelta().y()
        self.camera.process_mouse_scroll(num_pixels * 0.01)
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
    win = PBRTextureScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
