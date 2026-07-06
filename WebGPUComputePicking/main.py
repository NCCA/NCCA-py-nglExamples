#!/usr/bin/env -S uv run --script
"""
Object selection demo using GPU compute-based picking (WebGPU).

An alternative to the colour-ID picking used in ``SelectionManipulatorWebGPU``.
Objects still render into an offscreen ID target on click, but the ID is a
real integer written to an ``r32uint`` texture rather than a float colour,
and the readback is radically smaller: a compute shader inspects the 9x9
pixel block around the click on the GPU, atomicMin-reduces it to a single
u32 (nearest hit to the click centre wins) and the CPU maps back exactly
4 bytes instead of a whole image.

    click -> ID render pass (r32uint) -> compute reduce -> 4-byte readback

Compared with colour-ID picking this removes the float->byte ID encoding
(and its 16.7M-object ceiling and reserved-colour bookkeeping), and the
readback no longer scales with window size.

Controls:
    Left click   select object (replaces selection)
    Ctrl+click   toggle object in/out of the selection (multi-select)
    Alt+LMB      tumble camera
    Alt+RMB      pan camera
    Wheel        dolly camera
    Space        reset camera
    Escape       quit
"""

import sys

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, Vec4, look_at, perspective
from ncca.ngl.webgpu import PipelineFactory, PipelineType
from ObjectPipeline import ObjectPipeline, PickResolver
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from SelectionObject import (
    CubeObject,
    DodecahedronObject,
    SelectionObject,
    SphereObject,
    TeapotObject,
    TrollObject,
)
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device

LIGHT_POS = (0.0, 10.0, 10.0)
LIGHT_DIFFUSE = (1.0, 1.0, 1.0, 1.0)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Compute-Shader Picking (WebGPU)")

        self.view = look_at(
            Vec3(0.0, 8.0, 14.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(45.0, 1024.0 / 720.0, 0.01, 350.0, PerspMode.WebGPU)
        self.model_position = Vec3(0.0, 0.0, 0.0)

        # camera control state (Alt + mouse, Maya style)
        self.rotate_camera = False
        self.translate_camera = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        # offscreen ID targets (created in resizeWebGPU)
        self.id_texture_view = None
        self.id_depth_view = None

        self.device = get_default_device()
        self._create_render_buffer()
        self._create_id_buffer()

        self.object_pipeline = ObjectPipeline(self.device)
        self.pick_resolver = PickResolver(self.device)
        self.grid_pipeline = PipelineFactory.create_pipeline(
            self.device, PipelineType.SINGLE_COLOUR_LINES
        )

        self._build_geometry()

        self.objects: list[SelectionObject] = [
            TeapotObject(Vec3(-3.0, 0.0, 0.0), Vec4(0.8, 0.2, 0.2, 1.0)),
            CubeObject(Vec3(0.0, 0.5, -3.0), Vec4(0.2, 0.6, 0.8, 1.0)),
            SphereObject(Vec3(3.0, 1.0, 0.0), Vec4(0.9, 0.7, 0.1, 1.0)),
            TrollObject(Vec3(0.0, 0.8, 3.0), Vec4(0.6, 0.3, 0.7, 1.0)),
            DodecahedronObject(Vec3(0.0, 0.6, 0.0), Vec4(0.2, 0.7, 0.4, 1.0)),
        ]
        # the compute pass hands back an integer ID; resolve it here
        self.objects_by_id = {obj.pick_id: obj for obj in self.objects}

    # ------------------------------------------------------------------
    # geometry / buffers
    # ------------------------------------------------------------------
    def _build_geometry(self) -> None:
        for name, prim in {
            "teapot": Prims.TEAPOT,
            "cube": Prims.CUBE,
            "troll": Prims.TROLL,
            "dodecahedron": Prims.DODECAHEDRON,
        }.items():
            data = np.asarray(PrimData.primitive(prim).data, dtype=np.float32)
            self.object_pipeline.add_mesh(name, data)
        self.object_pipeline.add_mesh(
            "selSphere", np.asarray(PrimData.sphere(1.0, 40), dtype=np.float32)
        )
        self.object_pipeline.build()

        grid = np.asarray(PrimData.line_grid(20.0, 20.0, 20), dtype=np.float32).reshape(
            -1, 3
        )
        self.grid_pipeline.set_data(positions=grid)

    def _create_id_buffer(self) -> None:
        """(Re)create the integer ID target the pick pass renders into.

        r32uint, single-sampled (uint formats cannot be multisampled), with
        TEXTURE_BINDING usage so the compute kernel can textureLoad from it.
        There is no COPY_SRC: the texture never leaves the GPU.
        """
        size = self.texture_size
        id_texture = self.device.create_texture(
            size=size,
            sample_count=1,
            format=wgpu.TextureFormat.r32uint,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
            label="pick_id_texture",
        )
        self.id_texture_view = id_texture.create_view()
        depth = self.device.create_texture(
            size=size,
            sample_count=1,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            label="pick_id_depth",
        )
        self.id_depth_view = depth.create_view()

    def resizeWebGPU(self, w: int, h: int) -> None:
        self.project = perspective(
            45.0, float(w) / float(h), 0.01, 350.0, PerspMode.WebGPU
        )
        if self.device is not None:
            self._create_id_buffer()

    # ------------------------------------------------------------------
    # scene helpers
    # ------------------------------------------------------------------
    def selected_objects(self) -> list[SelectionObject]:
        return [o for o in self.objects if o.selected]

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

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
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
        mvp = (self.project @ self.view @ global_tx).to_numpy()
        self.grid_pipeline.update_uniforms(mvp=mvp, colour=(0.6, 0.6, 0.6))
        self.grid_pipeline.render(render_pass)

        self.object_pipeline.update_globals(
            self.view.to_numpy(), self.project.to_numpy(), LIGHT_POS, LIGHT_DIFFUSE
        )
        self.object_pipeline.set_instances(
            [o.instance(global_tx) for o in self.objects]
        )
        self.object_pipeline.render(render_pass)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

        self._update_colour_buffer()

        num = len(self.selected_objects())
        self.render_text(
            10,
            -18,
            f"Compute picking   selected {num}   "
            "click select, ctrl+click multi, alt+mouse camera, space reset",
            14,
            "Arial",
            QColor(255, 255, 255),
        )

    # ------------------------------------------------------------------
    # picking
    # ------------------------------------------------------------------
    def pick(self, x: float, y: float) -> SelectionObject | None:
        """Render the integer ID pass, reduce it on the GPU, return the object."""
        global_tx = self.scene_global_tx()

        # ID pass: every object flat in its integer ID, background clears to 0
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.id_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 0),
                }
            ],
            depth_stencil_attachment={
                "view": self.id_depth_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        self.object_pipeline.set_instances(
            [o.instance(global_tx) for o in self.objects]
        )
        self.object_pipeline.render_ids(render_pass)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

        # compute reduce + 4-byte readback
        pick_id = self.pick_resolver.resolve(self.id_texture_view, int(x), int(y))
        if pick_id is None:
            return None
        return self.objects_by_id.get(pick_id)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0.0, 0.0, 0.0)
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        modifiers = event.modifiers()

        # Alt + mouse drives the camera, Maya style
        if modifiers & Qt.AltModifier:
            if event.button() == Qt.LeftButton:
                self.original_x_rotation = position.x()
                self.original_y_rotation = position.y()
                self.rotate_camera = True
            elif event.button() == Qt.RightButton:
                self.original_x_pos = position.x()
                self.original_y_pos = position.y()
                self.translate_camera = True
            return

        if event.button() != Qt.LeftButton:
            return

        ratio = self.devicePixelRatio()
        obj = self.pick(position.x() * ratio, position.y() * ratio)

        if obj is not None:
            if modifiers & Qt.ControlModifier:
                obj.selected = not obj.selected
            else:
                for other in self.objects:
                    other.selected = False
                obj.selected = True
        elif not modifiers & Qt.ControlModifier:
            for other in self.objects:
                other.selected = False
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if self.rotate_camera and event.buttons() == Qt.LeftButton:
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate_camera and event.buttons() == Qt.RightButton:
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate_camera = False
        elif event.button() == Qt.RightButton:
            self.translate_camera = False
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


def main() -> None:
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
