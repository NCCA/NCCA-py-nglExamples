#!/usr/bin/env -S uv run --script
"""
Object selection and manipulation demo (WebGPU).

A WebGPU port of the OpenGL ``SelectionManipulator`` demo. A small scene of
SelectionObject-derived objects can be picked with the mouse (colour-ID
picking, read back from an offscreen texture) and transformed with Maya-style
gizmos.

Rendering uses the stock ``ncca.ngl.webgpu`` factory pipelines wherever they
fit and one small custom pipeline for the objects:

    grid    -> factory SINGLE_COLOUR_LINES
    gizmo   -> factory SINGLE_COLOUR_TRIANGLES (one pass per part, on top)
    picking -> objects via the object pipeline's flat pick mode,
               gizmo handles via factory SINGLE_COLOUR_TRIANGLES
    objects -> ObjectPipeline (diffuse + single-pass barycentric wireframe)

Controls (matching Maya):
    Q            select mode (no gizmo)
    W            translate mode
    E            rotate mode
    R            scale mode
    Left click   select object (replaces selection)
    Ctrl+click   toggle object in/out of the selection (multi-select)
    Drag handle  transform every selected object
    Alt+LMB      tumble camera
    Alt+RMB      pan camera
    Wheel        dolly camera
    Space        reset camera
    Escape       quit
"""

import sys

import numpy as np
import wgpu
from Manipulator import CENTER, Axis, ManipMode, Manipulator
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, Vec4, look_at, perspective
from ncca.ngl.webgpu import PipelineFactory, PipelineType
from ObjectPipeline import ObjectPipeline
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

# side of the square pixel block read back when picking; a few pixels of
# slop makes the thin gizmo handles much easier to grab
PICK_BLOCK = 9

LIGHT_POS = (0.0, 10.0, 10.0)
LIGHT_DIFFUSE = (1.0, 1.0, 1.0, 1.0)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Selection and Manipulation (WebGPU)")

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

        self.mode = ManipMode.TRANSLATE
        self.dragging = False

        # offscreen picking targets (created in resizeWebGPU)
        self.pick_msaa_view = None
        self.pick_resolve_texture = None
        self.pick_resolve_view = None
        self.pick_depth_view = None
        self.pick_readback = None

        self.device = get_default_device()
        self._create_render_buffer()
        self._create_pick_buffer()

        self.object_pipeline = ObjectPipeline(self.device)
        self.grid_pipeline = PipelineFactory.create_pipeline(
            self.device, PipelineType.SINGLE_COLOUR_LINES
        )
        self.gizmo_pipeline = PipelineFactory.create_pipeline(
            self.device,
            PipelineType.SINGLE_COLOUR_TRIANGLES,
            data_type="Vec3",
            stride=32,
        )
        self.manipulator = Manipulator(self.device)

        self._build_geometry()

        self.objects: list[SelectionObject] = [
            TeapotObject(Vec3(-3.0, 0.0, 0.0), Vec4(0.8, 0.2, 0.2, 1.0)),
            CubeObject(Vec3(0.0, 0.5, -3.0), Vec4(0.2, 0.6, 0.8, 1.0)),
            SphereObject(Vec3(3.0, 1.0, 0.0), Vec4(0.9, 0.7, 0.1, 1.0)),
            TrollObject(Vec3(0.0, 0.8, 3.0), Vec4(0.6, 0.3, 0.7, 1.0)),
            DodecahedronObject(Vec3(0.0, 0.6, 0.0), Vec4(0.2, 0.7, 0.4, 1.0)),
        ]

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

    def _create_pick_buffer(self) -> None:
        size = self.texture_size
        msaa = self.device.create_texture(
            size=size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.pick_msaa_view = msaa.create_view()
        self.pick_resolve_texture = self.device.create_texture(
            size=size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.pick_resolve_view = self.pick_resolve_texture.create_view()
        depth = self.device.create_texture(
            size=size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.pick_depth_view = depth.create_view()
        self.pick_readback = self.device.create_buffer(
            size=self._calculate_aligned_buffer_size(),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def resizeWebGPU(self, w: int, h: int) -> None:
        self.project = perspective(
            45.0, float(w) / float(h), 0.01, 350.0, PerspMode.WebGPU
        )
        if self.device is not None:
            self._create_pick_buffer()

    # ------------------------------------------------------------------
    # selection helpers
    # ------------------------------------------------------------------
    def selected_objects(self) -> list[SelectionObject]:
        return [o for o in self.objects if o.selected]

    def update_pivot(self) -> None:
        selected = self.selected_objects()
        if not selected:
            return
        centre = Vec3(0.0, 0.0, 0.0)
        for obj in selected:
            centre += obj.position
        self.manipulator.position = centre * (1.0 / len(selected))

    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # render pass helpers
    # ------------------------------------------------------------------
    def _begin_pass(
        self,
        encoder,
        colour_view,
        resolve_view,
        depth_view,
        clear_colour: bool,
        clear_depth: bool,
    ):
        return encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": colour_view,
                    "resolve_target": resolve_view,
                    "load_op": wgpu.LoadOp.clear if clear_colour else wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_load_op": wgpu.LoadOp.clear if clear_depth else wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )

    def _draw_gizmo_parts(self, draws, colour_view, resolve_view, depth_view) -> None:
        """Draw each gizmo part on top of the scene (clears depth on the first)."""
        for i, (buf, _count, mvp, colour) in enumerate(draws):
            encoder = self.device.create_command_encoder()
            render_pass = self._begin_pass(
                encoder,
                colour_view,
                resolve_view,
                depth_view,
                clear_colour=False,
                clear_depth=(i == 0),
            )
            self.gizmo_pipeline.set_data(positions=buf)
            self.gizmo_pipeline.update_uniforms(mvp=mvp, colour=colour)
            self.gizmo_pipeline.render(render_pass)
            render_pass.end()
            self.device.queue.submit([encoder.finish()])

    def paintWebGPU(self) -> None:
        global_tx = self.scene_global_tx()
        view_np = self.view.to_numpy()
        proj_np = self.project.to_numpy()

        # pass 1: grid + objects, clearing the frame
        encoder = self.device.create_command_encoder()
        render_pass = self._begin_pass(
            encoder,
            self.multisample_texture_view,
            self.colour_buffer_texture_view,
            self.depth_buffer_view,
            clear_colour=True,
            clear_depth=True,
        )
        mvp = (self.project @ self.view @ global_tx).to_numpy()
        self.grid_pipeline.update_uniforms(mvp=mvp, colour=(0.6, 0.6, 0.6))
        self.grid_pipeline.render(render_pass)

        self.object_pipeline.update_globals(
            view_np, proj_np, LIGHT_POS, LIGHT_DIFFUSE, 0
        )
        self.object_pipeline.set_instances(
            [o.instance(global_tx) for o in self.objects]
        )
        self.object_pipeline.render(render_pass)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

        # gizmo on top
        if self.selected_objects() and self.mode != ManipMode.SELECT:
            draws = self.manipulator.part_draws(
                self.mode, global_tx, self.view, self.project
            )
            self._draw_gizmo_parts(
                draws,
                self.multisample_texture_view,
                self.colour_buffer_texture_view,
                self.depth_buffer_view,
            )

        self._update_colour_buffer()

        num = len(self.selected_objects())
        self.render_text(
            10,
            -18,
            f"Mode [{self.mode.value}]  selected {num}   "
            "Q select W translate E rotate R scale | click select, ctrl+click multi, alt+mouse camera",
            14,
            "Arial",
            QColor(255, 255, 255),
        )

    # ------------------------------------------------------------------
    # picking
    # ------------------------------------------------------------------
    def pick(self, x: float, y: float):
        """Render the ID pass and return ('axis', Axis) or ('object', obj) or None."""
        global_tx = self.scene_global_tx()
        view_np = self.view.to_numpy()
        proj_np = self.project.to_numpy()

        # objects flat in their ID colour (clears black)
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.pick_msaa_view,
                    "resolve_target": self.pick_resolve_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.pick_depth_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        self.object_pipeline.update_globals(
            view_np, proj_np, LIGHT_POS, LIGHT_DIFFUSE, 1
        )
        self.object_pipeline.set_instances(
            [o.instance(global_tx) for o in self.objects]
        )
        self.object_pipeline.render(render_pass)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

        # gizmo handles in their reserved ID colours, on top
        if self.selected_objects() and self.mode != ManipMode.SELECT:
            draws = self.manipulator.pick_draws(
                self.mode, global_tx, self.view, self.project
            )
            self._draw_gizmo_parts(
                draws, self.pick_msaa_view, self.pick_resolve_view, self.pick_depth_view
            )
        # restore shade mode for the next visible frame
        self.object_pipeline.update_globals(
            view_np, proj_np, LIGHT_POS, LIGHT_DIFFUSE, 0
        )

        pixels = self._read_pick_block(int(x), int(y))
        if not pixels:
            return None
        centre = pixels[len(pixels) // 2]

        # gizmo handles win over objects so they stay grabbable in front of geometry
        for pixel in pixels:
            axis = Manipulator.axis_for_pick_colour(pixel)
            if axis is not None:
                return ("axis", axis)
        for pixel in [centre] + pixels:
            for obj in self.objects:
                if obj.matches_colour_id(pixel):
                    return ("object", obj)
        return None

    def _read_pick_block(self, x: int, y: int):
        """Copy the pick texture back and return the PICK_BLOCK pixels around (x, y)."""
        width, height = self.texture_size
        bytes_per_row = self._calculate_aligned_row_size()
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self.pick_resolve_texture},
            {
                "buffer": self.pick_readback,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        self.device.queue.submit([encoder.finish()])
        self.pick_readback.map_sync(mode=wgpu.MapMode.READ)
        raw = self.pick_readback.read_mapped()
        image = np.lib.stride_tricks.as_strided(
            np.frombuffer(raw, dtype=np.uint8),
            shape=(height, width, 4),
            strides=(bytes_per_row, 4, 1),
        )
        half = PICK_BLOCK // 2
        pixels = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px = min(max(x + dx, 0), width - 1)
                py = min(max(y + dy, 0), height - 1)
                pixels.append(tuple(int(c) for c in image[py, px, :3]))
        result = list(pixels)
        self.pick_readback.unmap()
        return result

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Q:
            self.mode = ManipMode.SELECT
        elif key == Qt.Key_W:
            self.mode = ManipMode.TRANSLATE
        elif key == Qt.Key_E:
            self.mode = ManipMode.ROTATE
        elif key == Qt.Key_R:
            self.mode = ManipMode.SCALE
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
        hit = self.pick(position.x() * ratio, position.y() * ratio)

        if hit is not None and hit[0] == "axis" and self.mode != ManipMode.SELECT:
            self.manipulator.start_drag(
                hit[1],
                position.x() * ratio,
                position.y() * ratio,
                self.scene_global_tx(),
                self.view,
                self.project,
                self.texture_size[0],
                self.texture_size[1],
            )
            self.dragging = True
        elif hit is not None and hit[0] == "object":
            obj = hit[1]
            if modifiers & Qt.ControlModifier:
                obj.selected = not obj.selected
            else:
                for other in self.objects:
                    other.selected = False
                obj.selected = True
            self.update_pivot()
        elif not modifiers & Qt.ControlModifier:
            for obj in self.objects:
                obj.selected = False
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if self.dragging:
            ratio = self.devicePixelRatio()
            x, y = position.x() * ratio, position.y() * ratio
            if self.mode == ManipMode.TRANSLATE:
                if self.manipulator.active_axis == CENTER:
                    delta = self.manipulator.drag_free_translate(x, y)
                else:
                    delta = self.manipulator.drag_translate(x, y)
                for obj in self.selected_objects():
                    obj.position += delta
                self.manipulator.position += delta
            elif self.mode == ManipMode.SCALE:
                if self.manipulator.active_axis == CENTER:
                    factors = self.manipulator.drag_uniform_scale(x, y)
                else:
                    factors = self.manipulator.drag_scale(x, y)
                for obj in self.selected_objects():
                    obj.scale = Vec3(
                        obj.scale.x * factors.x,
                        obj.scale.y * factors.y,
                        obj.scale.z * factors.z,
                    )
            elif self.mode == ManipMode.ROTATE:
                angle = self.manipulator.drag_rotate(x, y)
                axis = self.manipulator.active_axis
                if axis is not None:
                    for obj in self.selected_objects():
                        rot = obj.rotation
                        if axis == Axis.X:
                            rot.x += angle
                        elif axis == Axis.Y:
                            rot.y += angle
                        else:
                            rot.z += angle
            self.update()
        elif self.rotate_camera and event.buttons() == Qt.LeftButton:
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
            if self.dragging:
                self.dragging = False
                self.manipulator.end_drag()
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
