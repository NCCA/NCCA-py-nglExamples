#!/usr/bin/env -S uv run --script
"""
AffineTransforms: interactively compose translate/rotate/scale matrices (WebGPU).

Same matrix-order comparison as the OpenGL version (main.py) — Rotate-
Translate-Scale, Translate-Rotate-Scale, or Translate-(axis-angle)-Scale —
using a simpler diffuse shader (no PBR, no geometry-shader normal
visualization: WebGPU has no geometry-shader stage) and a primitive
selector limited to the baked mesh set.

Controls: all on the panel; left-drag in the viewport orbits the camera.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import (
    Mat4,
    MatrixError,
    PerspMode,
    PrimData,
    Quaternion,
    Vec3,
    look_at,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from ncca.ngl.widgets import Mat4Widget, Vec3Widget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from wgpu.utils import get_default_device

_MESH_NAMES = [
    "teapot",
    "cube",
    "troll",
    "buddah",
    "dragon",
    "bunny",
    "football",
    "octahedron",
    "dodecahedron",
    "icosahedron",
    "tetrahedron",
]

UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("colour", np.float32, 4),
    ]
)


class WebGPUScene(WebGPUWidget):
    _ORDERS = [
        ("Rotate -> Translate -> Scale", "RTS"),
        ("Translate -> Rotate -> Scale", "TRS"),
        ("Translate -> Axis-Angle -> Scale", "TAxisS"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (WebGPU)")
        self.msaa_sample_count = 4

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

        self.mesh_index = 0
        self.order = "RTS"
        self.translate_v = Vec3(0, 0, 0)
        self.rotate_v = Vec3(0, 0, 0)
        self.scale_v = Vec3(1, 1, 1)
        self.axis_angle = 0.0
        self.axis_v = Vec3(1, 0, 0)
        self.colour = (0.95, 0.71, 0.29)

        self.view = look_at(Vec3(0, 0, 8), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.05, 450.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._load_meshes()
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "AffineTransformsShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)
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
        self.pipeline = self.device.create_render_pipeline(
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
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
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
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = self.device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
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

    def _load_meshes(self) -> None:
        self.meshes = {}
        for name in _MESH_NAMES:
            data = PrimData.primitive(name)
            buf = self.device.create_buffer_with_data(
                data=data, usage=wgpu.BufferUsage.VERTEX
            )
            self.meshes[name] = (buf, data.size // 8)

    def transform_matrix(self) -> Mat4:
        t = Mat4().translate(self.translate_v.x, self.translate_v.y, self.translate_v.z)
        s = Mat4().scale(self.scale_v.x, self.scale_v.y, self.scale_v.z)
        if self.order in ("RTS", "TRS"):
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return (r @ t @ s) if self.order == "RTS" else (t @ r @ s)
        # "TAxisS": translate, axis-angle rotation, scale.
        # Quaternion.from_axis_angle() does not normalize its axis, so a
        # non-unit axis (e.g. the very natural (1,1,1)) would silently bake
        # extra scale into what is supposed to be a pure rotation. A
        # zero-length axis has no defined direction -- fall back to identity
        # rotation rather than letting Vec3.normalized() raise. Same fix as
        # main.py's (OpenGL sibling) transform_matrix().
        try:
            axis = self.axis_v.normalized()
            r = Quaternion.from_axis_angle(axis, self.axis_angle).to_mat4()
        except ZeroDivisionError:
            r = Mat4()
        return t @ r @ s

    def paintWebGPU(self) -> None:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        model = self.transform_matrix()
        mv = self.view @ self.mouse_global_tx @ model
        self.uniforms["mvp"] = (self.project @ mv).to_numpy()
        # A scale slider can reach exactly 0 (range is -20..20), which makes
        # `mv` singular -- Mat4.inverse() raises MatrixError in that case.
        # Fall back to the identity normal matrix rather than crashing the
        # next repaint; the object itself is degenerate (zero volume) at
        # that point anyway, so the shading is moot. Same fix as main.py's
        # (OpenGL sibling) paintGL().
        try:
            normal_matrix = mv.inverse().transposed()
        except MatrixError:
            normal_matrix = Mat4()
        self.uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.uniforms["colour"] = (*self.colour, 1.0)
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

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
        buf, count = self.meshes[_MESH_NAMES[self.mesh_index]]
        render_pass.set_vertex_buffer(0, buf)
        render_pass.draw(count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.05, 450.0, PerspMode.WebGPU)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (WebGPU)")
        self.scene = WebGPUScene()

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addWidget(self.scene, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.resize(1200, 720)

    def _build_panel(self) -> QWidget:
        panel = QWidget(self)
        outer = QVBoxLayout(panel)

        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems(_MESH_NAMES)
        self.mesh_combo.currentIndexChanged.connect(self._on_mesh_changed)
        outer.addWidget(QLabel("Mesh"))
        outer.addWidget(self.mesh_combo)

        self.order_combo = QComboBox()
        for label, _ in WebGPUScene._ORDERS:
            self.order_combo.addItem(label)
        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        outer.addWidget(QLabel("Matrix Order"))
        outer.addWidget(self.order_combo)

        self.translate_widget = Vec3Widget(panel, "Translate", Vec3(0, 0, 0))
        self.translate_widget.set_range(-20, 20)
        self.translate_widget.valueChanged.connect(self._on_translate_changed)
        outer.addWidget(self.translate_widget)

        self.rotate_widget = Vec3Widget(panel, "Rotate", Vec3(0, 0, 0))
        self.rotate_widget.set_range(-180, 180)
        self.rotate_widget.valueChanged.connect(self._on_rotate_changed)
        outer.addWidget(self.rotate_widget)

        self.scale_widget = Vec3Widget(panel, "Scale", Vec3(1, 1, 1))
        self.scale_widget.set_range(-20, 20)
        self.scale_widget.valueChanged.connect(self._on_scale_changed)
        outer.addWidget(self.scale_widget)

        axis_group = QGroupBox(
            "Axis-Angle (used when order is Translate -> Axis-Angle -> Scale)"
        )
        axis_layout = QVBoxLayout(axis_group)
        self.axis_widget = Vec3Widget(axis_group, "Axis", Vec3(1, 0, 0))
        self.axis_widget.valueChanged.connect(self._on_axis_changed)
        axis_layout.addWidget(self.axis_widget)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._on_angle_changed)
        axis_layout.addWidget(QLabel("Angle"))
        axis_layout.addWidget(self.angle_slider)
        outer.addWidget(axis_group)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._on_reset_clicked)
        outer.addWidget(reset_button)

        self.matrix_widget = Mat4Widget(panel, "Transform Matrix", read_only=True)
        outer.addWidget(self.matrix_widget)

        outer.addStretch(1)
        return panel

    def _refresh_matrix_display(self) -> None:
        self.matrix_widget.set_value(self.scene.transform_matrix())
        self.scene.update()

    def _on_mesh_changed(self, index: int) -> None:
        self.scene.mesh_index = index
        self.scene.update()

    def _on_order_changed(self, index: int) -> None:
        self.scene.order = WebGPUScene._ORDERS[index][1]
        self._refresh_matrix_display()

    def _on_translate_changed(self, value: Vec3) -> None:
        self.scene.translate_v = value
        self._refresh_matrix_display()

    def _on_rotate_changed(self, value: Vec3) -> None:
        self.scene.rotate_v = value
        self._refresh_matrix_display()

    def _on_scale_changed(self, value: Vec3) -> None:
        self.scene.scale_v = value
        self._refresh_matrix_display()

    def _on_axis_changed(self, value: Vec3) -> None:
        self.scene.axis_v = value
        self._refresh_matrix_display()

    def _on_angle_changed(self, value: int) -> None:
        self.scene.axis_angle = float(value)
        self._refresh_matrix_display()

    def _on_reset_clicked(self) -> None:
        self.translate_widget.set_value(Vec3(0, 0, 0))
        self.rotate_widget.set_value(Vec3(0, 0, 0))
        self.scale_widget.set_value(Vec3(1, 1, 1))
        self.angle_slider.setValue(0)
        self.scene.spin_x_face = 0
        self.scene.spin_y_face = 0
        self.scene.model_position.set(0, 0, 0)
        self._refresh_matrix_display()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()


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
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
