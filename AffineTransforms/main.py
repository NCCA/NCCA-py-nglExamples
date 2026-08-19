#!/usr/bin/env -S uv run --script
"""
AffineTransforms: interactively compose translate/rotate/scale matrices (OpenGL).

A PBR-shaded primitive sits at the origin next to an RGB axis gizmo. Sliders
set independent translate/rotate/scale values and a combo box picks the
*order* those are composed in — Rotate-Translate-Scale, Translate-Rotate-
Scale, or Translate-(axis-angle)-Scale — so you can see directly how order
changes the result. A read-only matrix grid shows the composed transform.

Controls: all on the panel; left-drag in the viewport orbits the camera.
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from axis import draw_axis
from ncca.ngl import (
    Mat3,
    Mat4,
    MatrixError,
    Prims,
    Quaternion,
    Vec3,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.opengl import Primitives, PySideEventHandlingMixin, ShaderLib
from ncca.ngl.widgets import Mat4Widget, Vec3Widget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
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

_VBO_NAMES = [
    "sphere",
    "cylinder",
    "cone",
    "disk",
    "plane",
    "torus",
    "teapot",
    "octahedron",
    "dodecahedron",
    "icosahedron",
    "tetrahedron",
    "football",
    "cube",
    "troll",
    "buddah",
    "dragon",
    "bunny",
]


class Scene(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 800
        self.window_height: int = 720
        self.setTitle("AffineTransforms (OpenGL)")

        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.draw_index: int = 6  # teapot
        self.wireframe: bool = False
        self.draw_normals: bool = False
        self.normal_size: float = 0.6
        self.colour: tuple[float, float, float] = (0.95, 0.71, 0.29)

        self.translate_v = Vec3(0, 0, 0)
        self.rotate_v = Vec3(0, 0, 0)
        self.scale_v = Vec3(1, 1, 1)
        self.axis_angle: float = 0.0
        self.axis_v = Vec3(1, 0, 0)
        self.order: str = "RTS"

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        from_pos = Vec3(0, 0, 8)
        self.view = look_at(from_pos, Vec3(0, 0, 0), Vec3(0, 1, 0))

        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)
        Primitives.create(Prims.CYLINDER, "cylinder", 0.5, 1.4, 40, 40)
        Primitives.create(Prims.CONE, "cone", 0.5, 1.4, 20, 20)
        Primitives.create(Prims.DISK, "disk", 0.5, 40)
        Primitives.create(
            Prims.TRIANGLE_PLANE, "plane", 1.0, 1.0, 10, 10, Vec3(0, 1, 0)
        )
        Primitives.create(Prims.TORUS, "torus", 0.15, 0.4, 40, 40)

        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "PBR",
            str(shader_dir / "PBRVertex.glsl"),
            str(shader_dir / "PBRFragment.glsl"),
        )
        ShaderLib.use("PBR")
        ShaderLib.set_uniform("camPos", from_pos)
        ShaderLib.set_uniform("lightPositions[0]", 0.0, 2.0, 2.0)
        ShaderLib.set_uniform("lightColours[0]", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("lightPositions[1]", -10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[1]", 0.0, 0.0, 0.0)
        ShaderLib.set_uniform("lightPositions[2]", 10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[2]", 0.0, 0.0, 0.0)
        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)
        # PBRFragment.glsl divides by exposure for its gamma step
        # (pow(colour, 1/exposure)) -- an unset uniform defaults to 0.0,
        # which collapses every pixel to black. Camera/main.py sets the
        # same value; matched here for the same reason.
        ShaderLib.set_uniform("exposure", 2.2)

        ShaderLib.load_shader(
            "NormalViz",
            str(shader_dir / "normalVertex.glsl"),
            str(shader_dir / "normalFragment.glsl"),
            str(shader_dir / "normalGeo.glsl"),
        )
        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("vertNormalColour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("faceNormalColour", 1.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("drawFaceNormals", True)
        ShaderLib.set_uniform("drawVertexNormals", True)

    def transform_matrix(self) -> Mat4:
        t = Mat4().translate(self.translate_v.x, self.translate_v.y, self.translate_v.z)
        s = Mat4().scale(self.scale_v.x, self.scale_v.y, self.scale_v.z)
        if self.order == "RTS":
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return r @ t @ s
        elif self.order == "TRS":
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return t @ r @ s
        else:  # "TAxisS": translate, axis-angle rotation, scale
            # Quaternion.from_axis_angle() does not normalize its axis, so a
            # non-unit axis (e.g. the very natural (1,1,1)) would silently
            # bake extra scale into what is supposed to be a pure rotation.
            # A zero-length axis has no defined direction -- fall back to
            # identity rotation rather than letting Vec3.normalized() raise.
            try:
                axis = self.axis_v.normalized()
                r = Quaternion.from_axis_angle(axis, self.axis_angle).to_mat4()
            except ZeroDivisionError:
                r = Mat4()
            return t @ r @ s

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPolygonMode(
            gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL
        )

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        model = self.transform_matrix()
        m = global_tx @ model
        mv = self.view @ m

        ShaderLib.use("PBR")
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("M", m)
        # PBRFragment.glsl lights in world space (V = camPos - WorldPos), so
        # the normal matrix comes from the model alone -- folding the view
        # in here would tilt the normals as soon as the camera orbits (see
        # commit 8b77482, the same fix applied family-wide to HDRI/IBL/
        # SimplePBR/PBRTexture).
        # A scale slider can reach exactly 0 (range is -20..20), which makes
        # `m` singular -- Mat3.inverse() raises MatrixError in that case.
        # Fall back to the identity normal matrix rather than crashing the
        # next repaint; the object itself is degenerate (zero volume) at
        # that point anyway, so the shading is moot.
        try:
            normal_matrix = Mat3.from_mat4(m).inverse().transposed()
        except MatrixError:
            normal_matrix = Mat3()
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("albedo", *self.colour)
        Primitives.draw(_VBO_NAMES[self.draw_index])

        if self.draw_normals:
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform("normalSize", self.normal_size / 10.0)
            Primitives.draw(_VBO_NAMES[self.draw_index])

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # draw_axis()'s own default scale (1.5) is sized for a much larger
        # scene -- against ncca.ngl's stock teapot (roughly 1.9 units along
        # its longest axis) it swallows the primitive whole, so the very
        # thing this demo exists to show never appears. Pass an explicit
        # smaller scale here rather than touching axis.py's default, since
        # the signature is this task's produced interface.
        draw_axis(self.view, self.project, global_tx, scale=0.35)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 450.0)

    def keyPressEvent(self, event) -> None:
        # PySideEventHandlingMixin.keyPressEvent handles Escape with
        # self.close() -- fine for a top-level window, but here `self` is
        # the embedded QOpenGLWindow shown via createWindowContainer, so
        # that would only close the viewport and leave the QMainWindow
        # shell up with a dead embedded window. Intercept Escape here and
        # quit the whole application instead; everything else still goes
        # through the mixin (orbit/pan/zoom keys, wireframe, reset, etc.).
        if event.key() == Qt.Key_Escape:
            QApplication.instance().quit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    # These name the matrix product built in Scene.transform_matrix() (e.g.
    # "Rotate -> Translate -> Scale" means r @ t @ s), not a left-to-right
    # application order. Since points transform as matrix @ point, the
    # rightmost term acts on the object first -- see README.md for the
    # worked-through reasoning.
    _ORDERS = [
        ("Rotate -> Translate -> Scale", "RTS"),
        ("Translate -> Rotate -> Scale", "TRS"),
        ("Translate -> Axis-Angle -> Scale", "TAxisS"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (OpenGL)")
        self.scene = Scene()
        gl_container = QWidget.createWindowContainer(self.scene, self)
        gl_container.setMinimumSize(600, 600)
        # createWindowContainer embeds a native child window; it only
        # receives keyboard/mouse focus if explicitly told to accept it,
        # otherwise clicks land in the container but keys never reach
        # Scene's mixin-provided keyPressEvent. Same fix BVHViewer and
        # SkinnedMeshImport already apply for their own embedded viewports.
        gl_container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        gl_container.setFocus()

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addWidget(gl_container, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.resize(1200, 720)

    def _build_panel(self) -> QWidget:
        panel = QWidget(self)
        outer = QVBoxLayout(panel)

        self.primitive_combo = QComboBox()
        self.primitive_combo.addItems(_VBO_NAMES)
        self.primitive_combo.setCurrentIndex(self.scene.draw_index)
        self.primitive_combo.currentIndexChanged.connect(self._on_primitive_changed)
        outer.addWidget(QLabel("Primitive"))
        outer.addWidget(self.primitive_combo)

        self.order_combo = QComboBox()
        for label, _ in self._ORDERS:
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

        toggles = QHBoxLayout()
        self.wireframe_check = QCheckBox("Wireframe")
        self.wireframe_check.toggled.connect(self._on_wireframe_toggled)
        toggles.addWidget(self.wireframe_check)
        self.normals_check = QCheckBox("Normals")
        self.normals_check.toggled.connect(self._on_normals_toggled)
        toggles.addWidget(self.normals_check)
        outer.addLayout(toggles)

        self.normal_size_slider = QSlider(Qt.Horizontal)
        self.normal_size_slider.setRange(1, 20)
        self.normal_size_slider.setValue(6)
        self.normal_size_slider.valueChanged.connect(self._on_normal_size_changed)
        outer.addWidget(QLabel("Normal Size"))
        outer.addWidget(self.normal_size_slider)

        colour_button = QPushButton("Colour")
        colour_button.clicked.connect(self._on_colour_clicked)
        outer.addWidget(colour_button)

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

    def _on_primitive_changed(self, index: int) -> None:
        self.scene.draw_index = index
        self.scene.update()

    def _on_order_changed(self, index: int) -> None:
        self.scene.order = self._ORDERS[index][1]
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

    def _on_wireframe_toggled(self, checked: bool) -> None:
        self.scene.wireframe = checked
        self.scene.update()

    def _on_normals_toggled(self, checked: bool) -> None:
        self.scene.draw_normals = checked
        self.scene.update()

    def _on_normal_size_changed(self, value: int) -> None:
        self.scene.normal_size = float(value)
        self.scene.update()

    def _on_colour_clicked(self) -> None:
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.scene.colour = (colour.redF(), colour.greenF(), colour.blueF())
            self.scene.update()

    def _on_reset_clicked(self) -> None:
        self.translate_widget.set_value(Vec3(0, 0, 0))
        self.rotate_widget.set_value(Vec3(0, 0, 0))
        self.scale_widget.set_value(Vec3(1, 1, 1))
        self.angle_slider.setValue(0)
        self.wireframe_check.setChecked(False)
        self.normals_check.setChecked(False)
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
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
