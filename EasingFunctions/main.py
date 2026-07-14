#!/usr/bin/env -S uv run --script
"""
EasingFunctions: animates two teapots between the same start and end points --
a gold reference teapot moving linearly, and a brass teapot driven by the
easing function selected in a combo box (the full Penner / easings.net set).
A matplotlib graph embedded in the Qt UI plots the selected easing curve
against the linear ramp, with a marker tracking the current time.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from easing import EASING_FUNCTIONS, get_source
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import Primitives, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DEMO_DIR = Path(__file__).parent


class GLWidget(QOpenGLWidget):
    """
    OpenGL viewport drawing a linearly interpolated gold teapot (top) and an
    eased brass teapot (bottom), both travelling between the same x extents.
    Standard arcball-style camera: LMB rotate, RMB pan, wheel zoom.
    """

    MATERIALS = {
        "gold": (
            Vec3(0.274725, 0.1995, 0.0745),
            Vec3(0.75164, 0.60648, 0.22648),
            Vec3(0.628281, 0.555802, 0.3666065),
            51.2,
        ),
        "brass": (
            Vec3(0.329412, 0.223529, 0.027451),
            Vec3(0.780392, 0.568627, 0.113725),
            Vec3(0.992157, 0.941176, 0.807843),
            27.8974,
        ),
    }

    # Start/end points and eased-teapot offset for each motion mode. In
    # "X only" the eased teapot travels a parallel path offset -4 in y so
    # the two never overlap; in "X and Y" both share the exact same path,
    # starting from the same position and separating as the easing diverges
    # from linear.
    MOTION_PATHS = {
        "X only": (Vec3(-8.0, 2.0, 0.0), Vec3(8.0, 2.0, 0.0), Vec3(0.0, -4.0, 0.0)),
        "X and Y": (Vec3(-8.0, -5.0, 0.0), Vec3(8.0, 5.0, 0.0), Vec3(0.0, 0.0, 0.0)),
    }

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        # --- Camera and transformation attributes ---
        self.mouse_global_tx: Mat4 = Mat4()
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()
        self.window_width: int = 1024
        self.window_height: int = 720

        # --- Mouse control attributes ---
        self.rotate: bool = False
        self.translate: bool = False
        self.spin_x_face: int = 0
        self.spin_y_face: int = 0
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.1

        # --- Animation state (driven by MainWindow) ---
        self.start, self.end, self.eased_offset = self.MOTION_PATHS["X only"]
        self.time: float = 0.0
        self.ease = EASING_FUNCTIONS["In Sine"]

    def set_motion(self, mode: str) -> None:
        """Switch between horizontal-only and diagonal (x and y) motion."""
        self.start, self.end, self.eased_offset = self.MOTION_PATHS[mode]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 25), Vec3(0, 0, 0), Vec3(0, 1, 0))
        # Resolve shader paths relative to this file so the demo works from
        # any cwd (e.g. when launched via RunDemos.py).
        ShaderLib.load_shader(
            "Phong",
            str(DEMO_DIR / "shaders/PhongVertex.glsl"),
            str(DEMO_DIR / "shaders/PhongFragment.glsl"),
        )
        Primitives.load_default_primitives()

    def _set_material(self, name: str) -> None:
        ambient, diffuse, specular, shininess = self.MATERIALS[name]
        ShaderLib.set_uniform("ambient", ambient.x, ambient.y, ambient.z)
        ShaderLib.set_uniform("diffuseColour", diffuse.x, diffuse.y, diffuse.z)
        ShaderLib.set_uniform("specularColour", specular.x, specular.y, specular.z)
        ShaderLib.set_uniform("shininess", shininess)

    def paintGL(self) -> None:
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Phong")
        ShaderLib.set_uniform("lightPos", 5.0, 5.0, 10.0)
        ShaderLib.set_uniform("viewerPos", 0.0, 0.0, 25.0)

        def draw_teapot(pos: Vec3, material: str) -> None:
            tx = Transform()
            tx.set_position(pos.x, pos.y, pos.z)
            tx.set_rotation(0, 90, 0)
            tx.set_scale(2.0, 2.0, 2.0)
            mv = self.view @ self.mouse_global_tx @ tx.matrix()
            mvp = self.project @ mv
            normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            self._set_material(material)
            Primitives.draw("teapot")

        # Gold reference teapot: plain linear interpolation.
        linear_pos = self.start + (self.end - self.start) * self.time
        # Brass teapot: same path, eased time, plus the per-mode offset.
        eased_pos = self.start + (self.end - self.start) * self.ease(self.time)
        eased_pos = eased_pos + self.eased_offset

        draw_teapot(linear_pos, "gold")
        draw_teapot(eased_pos, "brass")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class GraphWidget(FigureCanvas):
    """
    Matplotlib canvas plotting the selected easing curve against the linear
    ramp, with a marker tracking the current animation time.
    """

    def __init__(self, parent: QWidget = None) -> None:
        self.figure = Figure(figsize=(4, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self._t = np.linspace(0.0, 1.0, 200)
        self._curve_line = None
        self._marker = None
        self._linear_marker = None
        self.set_easing("In Sine")

    def set_easing(self, name: str) -> None:
        """Redraw the static curve for a newly selected easing function."""
        ease = EASING_FUNCTIONS[name]
        eased = np.array([ease(t) for t in self._t])
        self.axes.clear()
        self.axes.plot(self._t, self._t, "--", color="gray", label="linear")
        (self._curve_line,) = self.axes.plot(
            self._t, eased, color="tab:orange", lw=2, label=name
        )
        (self._marker,) = self.axes.plot([0.0], [0.0], "o", color="tab:red", ms=8)
        (self._linear_marker,) = self.axes.plot([0.0], [0.0], "o", color="gray", ms=8)
        self.axes.set_xlabel("t")
        self.axes.set_ylabel("eased value")
        self.axes.set_title(name)
        self.axes.set_xlim(0.0, 1.0)
        # Back/elastic overshoot outside [0,1] so fit the y range to the data.
        y_min = min(-0.05, float(eased.min()) - 0.05)
        y_max = max(1.05, float(eased.max()) + 0.05)
        self.axes.set_ylim(y_min, y_max)
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(loc="upper left", fontsize=8)
        self._ease = ease
        self.draw_idle()

    def set_time(self, t: float) -> None:
        """Move the eased and linear markers to the current animation time."""
        self._marker.set_data([t], [self._ease(t)])
        self._linear_marker.set_data([t], [t])
        self.draw_idle()


class MainWindow(QMainWindow):
    """
    Main window: GL viewport on the left, easing controls and the matplotlib
    graph on the right. A single timer drives both the animation and the
    graph marker so they stay in sync.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Easing Functions from https://easings.net/")

        self.gl_widget = GLWidget(self)
        self.graph = GraphWidget(self)

        self.combo = QComboBox()
        self.combo.addItems(EASING_FUNCTIONS.keys())
        self.combo.currentTextChanged.connect(self._easing_changed)

        self.motion_combo = QComboBox()
        self.motion_combo.addItems(GLWidget.MOTION_PATHS.keys())
        self.motion_combo.currentTextChanged.connect(self._motion_changed)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._toggle_animation)

        # Read-only view of the selected easing function's source code.
        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.code_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_view.setPlainText(get_source(self.combo.currentText()))

        # QLabel renders rich text; setOpenExternalLinks makes the anchor
        # open in the system browser when clicked.
        link_label = QLabel(
            'Easing functions from <a href="https://easings.net/">easings.net</a>'
        )
        link_label.setOpenExternalLinks(True)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(link_label)
        side_layout.addWidget(QLabel("Easing function"))
        side_layout.addWidget(self.combo)
        side_layout.addWidget(QLabel("Motion"))
        side_layout.addWidget(self.motion_combo)
        side_layout.addWidget(self.graph, stretch=2)
        side_layout.addWidget(QLabel("Algorithm"))
        side_layout.addWidget(self.code_view, stretch=1)
        side_layout.addWidget(self.pause_button)
        side_panel.setFixedWidth(420)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.gl_widget, stretch=1)
        layout.addWidget(side_panel)
        self.setCentralWidget(central)

        self.animate: bool = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def _easing_changed(self, name: str) -> None:
        self.gl_widget.ease = EASING_FUNCTIONS[name]
        self.graph.set_easing(name)
        self.code_view.setPlainText(get_source(name))
        self.gl_widget.time = 0.0

    def _motion_changed(self, mode: str) -> None:
        self.gl_widget.set_motion(mode)
        self.gl_widget.time = 0.0

    def _toggle_animation(self, paused: bool) -> None:
        self.animate = not paused
        self.pause_button.setText("Play" if paused else "Pause")

    def _tick(self) -> None:
        if self.animate:
            self.gl_widget.time += 0.005
            if self.gl_widget.time >= 1.0:
                self.gl_widget.time = 0.0
        self.graph.set_time(self.gl_widget.time)
        self.gl_widget.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            self.gl_widget.makeCurrent()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        elif key == Qt.Key_S:
            self.gl_widget.makeCurrent()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            self.pause_button.toggle()
        elif key == Qt.Key_Left:
            self.gl_widget.time = max(0.0, self.gl_widget.time - 0.01)
        elif key == Qt.Key_Right:
            self.gl_widget.time = min(1.0, self.gl_widget.time + 0.01)
        self.gl_widget.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    """
    QApplication subclass that surfaces exceptions raised inside Qt event
    handlers (paintGL etc.), which Qt would otherwise swallow.
    """

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

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    # OpenGL 4.1 Core is the highest supported on macOS
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1400, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
