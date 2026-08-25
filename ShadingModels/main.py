#!/usr/bin/env -S uv run  --script

import argparse
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from GLSLHighlighter import GLSLHighlighter
from ncca.ngl import Vec3, logger
from ncca.ngl.widgets import (
    LookAtWidget,
    RGBAColourWidget,
    RGBColourWidget,
    TransformWidget,
    Vec3Widget,
    Vec4Widget,
)
from PyNGLScene import PyNGLScene
from PySide6.QtCore import QEvent, QFile, QObject, Qt, QTimer
from PySide6.QtGui import QFont, QSurfaceFormat
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QMainWindow,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Loader(QUiLoader):
    """A custom QUiLoader to create custom widgets."""

    def createWidget(
        self, class_name: str, parent: QWidget | None = None, name: str = ""
    ) -> QWidget:
        """
        Create a custom widget by name.

        Parameters
        ----------
            class_name : str
                The name of the class to create.
            parent : QWidget | None
                The parent widget.
            name : str
                The name of the widget.

        Returns
        -------
            QWidget
                The created widget.
        """
        if class_name == "RGBColourWidget":
            return RGBColourWidget(parent)
        elif class_name == "TransformWidget":
            return TransformWidget(parent)
        elif class_name == "LookAtWidget":
            return LookAtWidget(parent)
        elif class_name == "Vec3Widget":
            return Vec3Widget(parent)
        elif class_name == "Vec4Widget":
            return Vec4Widget(parent)
        return super().createWidget(class_name, parent, name)


class MainWindow(QMainWindow):
    """
    The main window of the application, which hosts the OpenGL scene and UI controls.
    """

    def __init__(self) -> None:
        """Initialize the MainWindow with UI setup and configuration loading."""
        super().__init__()
        self.setWindowTitle("Shading Models")

        # track the current fullscreen widget (None when normal)
        self._fullscreen_widget: QWidget | None = None

        self.load_ui()
        self.vert_editor = QPlainTextEdit(self)
        self.frag_editor = QPlainTextEdit(self)
        for ed in (self.vert_editor, self.frag_editor):
            ed.setReadOnly(True)
            f = QFont()
            f.setFamily("Courier New")
            f.setStyleHint(QFont.Monospace)
            f.setPointSize(14)
            ed.setFont(f)
            # install event filter to catch double-click on editors
            ed.viewport().installEventFilter(self)

        # put editors inside the group boxes (add a layout if missing)
        for gb, ed in (
            (self.vertex_shader_gb, self.vert_editor),
            (self.fragment_shader_gb, self.frag_editor),
        ):
            layout = gb.layout()
            if layout is None:
                layout = QVBoxLayout()
                gb.setLayout(layout)
            # remove existing widgets in the group box layout, if you want a clean state
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
            layout.addWidget(ed)

        # create highlighters
        self.vert_highlighter = GLSLHighlighter(self.vert_editor.document())
        self.frag_highlighter = GLSLHighlighter(self.frag_editor.document())
        # Setup the custom promoted widgets
        self.lookat_widget.set_eye(Vec3(0.0, 1.0, 4.0))
        self.lookat_widget.set_name("Look At")
        self.transform_widget.set_name("Model Transform")
        self.scene = PyNGLScene()

        # make scene expand by default
        self.scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # add scene to the central grid layout (same as before)
        self.centralWidget().layout().addWidget(self.scene, 0, 0, 6, 1)

        # list of side widgets we want to hide when toggling fullscreen
        self._side_widgets = [
            getattr(self, nm)
            for nm in (
                "transform_gb",
                "camera_gb",
                "draw_gb",
                "vertex_shader_gb",
                "fragment_shader_gb",
                "uniforms_gb",
            )
            if hasattr(self, nm)
        ]

        # connect scene double click to toggle
        self.scene.double_clicked.connect(
            lambda: self.toggle_fullscreen_widget(self.scene)
        )

        self.uniform_layout = QFormLayout()
        self.uniforms_gb.setLayout(self.uniform_layout)
        self.load_shader.clicked.connect(self.load_shader_clicked)
        self.resize(1024, 720)
        self._connect_slots()
        self.scene.uniform_found.connect(self.add_uniform_widget)
        self._widget_factory: dict[str, Callable] = {
            "vec3": self._create_vec3_widget,
            "colour3": self._create_rgb_colour_widget,
            "vec4": self._create_vec4_widget,
            "colour4": self._create_rgba_colour_widget,
            "float": self._create_float_widget,
        }

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Filter events to catch double-clicks on editor viewports.

        Parameters
        ----------
            obj : QObject
                The object that sent the event.
            event : QEvent
                The event.

        Returns
        -------
            bool
                True if the event was handled, False otherwise.
        """
        if (
            event.type() == QEvent.Type.Wheel
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ) and obj in (self.vert_editor.viewport(), self.frag_editor.viewport()):
            editor = (
                self.vert_editor
                if obj is self.vert_editor.viewport()
                else self.frag_editor
            )
            font = editor.font()
            if event.angleDelta().y() > 0:
                font.setPointSize(font.pointSize() + 1)
            else:
                font.setPointSize(max(6, font.pointSize() - 1))
            editor.setFont(font)
            return True
        # catch double-click on editor viewports to toggle fullscreen for them
        if event.type() == QEvent.MouseButtonDblClick:
            if obj in (self.vert_editor.viewport(), self.frag_editor.viewport()):
                # map viewport back to the editor widget
                widget = (
                    self.vert_editor
                    if obj is self.vert_editor.viewport()
                    else self.frag_editor
                )
                self.toggle_fullscreen_widget(widget)
                return True
        # also allow normal event processing for other events
        return super().eventFilter(obj, event)

    def toggle_fullscreen_widget(self, widget: QWidget) -> None:
        """
        Toggle `widget` between normal layout placement and widget-only fullscreen.
        """
        layout = self.centralWidget().layout()

        if self._fullscreen_widget is None:
            # go fullscreen for this widget
            self._fullscreen_widget = widget

            # hide the side widgets
            for w in self._side_widgets:
                w.setVisible(False)

            # remove and re-add widget to span all columns (0..2) and full rows (0..5)
            try:
                layout.removeWidget(widget)
            except Exception:
                pass
            layout.addWidget(widget, 0, 0, 6, 3)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.update()
        else:
            # restore from fullscreen
            if self._fullscreen_widget is not widget:
                # restore current fullscreen, then set new fullscreen
                self.toggle_fullscreen_widget(self._fullscreen_widget)
                self.toggle_fullscreen_widget(widget)
                return

            # show side widgets again
            for w in self._side_widgets:
                w.setVisible(True)

            # remove and re-add scene/editor back to original span (scene: colSpan 1)
            try:
                layout.removeWidget(widget)
            except Exception:
                pass

            # If toggled widget is the scene, restore to its original span (0,0,6,1).
            if widget is self.scene:
                layout.addWidget(widget, 0, 0, 6, 1)
            else:
                # editors originally sit inside group boxes; put editors back into their group box layout
                # (we removed them only from group layout earlier; to be safe just re-add to their group)
                if widget is self.vert_editor:
                    gb = self.vertex_shader_gb
                elif widget is self.frag_editor:
                    gb = self.fragment_shader_gb
                else:
                    gb = None
                if gb is not None:
                    glayout = gb.layout()
                    if glayout is None:
                        glayout = QVBoxLayout()
                        gb.setLayout(glayout)
                    glayout.addWidget(widget)

            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._fullscreen_widget = None
            widget.update()

    def _create_vec3_widget(self, name: str, vrange: Any, value: Any) -> QWidget:
        widget = Vec3Widget(parent=self.uniforms_gb, name=name, value=value)
        if vrange:
            widget.set_range(vrange.x, vrange.y)
            widget.set_value(value)
        widget.valueChanged.connect(
            lambda val, name=name: self.scene.set_uniform_value(name, val)
        )
        return widget

    def _create_rgb_colour_widget(self, name: str, vrange: Any, value: Any) -> QWidget:
        widget = RGBColourWidget(self.uniforms_gb, name, value[0], value[1], value[2])
        widget.colourChanged.connect(
            lambda val, name=name: self.scene.set_uniform_value(name, val)
        )
        return widget

    def _create_vec4_widget(self, name: str, vrange: Any, value: Any) -> QWidget:
        widget = Vec4Widget(parent=self.uniforms_gb, name=name, value=value)
        widget.valueChanged.connect(
            lambda val, name=name: self.scene.set_uniform_value(name, val)
        )
        return widget

    def _create_rgba_colour_widget(self, name: str, vrange: Any, value: Any) -> QWidget:
        widget = RGBAColourWidget(
            self.uniforms_gb, name, value[0], value[1], value[2], value[3]
        )
        widget.colourChanged.connect(
            lambda val, name=name: self.scene.set_uniform_value(name, val)
        )
        return widget

    def _create_float_widget(self, name: str, vrange: Any, value: Any) -> QWidget:
        spin = QDoubleSpinBox(self.uniforms_gb)
        spin.setObjectName(f"uniform_{name}")
        if vrange:
            spin.setRange(vrange.x, vrange.y)
        else:
            spin.setRange(-100, 100)
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        try:
            spin.setValue(float(value))
        except Exception:
            spin.setValue(0.0)
        spin.valueChanged.connect(
            lambda val, name=name: self.scene.set_uniform_value(name, float(val))
        )
        return spin

    def add_uniform_widget(
        self, name: str, data_type: str, vrange: Any, value: Any
    ) -> None:
        """
        Add a widget to the UI for a uniform variable using a factory pattern.

        Parameters
        ----------
            name : str
                The name of the uniform.
            data_type : str
                The type of the uniform.
            vrange : Any
                The range of the uniform value.
            value : Any
                The value of the uniform.
        """
        creator_func = self._widget_factory.get(data_type.lower())
        if creator_func:
            widget = creator_func(name, vrange, value)
            self.uniform_layout.addRow(name, widget)
        else:
            logger.warning(f"No widget factory for uniform type '{data_type}'")

    def load_shader_clicked(self) -> None:
        """Open a file dialog to load a shader."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Json Files (*.json)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.generate_ui_layout(file_path)

    def generate_ui_layout(self, file_path: str) -> None:
        """
        Generate the UI layout for the shader.

        Parameters
        ----------
            file_path : str
                The path to the shader file.
        """
        # Clear existing widgets from the form layout
        while self.uniform_layout.count():
            item = self.uniform_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()

        # Now load the new shader (the scene will emit uniform_found for each uniform)
        self.scene.new_shader(file_path)
        # populate the shader editors with the actual shader source
        try:
            base = Path(file_path).parent
            vert_file = base / self.scene.shader.shader_data["VertexShader"]
            frag_file = base / self.scene.shader.shader_data["FragmentShader"]
            if vert_file.exists():
                self.vert_editor.setPlainText(vert_file.read_text())
            else:
                self.vert_editor.setPlainText(f"Vertex shader not found: {vert_file}")
            if frag_file.exists():
                self.frag_editor.setPlainText(frag_file.read_text())
            else:
                self.frag_editor.setPlainText(f"Fragment shader not found: {frag_file}")
        except Exception as e:
            self.vert_editor.setPlainText(f"Error loading shader sources: {e}")
            self.frag_editor.setPlainText(f"Error loading shader sources: {e}")

    def _connect_slots(self) -> None:
        """Connect UI element signals to their corresponding slots."""
        self.wireframe.toggled.connect(self.scene.set_wireframe)
        self.object_selection.currentTextChanged.connect(self.scene.set_model_name)
        self.transform_widget.valueChanged.connect(self.scene.set_transform)
        self.lookat_widget.valueChanged.connect(self.scene.set_view_matrix)
        self.fov.valueChanged.connect(self.update_perspective)
        self.near.valueChanged.connect(self.update_perspective)
        self.far.valueChanged.connect(self.update_perspective)

    def update_perspective(self) -> None:
        """Update the perspective projection matrix."""
        self.scene.update_perspective(
            self.fov.value(), self.near.value(), self.far.value()
        )
        self.update()

    def load_ui(self) -> None:
        """Load the UI from a .ui file and set up the connections."""
        try:
            loader = Loader()
            ui_file = QFile("MainWindow.ui")
            ui_file.open(QFile.ReadOnly)

            loaded_ui = loader.load(ui_file, self)
            self.setCentralWidget(loaded_ui)

            # Add all children with object names as attributes
            for child in loaded_ui.findChildren(QWidget):
                name = child.objectName()
                if name:
                    setattr(self, name, child)

            ui_file.close()

        except Exception as e:
            print(f"Error loading UI file: {e}")
            raise

    def keyPressEvent(self, QKeyEvent) -> None:
        """Handle key press events to close the window on Escape."""
        if QKeyEvent.key() == Qt.Key_Escape:
            # if fullscreen widget is active, restore first; otherwise close
            if self._fullscreen_widget is not None:
                self.toggle_fullscreen_widget(self._fullscreen_widget)
            else:
                self.close()


class DebugApplication(QApplication):
    """
    A custom QApplication subclass for improved debugging.

    By default, Qt's event loop can suppress exceptions that occur within event handlers
    (like paintGL or mouseMoveEvent), making it very difficult to debug as the application
    may simply crash or freeze without any error message. This class overrides the `notify`
    method to catch these exceptions, print a full traceback to the console, and then
    re-raise the exception to halt the program, making the error immediately visible.
    """

    def __init__(self, argv: list[str]) -> None:
        """
        Initialize the DebugApplication.

        Parameters
        ----------
            argv : list[str]
                The command line arguments.
        """
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            # Attempt to process the event as usual
            return super().notify(receiver, event)
        except Exception:
            # If an exception occurs, print the full traceback
            traceback.print_exc()
            # Re-raise the exception to stop the application
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

    # --- Application Entry Point ---

    format: QSurfaceFormat = QSurfaceFormat()
    # Request 4x multisampling for anti-aliasing
    format.setSamples(4)
    # Request OpenGL version 4.1 as this is the highest supported on macOS
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    # Request a Core Profile context, which removes deprecated, fixed-function pipeline features
    format.setProfile(QSurfaceFormat.CoreProfile)
    # Request a 24-bit depth buffer for proper 3D sorting
    format.setDepthBufferSize(24)
    # Set default format for all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    # Apply this format to all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()
    window.generate_ui_layout("shaders/Constant.json")

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    # Start the application's event loop
    sys.exit(app.exec())
