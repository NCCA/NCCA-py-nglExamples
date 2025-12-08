#!/usr/bin/env -S uv run  --script

import sys
import traceback
from pathlib import Path

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
from PySide6.QtCore import QEvent, QFile, Qt
from PySide6.QtGui import QFont, QKeyEvent, QSurfaceFormat
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
    def createWidget(self, class_name, parent=None, name=""):
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
        self._fullscreen_widget = None

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

    def eventFilter(self, obj, event):
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

    def toggle_fullscreen_widget(self, widget: QWidget):
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

    def add_uniform_widget(self, name: str, data_type: str, value) -> None:
        if data_type == "Vec3":
            widget = Vec3Widget(parent=self.uniforms_gb, name=name, value=value)
            widget.valueChanged.connect(
                lambda val, name=name: self.scene.set_uniform_value(name, val)
            )
        elif data_type == "Colour3":
            widget = RGBColourWidget(
                self.uniforms_gb, name, value[0], value[1], value[2]
            )
            widget.colourChanged.connect(
                lambda val, name=name: self.scene.set_uniform_value(name, val)
            )
        elif data_type == "Vec4":
            widget = Vec4Widget(parent=self.uniforms_gb, name=name, value=value)
            widget.valueChanged.connect(
                lambda val, name=name: self.scene.set_uniform_value(name, val)
            )
        elif data_type == "Colour4":
            widget = RGBAColourWidget(
                self.uniforms_gb, name, value[0], value[1], value[2], value[3]
            )
            widget.colourChanged.connect(
                lambda val, name=name: self.scene.set_uniform_value(name, val)
            )
        elif data_type.lower() == "float":
            spin = QDoubleSpinBox(self.uniforms_gb)
            spin.setObjectName(f"uniform_{name}")
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
            widget = spin
        else:
            return

        self.uniform_layout.addRow(name, widget)

    def load_shader_clicked(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Json Files (*.json)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.generate_ui_layout(file_path)

    def generate_ui_layout(self, file_path):
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
        self.lookat_widget.valueChanged.connect(self.scene.set_camera)
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events to close the window on Escape."""
        if event.key() == Qt.Key_Escape:
            # if fullscreen widget is active, restore first; otherwise close
            if self._fullscreen_widget is not None:
                self.toggle_fullscreen_widget(self._fullscreen_widget)
            else:
                self.close()


if __name__ == "__main__":
    # --- Application Entry Point ---

    # Create a QSurfaceFormat object to request a specific OpenGL context
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

    # Check for a "--debug" command-line argument to run the DebugApplication
    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    # Set the initial window size
    window.resize(1024, 720)
    # Show the window
    window.show()
    window.generate_ui_layout("shaders/Constant.json")
    # Start the application's event loop
    sys.exit(app.exec())
