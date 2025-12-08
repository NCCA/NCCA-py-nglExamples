#!/usr/bin/env -S uv run  --script

import sys
import traceback

from ncca.ngl import Vec3, logger
from ncca.ngl.widgets import LookAtWidget, RGBColourWidget, TransformWidget
from PyNGLScene import PyNGLScene
from PySide6.QtCore import QFile, Qt, Signal
from PySide6.QtGui import QKeyEvent, QSurfaceFormat
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QColorDialog, QMainWindow, QWidget


class Loader(QUiLoader):
    def createWidget(self, class_name, parent=None, name=""):
        if class_name == "RGBColourWidget":
            return RGBColourWidget(parent)
        elif class_name == "TransformWidget":
            return TransformWidget(parent)
        elif class_name == "LookAtWidget":
            return LookAtWidget(parent)
        return super().createWidget(class_name, parent, name)


class MainWindow(QMainWindow):
    """
    The main window of the application, which hosts the OpenGL scene and UI controls.

    This class loads the user interface from a .ui file, integrates the PyNGLScene
    OpenGL widget, and connects UI signals (e.g., button clicks, slider changes)
    to the corresponding slots in the OpenGL scene to manipulate the 3D object.

    Attributes:
        colour_update (Signal): A signal that emits RGB float values when a new color is selected.
        scene (PyNGLScene): The OpenGL widget where the 3D scene is rendered.
    """

    # signal to emit when the colour is changed
    # colour_update = Signal(float, float, float)

    def __init__(self) -> None:
        """Initialize the MainWindow with UI setup and configuration loading."""
        super().__init__()

        self.load_ui()
        # Setup the custom promoted widgets
        self.lookat_widget.set_eye(Vec3(0.0, 1.0, 4.0))
        self.lookat_widget.set_name("Look At")
        self.transform_widget.set_name("Model Transform")
        self.scene = PyNGLScene()
        self.centralWidget().layout().addWidget(self.scene, 0, 0, 6, 1)

        self.resize(1024, 720)
        self._connect_slots()

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

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
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
    # Start the application's event loop
    sys.exit(app.exec())
