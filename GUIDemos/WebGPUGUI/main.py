#!/usr/bin/env -S uv run --script
import argparse
import sys
import traceback

from ncca.ngl import Vec3
from PySide6.QtCore import QFile, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QColorDialog, QMainWindow, QWidget
from WebGPUScene import WebGPUScene


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


class MainWindow(QMainWindow):
    """
    The main window of the application, which hosts the WebGPU scene and UI controls.

    This class loads the user interface from a .ui file, integrates the WebGPUScene
    widget, and connects UI signals (e.g., button clicks, slider changes)
    to the corresponding slots in the WebGPU scene to manipulate the 3D object.

    Attributes:
        colour_update (Signal): A signal that emits RGB float values when a new color is selected.
        scene (WebGPUScene): The WebGPU widget where the 3D scene is rendered.
    """

    colour_update = Signal(float, float, float)

    def __init__(self) -> None:
        """Initialize the MainWindow with UI setup and configuration loading."""
        super().__init__()
        self._light_q_colour = QColor("white")
        self.load_ui()
        self.scene = WebGPUScene()
        self.centralWidget().layout().addWidget(self.scene, 0, 0, 3, 1)
        self.resize(1024, 720)
        self._connect_slots()
        self._update_light_properties()

    def _connect_slots(self) -> None:
        """Connect UI element signals to their corresponding slots."""
        self.position_x.valueChanged.connect(self._set_model_position)
        self.position_y.valueChanged.connect(self._set_model_position)
        self.position_z.valueChanged.connect(self._set_model_position)
        self.scale_x.valueChanged.connect(self._set_model_scale)
        self.scale_y.valueChanged.connect(self._set_model_scale)
        self.scale_z.valueChanged.connect(self._set_model_scale)
        self.rotation_x.valueChanged.connect(self._set_model_rotation)
        self.rotation_y.valueChanged.connect(self._set_model_rotation)
        self.rotation_z.valueChanged.connect(self._set_model_rotation)
        self.colour_button.clicked.connect(self._select_colour)
        self.metallic.valueChanged.connect(self.scene.set_metallic)
        self.roughness.valueChanged.connect(self.scene.set_roughness)
        self.ao.valueChanged.connect(self.scene.set_ao)
        self.colour_update.connect(self.scene.set_colour)
        self.light_x.valueChanged.connect(self._update_light_properties)
        self.light_y.valueChanged.connect(self._update_light_properties)
        self.light_z.valueChanged.connect(self._update_light_properties)
        self.light_colour.clicked.connect(self._set_light_colour_from_dialog)
        self.colour_scale.valueChanged.connect(self._update_light_properties)

    def _select_colour(self) -> None:
        """Open a color dialog and emit the selected color for the model."""
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.colour_update.emit(colour.redF(), colour.greenF(), colour.blueF())

    def _set_light_colour_from_dialog(self) -> None:
        """Open a color dialog and set the light color."""
        colour = QColorDialog.getColor(self._light_q_colour)
        if colour.isValid():
            self._light_q_colour = colour
            self._update_light_properties()

    def _update_light_properties(self) -> None:
        """Set the light's properties based on the UI controls."""
        scale = self.colour_scale.value()
        light_colour = Vec3(
            self._light_q_colour.redF() * scale,
            self._light_q_colour.greenF() * scale,
            self._light_q_colour.blueF() * scale,
        )
        light_pos = Vec3(
            self.light_x.value(), self.light_y.value(), self.light_z.value()
        )
        self.scene.set_light(light_pos, light_colour)

    def _set_model_position(self) -> None:
        """Set the model's position based on the UI's position sliders."""
        self.scene.set_model_position(
            self.position_x.value(), self.position_y.value(), self.position_z.value()
        )

    def _set_model_scale(self) -> None:
        """Set the model's scale based on the UI's scale sliders."""
        self.scene.set_model_scale(
            self.scale_x.value(), self.scale_y.value(), self.scale_z.value()
        )

    def _set_model_rotation(self) -> None:
        """Set the model's rotation based on the UI's rotation sliders."""
        self.scene.set_model_rotation(
            self.rotation_x.value(), self.rotation_y.value(), self.rotation_z.value()
        )

    def load_ui(self) -> None:
        """Load the UI from a .ui file and set up the connections."""
        try:
            loader = QUiLoader()
            ui_file = QFile("MainWindow.ui")
            ui_file.open(QFile.ReadOnly)
            loaded_ui = loader.load(ui_file, self)
            self.setCentralWidget(loaded_ui)
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


def main():
    """Main application entry point."""
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()

        if args.smoketest is not None:
            QTimer.singleShot(
                args.smoketest, lambda: (print("SMOKETEST OK"), app.quit())
            )

        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
