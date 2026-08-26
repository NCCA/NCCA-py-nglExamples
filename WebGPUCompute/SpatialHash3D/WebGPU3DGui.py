#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from WebGPU3D import (
    GRID_CELL_SIZE,
    PARTICLE_RADIUS,
    SIM_DEPTH,
    SIM_HEIGHT,
    SIM_WIDTH,
    WebGPUScene3D,
)


class WebGPUControlPanel3D(QWidget):
    """Control panel for WebGPU3D simulation parameters."""

    parameter_changed = Signal()
    reset_simulation = Signal()

    def __init__(self, webgpu_widget):
        super().__init__()
        self.webgpu_widget = webgpu_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Simulation Parameters Group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QGridLayout()

        # Particle count
        sim_layout.addWidget(QLabel("Particles:"), 0, 0)
        self.particle_spinbox = QSpinBox()
        self.particle_spinbox.setRange(100, 2500000)
        self.particle_spinbox.setSingleStep(1000)
        self.particle_spinbox.setValue(self.webgpu_widget.num_points)
        self.particle_spinbox.valueChanged.connect(self.on_particle_count_changed)
        sim_layout.addWidget(self.particle_spinbox, 0, 1)

        # Distribution type
        sim_layout.addWidget(QLabel("Distribution:"), 1, 0)
        self.distribution_combo = QComboBox()
        self.distribution_combo.addItems(["random", "equispaced"])
        self.distribution_combo.currentTextChanged.connect(self.on_distribution_changed)
        sim_layout.addWidget(self.distribution_combo, 1, 1)

        # SIM_WIDTH
        sim_layout.addWidget(QLabel("Sim Width:"), 2, 0)
        self.sim_width_spinbox = QDoubleSpinBox()
        self.sim_width_spinbox.setRange(100.0, 2000.0)
        self.sim_width_spinbox.setSingleStep(50.0)
        self.sim_width_spinbox.setValue(SIM_WIDTH)
        self.sim_width_spinbox.setSuffix(" px")
        self.sim_width_spinbox.valueChanged.connect(self.on_sim_width_changed)
        sim_layout.addWidget(self.sim_width_spinbox, 2, 1)

        # SIM_HEIGHT
        sim_layout.addWidget(QLabel("Sim Height:"), 3, 0)
        self.sim_height_spinbox = QDoubleSpinBox()
        self.sim_height_spinbox.setRange(100.0, 2000.0)
        self.sim_height_spinbox.setSingleStep(50.0)
        self.sim_height_spinbox.setValue(SIM_HEIGHT)
        self.sim_height_spinbox.setSuffix(" px")
        self.sim_height_spinbox.valueChanged.connect(self.on_sim_height_changed)
        sim_layout.addWidget(self.sim_height_spinbox, 3, 1)

        # SIM_DEPTH
        sim_layout.addWidget(QLabel("Sim Depth:"), 4, 0)
        self.sim_depth_spinbox = QDoubleSpinBox()
        self.sim_depth_spinbox.setRange(100.0, 2000.0)
        self.sim_depth_spinbox.setSingleStep(50.0)
        self.sim_depth_spinbox.setValue(SIM_DEPTH)
        self.sim_depth_spinbox.setSuffix(" px")
        self.sim_depth_spinbox.valueChanged.connect(self.on_sim_depth_changed)
        sim_layout.addWidget(self.sim_depth_spinbox, 4, 1)

        # Grid cell size
        sim_layout.addWidget(QLabel("Grid Cell Size:"), 5, 0)
        self.grid_size_spinbox = QDoubleSpinBox()
        self.grid_size_spinbox.setRange(5.0, 250.0)
        self.grid_size_spinbox.setSingleStep(1.0)
        self.grid_size_spinbox.setValue(GRID_CELL_SIZE)
        self.grid_size_spinbox.setSuffix(" px")
        self.grid_size_spinbox.valueChanged.connect(self.on_grid_size_changed)
        sim_layout.addWidget(self.grid_size_spinbox, 5, 1)

        # Particle radius
        sim_layout.addWidget(QLabel("Particle Radius:"), 6, 0)
        self.particle_radius_spinbox = QDoubleSpinBox()
        self.particle_radius_spinbox.setRange(0.1, 15.0)
        self.particle_radius_spinbox.setSingleStep(0.1)
        self.particle_radius_spinbox.setValue(PARTICLE_RADIUS)
        self.particle_radius_spinbox.setSuffix(" px")
        self.particle_radius_spinbox.valueChanged.connect(
            self.on_particle_radius_changed
        )
        sim_layout.addWidget(self.particle_radius_spinbox, 6, 1)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Physics Parameters Group
        physics_group = QGroupBox("Physics Parameters")
        physics_layout = QGridLayout()

        # Wind X
        physics_layout.addWidget(QLabel("Wind X:"), 0, 0)
        self.wind_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_x_slider.setRange(-500, 500)
        self.wind_x_slider.setValue(int(self.webgpu_widget.wind[0] * 100))
        self.wind_x_slider.valueChanged.connect(self.on_wind_x_changed)
        physics_layout.addWidget(self.wind_x_slider, 0, 1)
        self.wind_x_label = QLabel(f"{self.webgpu_widget.wind[0]:.2f}")
        physics_layout.addWidget(self.wind_x_label, 0, 2)

        # Wind Y
        physics_layout.addWidget(QLabel("Wind Y:"), 1, 0)
        self.wind_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_y_slider.setRange(-500, 500)
        self.wind_y_slider.setValue(int(self.webgpu_widget.wind[1] * 100))
        self.wind_y_slider.valueChanged.connect(self.on_wind_y_changed)
        physics_layout.addWidget(self.wind_y_slider, 1, 1)
        self.wind_y_label = QLabel(f"{self.webgpu_widget.wind[1]:.2f}")
        physics_layout.addWidget(self.wind_y_label, 1, 2)

        # Wind Z
        physics_layout.addWidget(QLabel("Wind Z:"), 2, 0)
        self.wind_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.wind_z_slider.setRange(-500, 500)
        self.wind_z_slider.setValue(int(self.webgpu_widget.wind[2] * 100))
        self.wind_z_slider.valueChanged.connect(self.on_wind_z_changed)
        physics_layout.addWidget(self.wind_z_slider, 2, 1)
        self.wind_z_label = QLabel(f"{self.webgpu_widget.wind[2]:.2f}")
        physics_layout.addWidget(self.wind_z_label, 2, 2)

        # Reset wind button
        reset_wind_btn = QPushButton("Reset Wind")
        reset_wind_btn.clicked.connect(self.reset_wind)
        physics_layout.addWidget(reset_wind_btn, 3, 0, 1, 3)

        physics_group.setLayout(physics_layout)
        layout.addWidget(physics_group)

        # Display Parameters Group
        display_group = QGroupBox("Display Parameters")
        display_layout = QGridLayout()

        # Animation checkbox
        self.animate_checkbox = QCheckBox("Animate")
        self.animate_checkbox.setChecked(self.webgpu_widget.animate)
        self.animate_checkbox.toggled.connect(self.on_animate_toggled)
        display_layout.addWidget(self.animate_checkbox, 0, 0)

        # Show grid checkbox
        self.show_grid_checkbox = QCheckBox("Show Grid")
        self.show_grid_checkbox.setChecked(self.webgpu_widget.show_grid)
        self.show_grid_checkbox.toggled.connect(self.on_show_grid_toggled)
        display_layout.addWidget(self.show_grid_checkbox, 0, 1)

        # Point size
        display_layout.addWidget(QLabel("Sphere Scale:"), 1, 0)
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(1, 500)
        self.point_size_slider.setValue(int(self.webgpu_widget.point_size))
        self.point_size_slider.valueChanged.connect(self.on_point_size_changed)
        display_layout.addWidget(self.point_size_slider, 1, 1)
        self.point_size_label = QLabel(f"{self.webgpu_widget.point_size:.1f}")
        display_layout.addWidget(self.point_size_label, 1, 2)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Camera Controls Group
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QGridLayout()

        # Camera distance
        camera_layout.addWidget(QLabel("Distance:"), 0, 0)
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(100, 5000)
        self.distance_slider.setValue(int(self.webgpu_widget.camera_distance))
        self.distance_slider.valueChanged.connect(self.on_distance_changed)
        camera_layout.addWidget(self.distance_slider, 0, 1)
        self.distance_label = QLabel(f"{self.webgpu_widget.camera_distance:.0f}")
        camera_layout.addWidget(self.distance_label, 0, 2)

        # Rotation X
        camera_layout.addWidget(QLabel("Rotation X:"), 1, 0)
        self.rotation_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_x_slider.setRange(-89, 89)
        self.rotation_x_slider.setValue(int(self.webgpu_widget.rotation_x))
        self.rotation_x_slider.valueChanged.connect(self.on_rotation_x_changed)
        camera_layout.addWidget(self.rotation_x_slider, 1, 1)
        self.rotation_x_label = QLabel(f"{self.webgpu_widget.rotation_x:.0f}°")
        camera_layout.addWidget(self.rotation_x_label, 1, 2)

        # Rotation Y
        camera_layout.addWidget(QLabel("Rotation Y:"), 2, 0)
        self.rotation_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_y_slider.setRange(-180, 180)
        self.rotation_y_slider.setValue(int(self.webgpu_widget.rotation_y))
        self.rotation_y_slider.valueChanged.connect(self.on_rotation_y_changed)
        camera_layout.addWidget(self.rotation_y_slider, 2, 1)
        self.rotation_y_label = QLabel(f"{self.webgpu_widget.rotation_y:.0f}°")
        camera_layout.addWidget(self.rotation_y_label, 2, 2)

        # Reset camera button
        reset_camera_btn = QPushButton("Reset Camera")
        reset_camera_btn.clicked.connect(self.reset_camera)
        camera_layout.addWidget(reset_camera_btn, 3, 0, 1, 3)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Action Buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        regenerate_btn = QPushButton("Regenerate Particles")
        regenerate_btn.clicked.connect(self.regenerate_particles)
        action_layout.addWidget(regenerate_btn)

        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self.reset_all)
        action_layout.addWidget(reset_all_btn)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # Instructions
        instructions_group = QGroupBox("Controls")
        instructions_layout = QVBoxLayout()

        instructions_text = QTextEdit()
        instructions_text.setReadOnly(True)
        instructions_text.setMaximumHeight(120)
        instructions_text.setFont(QFont("Courier", 9))
        instructions_text.setPlainText(
            "Mouse:\n"
            "  Left Drag: Rotate camera\n"
            "  Wheel: Zoom in/out\n\n"
            "Keyboard:\n"
            "  A: Toggle animation\n"
            "  G: Toggle grid\n"
            "  Space: Reset camera & wind\n"
            "  Arrow Keys: Wind X/Y\n"
            "  Page Up/Down: Wind Z\n"
            "  ESC: Exit"
        )
        instructions_layout.addWidget(instructions_text)

        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        layout.addStretch()
        self.setLayout(layout)

    def on_particle_count_changed(self, value):
        self.webgpu_widget.num_points = value
        self.webgpu_widget.gen_points(value, self.distribution_combo.currentText())
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_distribution_changed(self, value):
        distribution = value
        self.webgpu_widget.gen_points(self.webgpu_widget.num_points, distribution)
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_sim_width_changed(self, value):
        import WebGPU3D

        WebGPU3D.SIM_WIDTH = value
        self.webgpu_widget.gen_points(
            self.webgpu_widget.num_points, self.distribution_combo.currentText()
        )
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_sim_height_changed(self, value):
        import WebGPU3D

        WebGPU3D.SIM_HEIGHT = value
        self.webgpu_widget.gen_points(
            self.webgpu_widget.num_points, self.distribution_combo.currentText()
        )
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_sim_depth_changed(self, value):
        import WebGPU3D

        WebGPU3D.SIM_DEPTH = value
        self.webgpu_widget.gen_points(
            self.webgpu_widget.num_points, self.distribution_combo.currentText()
        )
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_grid_size_changed(self, value):
        import WebGPU3D

        WebGPU3D.GRID_CELL_SIZE = value
        self.webgpu_widget.gen_points(
            self.webgpu_widget.num_points, self.distribution_combo.currentText()
        )
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_particle_radius_changed(self, value):
        import WebGPU3D

        WebGPU3D.PARTICLE_RADIUS = value
        self.webgpu_widget.update_simulation_params()
        self.parameter_changed.emit()

    def on_wind_x_changed(self, value):
        wind_x = value / 100.0
        self.webgpu_widget.wind[0] = wind_x
        self.wind_x_label.setText(f"{wind_x:.2f}")
        self.parameter_changed.emit()

    def on_wind_y_changed(self, value):
        wind_y = value / 100.0
        self.webgpu_widget.wind[1] = wind_y
        self.wind_y_label.setText(f"{wind_y:.2f}")
        self.parameter_changed.emit()

    def on_wind_z_changed(self, value):
        wind_z = value / 100.0
        self.webgpu_widget.wind[2] = wind_z
        self.wind_z_label.setText(f"{wind_z:.2f}")
        self.parameter_changed.emit()

    def on_animate_toggled(self, checked):
        self.webgpu_widget.animate = checked
        self.parameter_changed.emit()

    def on_show_grid_toggled(self, checked):
        self.webgpu_widget.show_grid = checked

    def on_point_size_changed(self, value):
        point_size = value / 1.0
        self.webgpu_widget.point_size = point_size
        self.point_size_label.setText(f"{point_size:.1f}")

    def on_distance_changed(self, value):
        distance = float(value)
        self.webgpu_widget.camera_distance = distance
        self.distance_label.setText(f"{distance:.0f}")
        self.webgpu_widget.update()

    def on_rotation_x_changed(self, value):
        rotation_x = float(value)
        self.webgpu_widget.rotation_x = rotation_x
        self.rotation_x_label.setText(f"{rotation_x:.0f}°")
        self.webgpu_widget.update()

    def on_rotation_y_changed(self, value):
        rotation_y = float(value)
        self.webgpu_widget.rotation_y = rotation_y
        self.rotation_y_label.setText(f"{rotation_y:.0f}°")
        self.webgpu_widget.update()

    def reset_wind(self):
        self.webgpu_widget.wind[0] = 0.0
        self.webgpu_widget.wind[1] = 0.0
        self.webgpu_widget.wind[2] = 0.0
        self.wind_x_slider.setValue(0)
        self.wind_y_slider.setValue(0)
        self.wind_z_slider.setValue(0)
        self.parameter_changed.emit()

    def reset_camera(self):
        self.webgpu_widget.camera_distance = 1500.0
        self.webgpu_widget.rotation_x = 30.0
        self.webgpu_widget.rotation_y = 45.0
        self.distance_slider.setValue(1500)
        self.rotation_x_slider.setValue(30)
        self.rotation_y_slider.setValue(45)
        self.webgpu_widget.update()

    def regenerate_particles(self):
        distribution = self.distribution_combo.currentText()
        self.webgpu_widget.gen_points(self.webgpu_widget.num_points, distribution)
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def reset_all(self):
        self.reset_wind()
        self.reset_camera()
        self.particle_spinbox.setValue(1000)
        self.distribution_combo.setCurrentText("random")
        self.sim_width_spinbox.setValue(SIM_WIDTH)
        self.sim_height_spinbox.setValue(SIM_HEIGHT)
        self.sim_depth_spinbox.setValue(SIM_DEPTH)
        self.grid_size_spinbox.setValue(GRID_CELL_SIZE)
        self.particle_radius_spinbox.setValue(PARTICLE_RADIUS)
        self.animate_checkbox.setChecked(False)
        self.show_grid_checkbox.setChecked(True)
        self.point_size_slider.setValue(2)
        self.regenerate_particles()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


class WebGPU3DGui(QMainWindow):
    """Main GUI window for WebGPU3D simulation."""

    def __init__(self, num_points=1000, distribution="random"):
        super().__init__()
        self.setWindowTitle("WebGPU 3D Particle Simulation - GUI")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Create WebGPU widget
        self.webgpu_widget = WebGPUScene3D(
            num_points=num_points, distribution=distribution
        )
        self.webgpu_widget.setMinimumSize(800, 600)
        self.webgpu_widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        splitter.addWidget(self.webgpu_widget)

        # Create control panel
        self.control_panel = WebGPUControlPanel3D(self.webgpu_widget)
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setMinimumWidth(300)
        splitter.addWidget(self.control_panel)

        # Set splitter sizes (70% for WebGPU, 30% for controls)
        splitter.setSizes([980, 420])

        # Connect signals
        self.control_panel.reset_simulation.connect(self.reset_simulation)
        self.control_panel.parameter_changed.connect(self.on_parameter_changed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_R:
            self.reset_simulation()
        elif event.key() == Qt.Key.Key_Space:
            self.control_panel.animate_checkbox.click()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            # Pass unhandled keys to WebGPU widget
            super().keyPressEvent(event)

    def reset_simulation(self):
        """Reset the simulation with current parameters."""
        try:
            # Reinitialize WebGPU components
            self.webgpu_widget._initialize_web_gpu()
            self.webgpu_widget.update()
        except Exception as e:  # noqa: BLE001 - reset needs the backend traceback.
            print(f"Error resetting simulation: {e}")
            traceback.print_exc()

    def on_parameter_changed(self):
        """Handle parameter changes that don't require full reset."""
        self.webgpu_widget.update()


def main():
    """Main function to run the GUI application."""
    parser = argparse.ArgumentParser(
        description="WebGPU 3D Particle Simulation with GUI"
    )
    parser.add_argument(
        "-p", "--points", type=int, default=1000, help="Initial number of particles"
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_const",
        dest="distribution",
        const="random",
        help="Random particle distribution",
    )
    parser.add_argument(
        "-e",
        "--equispaced",
        action="store_const",
        dest="distribution",
        const="equispaced",
        help="Equispaced particle distribution",
    )
    parser.set_defaults(distribution="random")
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

    try:
        window = WebGPU3DGui(num_points=args.points, distribution=args.distribution)
        window.show()
        if args.smoketest is not None:
            QTimer.singleShot(
                args.smoketest, lambda: (print("SMOKETEST OK"), app.quit())
            )
        return_code = app.exec()
        sys.exit(return_code)
    except Exception as e:  # noqa: BLE001 - report any startup failure to the user.
        print(f"Application error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
