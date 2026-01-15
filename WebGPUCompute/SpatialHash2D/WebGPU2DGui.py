#!/usr/bin/env -S uv run --active --script
import sys
import traceback

import numpy as np
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
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from WebGPU2D import WebGPUScene


class WebGPUControlPanel(QWidget):
    """Control panel for WebGPU2D simulation parameters."""

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
        self.particle_spinbox.setRange(100, 500000)
        self.particle_spinbox.setSingleStep(100)
        self.particle_spinbox.setValue(self.webgpu_widget.num_points)
        self.particle_spinbox.valueChanged.connect(self.on_particle_count_changed)
        sim_layout.addWidget(self.particle_spinbox, 0, 1)

        # Distribution type
        sim_layout.addWidget(QLabel("Distribution:"), 1, 0)
        self.distribution_combo = QComboBox()
        self.distribution_combo.addItems(["random", "equispaced"])
        self.distribution_combo.currentTextChanged.connect(self.on_distribution_changed)
        sim_layout.addWidget(self.distribution_combo, 1, 1)

        # Grid cell size
        sim_layout.addWidget(QLabel("Grid Cell Size:"), 2, 0)
        self.grid_size_spinbox = QDoubleSpinBox()
        self.grid_size_spinbox.setRange(5.0, 250.0)
        self.grid_size_spinbox.setSingleStep(1.0)
        self.grid_size_spinbox.setValue(50.0)
        self.grid_size_spinbox.setSuffix(" px")
        self.grid_size_spinbox.valueChanged.connect(self.on_grid_size_changed)
        sim_layout.addWidget(self.grid_size_spinbox, 2, 1)

        # Particle radius
        sim_layout.addWidget(QLabel("Particle Radius:"), 3, 0)
        self.particle_radius_spinbox = QDoubleSpinBox()
        self.particle_radius_spinbox.setRange(0.1, 15.0)
        self.particle_radius_spinbox.setSingleStep(0.1)
        self.particle_radius_spinbox.setValue(1.0)
        self.particle_radius_spinbox.setSuffix(" px")
        self.particle_radius_spinbox.valueChanged.connect(
            self.on_particle_radius_changed
        )
        sim_layout.addWidget(self.particle_radius_spinbox, 3, 1)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Physics Parameters Group
        physics_group = QGroupBox("Physics Parameters")
        physics_layout = QGridLayout()

        # Wind X
        physics_layout.addWidget(QLabel("Wind X:"), 0, 0)
        self.wind_x_slider = QSlider(Qt.Horizontal)
        self.wind_x_slider.setRange(-100, 100)
        self.wind_x_slider.setValue(int(self.webgpu_widget.wind[0] * 100))
        self.wind_x_slider.valueChanged.connect(self.on_wind_x_changed)
        physics_layout.addWidget(self.wind_x_slider, 0, 1)
        self.wind_x_label = QLabel(f"{self.webgpu_widget.wind[0]:.2f}")
        physics_layout.addWidget(self.wind_x_label, 0, 2)

        # Wind Y
        physics_layout.addWidget(QLabel("Wind Y:"), 1, 0)
        self.wind_y_slider = QSlider(Qt.Horizontal)
        self.wind_y_slider.setRange(-100, 100)
        self.wind_y_slider.setValue(int(self.webgpu_widget.wind[1] * 100))
        self.wind_y_slider.valueChanged.connect(self.on_wind_y_changed)
        physics_layout.addWidget(self.wind_y_slider, 1, 1)
        self.wind_y_label = QLabel(f"{self.webgpu_widget.wind[1]:.2f}")
        physics_layout.addWidget(self.wind_y_label, 1, 2)

        # Reset wind button
        reset_wind_btn = QPushButton("Reset Wind")
        reset_wind_btn.clicked.connect(self.reset_wind)
        physics_layout.addWidget(reset_wind_btn, 2, 0, 1, 3)

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

        # Show numbers checkbox
        self.show_numbers_checkbox = QCheckBox("Show Cell Counts")
        self.show_numbers_checkbox.setChecked(self.webgpu_widget.show_numbers)
        self.show_numbers_checkbox.toggled.connect(self.on_show_numbers_toggled)
        display_layout.addWidget(self.show_numbers_checkbox, 1, 0)

        # Point size
        display_layout.addWidget(QLabel("Point Size:"), 2, 0)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(int(self.webgpu_widget.point_size))
        self.point_size_slider.valueChanged.connect(self.on_point_size_changed)
        display_layout.addWidget(self.point_size_slider, 2, 1)
        self.point_size_label = QLabel(f"{self.webgpu_widget.point_size:.1f}")
        display_layout.addWidget(self.point_size_label, 2, 2)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # View Controls Group
        view_group = QGroupBox("View Controls")
        view_layout = QGridLayout()

        # Zoom controls
        view_layout.addWidget(QLabel("Zoom:"), 0, 0)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(5, 200)  # 0.05 to 2.0
        self.zoom_slider.setValue(int(self.webgpu_widget.zoom * 100))
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        view_layout.addWidget(self.zoom_slider, 0, 1)
        self.zoom_label = QLabel(f"{self.webgpu_widget.zoom:.2f}")
        view_layout.addWidget(self.zoom_label, 0, 2)

        # Reset view button
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.clicked.connect(self.reset_view)
        view_layout.addWidget(reset_view_btn, 1, 0, 1, 3)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

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

        # Statistics Display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setFont(QFont("Courier", 9))
        stats_layout.addWidget(self.stats_text)

        # Update stats timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(500)  # Update every 500ms

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        self.setLayout(layout)

    def on_particle_count_changed(self, value):
        self.webgpu_widget.num_points = value
        self.reset_simulation.emit()

    def on_distribution_changed(self, value):
        distribution = value
        self.webgpu_widget.gen_points(self.webgpu_widget.num_points, distribution)
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def on_grid_size_changed(self, value):
        # Note: This would require recreating buffers, so emit reset
        self.reset_simulation.emit()

    def on_particle_radius_changed(self, value):
        # Note: This would require updating sim params, so emit parameter change
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

    def on_animate_toggled(self, checked):
        self.webgpu_widget.animate = checked

    def on_show_grid_toggled(self, checked):
        self.webgpu_widget.show_grid = checked

    def on_show_numbers_toggled(self, checked):
        self.webgpu_widget.show_numbers = checked

    def on_point_size_changed(self, value):
        point_size = value / 1.0
        self.webgpu_widget.point_size = point_size
        self.point_size_label.setText(f"{point_size:.1f}")

    def on_zoom_changed(self, value):
        zoom = value / 100.0
        self.webgpu_widget.zoom = zoom
        self.zoom_label.setText(f"{zoom:.2f}")
        self.webgpu_widget.update()

    def reset_wind(self):
        self.webgpu_widget.wind[0] = 0.0
        self.webgpu_widget.wind[1] = 0.0
        self.wind_x_slider.setValue(0)
        self.wind_y_slider.setValue(0)
        self.parameter_changed.emit()

    def reset_view(self):
        self.webgpu_widget.zoom = 1.0
        self.webgpu_widget.pan[:] = 0.0
        self.zoom_slider.setValue(100)
        self.webgpu_widget.update()

    def regenerate_particles(self):
        distribution = self.distribution_combo.currentText()
        self.webgpu_widget.gen_points(self.webgpu_widget.num_points, distribution)
        self.webgpu_widget._init_buffers()
        self.reset_simulation.emit()

    def reset_all(self):
        self.reset_wind()
        self.reset_view()
        self.particle_spinbox.setValue(1000)
        self.distribution_combo.setCurrentText("random")
        self.grid_size_spinbox.setValue(20.0)
        self.particle_radius_spinbox.setValue(1.0)
        self.animate_checkbox.setChecked(False)
        self.show_grid_checkbox.setChecked(True)
        self.show_numbers_checkbox.setChecked(True)
        self.point_size_slider.setValue(1)
        self.regenerate_particles()

    def update_statistics(self):
        try:
            if hasattr(self.webgpu_widget, "read_cell_particle_counts"):
                cell_counts = self.webgpu_widget.read_cell_particle_counts()
                total_particles = np.sum(cell_counts)
                max_in_cell = np.max(cell_counts)
                avg_per_cell = np.mean(cell_counts)
                non_empty_cells = np.count_nonzero(cell_counts)
                total_cells = cell_counts.size

                stats_text = f"""Particles: {total_particles}
Max in cell: {max_in_cell}
Avg per cell: {avg_per_cell:.2f}
Non-empty cells: {non_empty_cells}/{total_cells}
Occupancy: {(non_empty_cells / total_cells) * 100:.1f}%
FPS: {1.0 / self.webgpu_widget.dt if self.webgpu_widget.dt > 0 else 0:.1f}
Wind: [{self.webgpu_widget.wind[0]:.2f}, {self.webgpu_widget.wind[1]:.2f}]
Zoom: {self.webgpu_widget.zoom:.2f}"""

                self.stats_text.setPlainText(stats_text)
        except Exception as e:
            self.stats_text.setPlainText(f"Error updating stats: {e}")


class WebGPU2DGui(QMainWindow):
    """Main GUI window for WebGPU2D simulation."""

    def __init__(self, num_points=1000, distribution="random"):
        super().__init__()
        self.setWindowTitle("WebGPU 2D Particle Simulation - GUI")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create WebGPU widget
        self.webgpu_widget = WebGPUScene(
            num_points=num_points, distribution=distribution
        )
        self.webgpu_widget.setMinimumSize(800, 600)
        splitter.addWidget(self.webgpu_widget)

        # Create control panel
        self.control_panel = WebGPUControlPanel(self.webgpu_widget)
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setMinimumWidth(300)
        splitter.addWidget(self.control_panel)

        # Set splitter sizes (70% for WebGPU, 30% for controls)
        splitter.setSizes([980, 420])

        # Connect signals
        self.control_panel.reset_simulation.connect(self.reset_simulation)
        self.control_panel.parameter_changed.connect(self.on_parameter_changed)

    def reset_simulation(self):
        """Reset the simulation with current parameters."""
        try:
            # Reinitialize WebGPU components
            self.webgpu_widget._initialize_web_gpu()
            self.webgpu_widget.update()
        except Exception as e:
            print(f"Error resetting simulation: {e}")
            traceback.print_exc()

    def on_parameter_changed(self):
        """Handle parameter changes that don't require full reset."""
        self.webgpu_widget.update()


def main():
    """Main function to run the GUI application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WebGPU 2D Particle Simulation with GUI"
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
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        window = WebGPU2DGui(num_points=args.points, distribution=args.distribution)
        window.show()
        return_code = app.exec()
        sys.exit(return_code)
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
