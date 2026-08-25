#!/usr/bin/env -S uv run --script
"""Mass spring chain with RK4 integration (PyNGL / PySide6).

A port of the C++ NGL MassSpring demo, with the single spring generalised to
a chain: set the mass count to 2 and this is the original demo, wind it up
and you get a rope. Start and End place the ends of the chain and the masses
in between are spaced evenly along it; pin either end and turn gravity on to
watch it swing.

Controls: left mouse rotates. Everything else is on the panel.
"""

import argparse
import sys
import traceback

from mass_spring import MassSpringChain
from MassSpringScene import MassSpringScene
from ncca.ngl import Vec3
from ncca.ngl.widgets import Vec3Widget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_MAX_MASSES = 32


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
    """Hosts the chain, the scene drawing it and the controls driving it."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mass Spring Chain (RK4)")
        # MassSpringChain's defaults are the C++ demo's, so an empty
        # constructor opens on exactly the original single spring.
        self.chain = MassSpringChain()
        self.chain.set_fix_first(True)
        self.scene = MassSpringScene(self.chain)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.scene, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.resize(1024, 720)

    # ---------------------------------------------------------------- panel
    def _spinbox(
        self, value: float, low: float, high: float, step: float
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(360)
        outer = QVBoxLayout(panel)

        # --- chain shape, using the PyNGL Vec3 widgets for the two ends
        shape = QGroupBox("Chain")
        shape_layout = QVBoxLayout(shape)
        self.start_widget = Vec3Widget(value=Vec3(0.0, 2.0, 0.0))
        self.start_widget.set_name("Start (A)")
        self.start_widget.set_range(-5.0, 5.0)
        self.end_widget = Vec3Widget(value=Vec3(0.0, -2.0, 0.0))
        self.end_widget.set_name("End (B)")
        self.end_widget.set_range(-5.0, 5.0)
        shape_layout.addWidget(self.start_widget)
        shape_layout.addWidget(self.end_widget)

        counts = QGridLayout()
        counts.addWidget(QLabel("Masses"), 0, 0)
        self.masses = QSpinBox()
        self.masses.setRange(2, _MAX_MASSES)
        self.masses.setValue(2)
        counts.addWidget(self.masses, 0, 1)
        self.fix_first = QCheckBox("Fix Start")
        self.fix_first.setChecked(True)
        self.fix_last = QCheckBox("Fix End")
        counts.addWidget(self.fix_first, 1, 0)
        counts.addWidget(self.fix_last, 1, 1)
        shape_layout.addLayout(counts)
        outer.addWidget(shape)

        # --- spring parameters, shared by every spring in the chain
        spring = QGroupBox("Spring")
        grid = QGridLayout(spring)
        self.k = self._spinbox(5.0, 0.0, 100.0, 0.1)
        self.damping = self._spinbox(2.0, 0.0, 10.0, 0.01)
        self.rest_length = self._spinbox(1.0, 0.01, 10.0, 0.01)
        for row, (label, box) in enumerate(
            (
                ("k", self.k),
                ("damping", self.damping),
                ("rest length", self.rest_length),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(box, row, 1)
        outer.addWidget(spring)

        # --- gravity
        gravity = QGroupBox("Gravity")
        gravity_layout = QGridLayout(gravity)
        self.gravity = QCheckBox("Enabled")
        self.gravity_strength = self._spinbox(9.81, 0.0, 50.0, 0.1)
        gravity_layout.addWidget(self.gravity, 0, 0)
        gravity_layout.addWidget(QLabel("strength"), 1, 0)
        gravity_layout.addWidget(self.gravity_strength, 1, 1)
        outer.addWidget(gravity)

        # --- simulation
        sim = QGroupBox("Simulation")
        sim_layout = QGridLayout(sim)
        self.dt = self._spinbox(0.01, 0.001, 1.0, 0.001)
        self.dt.setDecimals(3)
        self.timer_value = QSpinBox()
        self.timer_value.setRange(1, 200)
        self.timer_value.setValue(10)
        sim_layout.addWidget(QLabel("dt"), 0, 0)
        sim_layout.addWidget(self.dt, 0, 1)
        sim_layout.addWidget(QLabel("timer (ms)"), 1, 0)
        sim_layout.addWidget(self.timer_value, 1, 1)
        self.sim_button = QPushButton("Simulate")
        self.sim_button.setCheckable(True)
        self.sim_button.setChecked(True)
        self.reset_button = QPushButton("Reset")
        sim_layout.addWidget(self.sim_button, 2, 0)
        sim_layout.addWidget(self.reset_button, 2, 1)
        outer.addWidget(sim)

        outer.addStretch(1)
        self._connect_slots()
        return panel

    def _connect_slots(self) -> None:
        self.start_widget.valueChanged.connect(self._set_start)
        self.end_widget.valueChanged.connect(self._set_end)
        self.masses.valueChanged.connect(self._set_masses)
        self.fix_first.toggled.connect(self._set_fix_first)
        self.fix_last.toggled.connect(self._set_fix_last)
        self.k.valueChanged.connect(self.chain.set_k)
        self.damping.valueChanged.connect(self.chain.set_damping)
        self.rest_length.valueChanged.connect(self.chain.set_rest_length)
        self.gravity.toggled.connect(self._set_gravity)
        self.gravity_strength.valueChanged.connect(self.chain.set_gravity_strength)
        self.dt.valueChanged.connect(self.chain.set_timestep)
        self.timer_value.valueChanged.connect(self.scene.set_timer_duration)
        self.sim_button.toggled.connect(self.scene.toggle_sim)
        self.reset_button.clicked.connect(self._reset)

    # ---------------------------------------------------------------- slots
    def _set_start(self, value: Vec3) -> None:
        self.chain.set_start(value)
        self.scene.update()

    def _set_end(self, value: Vec3) -> None:
        self.chain.set_end(value)
        self.scene.update()

    def _set_masses(self, count: int) -> None:
        self.chain.set_num_masses(count)
        self.scene.update()

    def _set_fix_first(self, fixed: bool) -> None:
        self.chain.set_fix_first(fixed)
        self.scene.update()

    def _set_fix_last(self, fixed: bool) -> None:
        self.chain.set_fix_last(fixed)
        self.scene.update()

    def _set_gravity(self, enabled: bool) -> None:
        self.chain.set_gravity(enabled)
        self.scene.update()

    def _reset(self) -> None:
        self.chain.reset()
        self.scene.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()


def main() -> None:
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

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)

    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)

    window = MainWindow()
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
