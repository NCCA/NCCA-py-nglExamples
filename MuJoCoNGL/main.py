#!/usr/bin/env -S uv run --script
"""MuJoCo physics with PyNGL, a port of the C++ BulletNGL demo.

Same demo as the Bullet one -- press the number keys, watch things fall on a
ground plane -- but with MuJoCo underneath, which does not let you add a body to
a model once it has been compiled. There are two ways round that and the panel
switches between them live, with the spawn cost in milliseconds next to the
switch so you can see what each one costs.

All of the original key bindings still work:

    1-7      drop a box, sphere, capsule, cylinder, cone, teapot or apple
    arrows   shove everything left, right, up or down
    space    pause and resume
    x        single step whilst paused
    r        toggle random placement
    0        reset
    w / s    wireframe on and off
    f / n    fullscreen and windowed
    escape   quit
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import mujoco
from collision_shapes import SPAWN_HEIGHT, ShapeCatalogue
from ncca.ngl import Vec3
from physics_world import DEFAULT_GRAVITY, make_world
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
from scene import MuJoCoScene

DEMO_DIR = Path(__file__).resolve().parent

# MuJoCo's integrators, in the order its enum defines them. Euler is the
# default; RK4 is steadier at a large timestep and costs four evaluations a
# step, which is a fair thing to let students feel for themselves.
INTEGRATORS = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


class DebugApplication(QApplication):
    """Qt swallows exceptions in event handlers, this puts the traceback back."""

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


class MainWindow(QMainWindow):
    """The scene, the world driving it, and the panel driving that."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MuJoCo Physics and PyNGL")

        self.catalogue = ShapeCatalogue.default(model_dir=str(DEMO_DIR / "models"))
        self.world = make_world("recompile", self.catalogue, DEFAULT_GRAVITY)
        self.scene = MuJoCoScene(self.world, model_dir=str(DEMO_DIR / "models"))

        self.animate = True
        self.random_place = False

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.scene, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(10)
        self.scene.setFocus()

    # ---------------------------------------------------------------- panel
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(240)
        outer = QVBoxLayout(panel)

        spawn = QGroupBox("Spawn")
        spawn_layout = QVBoxLayout(spawn)
        for index, name in enumerate(self.catalogue.names, start=1):
            button = QPushButton(f"{index}  {name}")
            button.clicked.connect(lambda _, n=name: self.spawn(n))
            button.setFocusPolicy(Qt.NoFocus)
            spawn_layout.addWidget(button)
        outer.addWidget(spawn)

        strategy = QGroupBox("Spawning strategy")
        strategy_layout = QVBoxLayout(strategy)
        self.strategy = QComboBox()
        self.strategy.addItems(["recompile", "pool"])
        self.strategy.currentTextChanged.connect(self._change_strategy)
        self.strategy.setFocusPolicy(Qt.NoFocus)
        strategy_layout.addWidget(self.strategy)
        self.spawn_cost = QLabel("last spawn: -")
        strategy_layout.addWidget(self.spawn_cost)
        outer.addWidget(strategy)

        solver = QGroupBox("Solver")
        grid = QGridLayout(solver)
        self.integrator = QComboBox()
        self.integrator.addItems(list(INTEGRATORS))
        self.integrator.currentTextChanged.connect(self._change_integrator)
        self.integrator.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(QLabel("integrator"), 0, 0)
        grid.addWidget(self.integrator, 0, 1)

        self.timestep = QDoubleSpinBox()
        self.timestep.setRange(0.0005, 0.05)
        self.timestep.setSingleStep(0.001)
        self.timestep.setDecimals(4)
        self.timestep.setValue(1.0 / 60.0)
        self.timestep.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(QLabel("dt"), 1, 0)
        grid.addWidget(self.timestep, 1, 1)

        self.substeps = QSpinBox()
        self.substeps.setRange(1, 20)
        self.substeps.setValue(4)
        self.substeps.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(QLabel("substeps"), 2, 0)
        grid.addWidget(self.substeps, 2, 1)

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 200)
        self.iterations.setValue(self.world.solver_iterations)
        self.iterations.valueChanged.connect(self._change_iterations)
        self.iterations.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(QLabel("iterations"), 3, 0)
        grid.addWidget(self.iterations, 3, 1)

        self.gravity = QDoubleSpinBox()
        self.gravity.setRange(-50.0, 50.0)
        self.gravity.setValue(DEFAULT_GRAVITY.y)
        self.gravity.valueChanged.connect(self._change_gravity)
        self.gravity.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(QLabel("gravity y"), 4, 0)
        grid.addWidget(self.gravity, 4, 1)
        outer.addWidget(solver)

        controls = QGroupBox("Simulation")
        controls_layout = QVBoxLayout(controls)
        self.pause = QPushButton("Pause")
        self.pause.clicked.connect(self.toggle_animation)
        self.pause.setFocusPolicy(Qt.NoFocus)
        controls_layout.addWidget(self.pause)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)
        reset.setFocusPolicy(Qt.NoFocus)
        controls_layout.addWidget(reset)
        self.random_button = QPushButton("Random placement: off")
        self.random_button.clicked.connect(self.toggle_random)
        self.random_button.setFocusPolicy(Qt.NoFocus)
        controls_layout.addWidget(self.random_button)
        outer.addWidget(controls)

        self.body_count = QLabel("bodies: 0")
        outer.addWidget(self.body_count)
        outer.addStretch(1)
        return panel

    # ------------------------------------------------------------- physics
    def spawn(self, shape: str) -> None:
        if self.random_place:
            position = Vec3(
                random.uniform(-10.0, 10.0),
                SPAWN_HEIGHT,
                random.uniform(-10.0, 10.0),
            )
        else:
            position = Vec3(0.0, SPAWN_HEIGHT, 0.0)
        self.world.add_body(shape, position)
        self.spawn_cost.setText(f"last spawn: {self.world.last_spawn_ms:.2f} ms")
        self._update_labels()

    def _change_strategy(self, name: str) -> None:
        """Rebuilds the world, carrying the current bodies over.

        The two strategies hold completely different models, so switching means
        building a new one. The shapes and where they were dropped come across,
        but not their current state -- everything drops again from the top.
        """
        previous = list(self.world._spawned)
        self.world = make_world(name, self.catalogue, self.world.gravity)
        self.world.integrator = INTEGRATORS[self.integrator.currentText()]
        self.world.solver_iterations = self.iterations.value()
        for shape, position in previous:
            self.world.add_body(shape, position)
        self.scene.world = self.world
        self._update_labels()

    def _change_integrator(self, name: str) -> None:
        self.world.integrator = INTEGRATORS[name]

    def _change_iterations(self, value: int) -> None:
        self.world.solver_iterations = value

    def _change_gravity(self, value: float) -> None:
        self.world.gravity = Vec3(0.0, value, 0.0)

    def reset(self) -> None:
        self.world.reset()
        self._update_labels()
        self.scene.update()

    def toggle_animation(self) -> None:
        self.animate = not self.animate
        self.pause.setText("Resume" if not self.animate else "Pause")

    def toggle_random(self) -> None:
        self.random_place = not self.random_place
        state = "on" if self.random_place else "off"
        self.random_button.setText(f"Random placement: {state}")

    def step_once(self) -> None:
        self.world.step(self.timestep.value(), self.substeps.value())

    def _tick(self) -> None:
        if self.animate:
            self.step_once()
        self.scene.update()

    def _update_labels(self) -> None:
        self.body_count.setText(f"bodies: {self.world.num_bodies}")

    # -------------------------------------------------------------- keyboard
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        names = self.catalogue.names
        number_keys = {
            getattr(Qt, f"Key_{i + 1}"): name for i, name in enumerate(names)
        }

        if key == Qt.Key_Escape:
            self.close()
        elif key in number_keys:
            self.spawn(number_keys[key])
        elif key == Qt.Key_Space:
            self.toggle_animation()
        elif key == Qt.Key_X:
            self.step_once()
        elif key == Qt.Key_R:
            self.toggle_random()
        elif key == Qt.Key_0:
            self.reset()
        elif key == Qt.Key_W:
            self.scene.wireframe = True
        elif key == Qt.Key_S:
            self.scene.wireframe = False
        elif key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Left:
            self.world.add_impulse(Vec3(-5.0, 0.0, 0.0))
        elif key == Qt.Key_Right:
            self.world.add_impulse(Vec3(5.0, 0.0, 0.0))
        elif key == Qt.Key_Up:
            self.world.add_impulse(Vec3(0.0, 5.0, 0.0))
        elif key == Qt.Key_Down:
            self.world.add_impulse(Vec3(0.0, -5.0, 0.0))
        self.scene.update()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="show Qt tracebacks")
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    args = parser.parse_args()

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setMajorVersion(4)
    fmt.setMinorVersion(1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = (DebugApplication if args.debug else QApplication)(sys.argv)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
