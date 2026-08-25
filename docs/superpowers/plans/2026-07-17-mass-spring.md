# Mass Spring Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the C++ `NGL9Demos/MassSpring` demo to PyNGL + PySide6, generalised from a single spring to a chain of N masses joined by N-1 springs, with a GUI built from `ncca.ngl.widgets`.

**Architecture:** State moves off the spring and onto the masses. `rk4.py` holds a reusable RK4 integrator (ABC + abstract `motion_function`) over `(N,3)` position/velocity arrays; `mass_spring.py` derives from it and turns springs into pure force producers between mass indices, so masses shared by two springs accumulate into one acceleration slot. `MassSpringScene.py` (QOpenGLWidget) draws; `main.py` builds the GUI and owns the chain. At N=2 the demo is exactly the original single spring.

**Tech Stack:** Python 3, `uv`, PySide6, `ncca.ngl` (PyNGL), `ncca.ngl.opengl` (ShaderLib/Primitives/VAOFactory), `ncca.ngl.widgets` (Vec3Widget), numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-17-mass-spring-design.md`

## Global Constraints

- Demo folder is `MassSpring/` at the repo root. Self-contained; no shared code with other demos.
- All commands via `uv` (`uv run ...`). Never `pip`/`python` directly.
- Entry point `MassSpring/main.py` is executable: `chmod +x`, shebang `#!/usr/bin/env -S uv run --script`.
- `main.py` MUST accept `--smoketest [MS]` (default 200) printing `SMOKETEST OK` then exiting, and `--debug`, matching `GUIDemos/NGLWidgetsOpenGL/main.py`.
- GL demos load resources by relative path — run them from their own folder (`cd MassSpring && uv run main.py`).
- Lint with `uv run ruff check .` and `uv run ruff format .` before every commit. Both must pass.
- Physics files (`rk4.py`, `mass_spring.py`) MUST NOT import PySide6 or OpenGL — the tests are headless.
- Tests live in `MassSpring/tests/` and must start with `sys.path.insert(0, str(Path(__file__).parent.parent))` before importing demo modules (repo convention; without it `uv run pytest` from the root fails collection).
- Prose (README, docstrings, comments) uses the jon-writing-style skill: first person, British English, no marketing adjectives, no emoji.
- Work happens in the existing worktree `.worktrees/mass-spring` on branch `agent/mass-spring`. Never commit to `Version1.0`/`main`.

### Two API traps, verified — do not get these wrong

1. **`Prims.CUBE` is NOT a `Primitives.create()` type.** `create()` only takes parametric prims (SPHERE, TORUS, ...). Cube/teapot are *mesh* defaults: call `Primitives.load_default_primitives()` then `Primitives.draw("cube")`. Calling `create(Prims.CUBE, ...)` raises `ValueError` and aborts `initializeGL`.
2. **The built-in diffuse shader is VIEW space** — `diffuse_vertex.glsl` does `fragmentPosition = MV * inVert` and transforms the normal by a `normalMatrix` meant to be built from **MV**. So `Mat3.from_mat4(MV).inverse().transposed()` is **correct here**. This is the opposite of the world-space PBR demos (fixed in commit `8b77482`, where the normal matrix must come from M alone). Do not "fix" the diffuse path to use M.

---

### Task 1: RK4 integrator core

**Files:**
- Create: `MassSpring/rk4.py`
- Test: `MassSpring/tests/test_rk4.py`
- Create: `MassSpring/tests/__init__.py` (empty file)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `State(position: np.ndarray, velocity: np.ndarray)` — dataclass, both `(N,3)` float64.
  - `AbstractRK4Integrator` — ABC. Attribute `state: State`. Abstract `motion_function(self, state: State, t: float) -> np.ndarray` returning `(N,3)` accelerations. Concrete `evaluate(self, initial: State, t: float) -> State`, `evaluate_with_derivative(self, initial: State, t: float, dt: float, d: State) -> State`, `integrate(self, t: float, dt: float) -> None`.

- [ ] **Step 1: Write the failing test**

Create `MassSpring/tests/__init__.py` as an empty file, then `MassSpring/tests/test_rk4.py`:

```python
"""Tests for the RK4 integrator.

The integrator is checked against the analytic simple-harmonic-oscillator
solution rather than a golden array, so these prove the maths is right and
not merely that it runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rk4 import AbstractRK4Integrator, State  # noqa: E402


class _Oscillator(AbstractRK4Integrator):
    """x'' = -x, so x(t) = cos(t) when x(0) = 1 and v(0) = 0."""

    def motion_function(self, state: State, t: float) -> np.ndarray:
        return -state.position


def _oscillator() -> _Oscillator:
    sim = _Oscillator()
    sim.state = State(
        position=np.array([[1.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
    )
    return sim


def test_matches_the_analytic_oscillator_over_a_full_period():
    sim = _oscillator()
    dt = 0.001
    steps = int(round(2.0 * np.pi / dt))
    for step in range(steps):
        sim.integrate(step * dt, dt)
    # after one full period cos(2pi) = 1, back where it started
    assert sim.state.position[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert sim.state.velocity[0, 0] == pytest.approx(0.0, abs=1e-4)


def test_matches_the_analytic_oscillator_at_half_a_period():
    sim = _oscillator()
    dt = 0.001
    steps = int(round(np.pi / dt))
    for step in range(steps):
        sim.integrate(step * dt, dt)
    # cos(pi) = -1
    assert sim.state.position[0, 0] == pytest.approx(-1.0, abs=1e-4)


def test_integrates_every_mass_independently():
    sim = _Oscillator()
    sim.state = State(
        position=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        velocity=np.zeros((2, 3)),
    )
    dt = 0.001
    steps = int(round(np.pi / dt))
    for step in range(steps):
        sim.integrate(step * dt, dt)
    np.testing.assert_allclose(sim.state.position[0], [-1.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(sim.state.position[1], [0.0, -2.0, 0.0], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_rk4.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rk4'`

- [ ] **Step 3: Write minimal implementation**

Create `MassSpring/rk4.py`:

```python
"""A reusable RK4 integrator, ported from the C++ AbstractRK4Integrator.

The structure follows the original (and the tutorials at
http://gafferongames.com/game-physics/): a State of position and velocity,
a motion function supplying accelerations, and four evaluations combined
into the classic Runge-Kutta step. The only change is that position and
velocity are (N,3) arrays rather than a single Vec3, so one integrator can
carry a whole system of masses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """Position and velocity of every mass in the system.

    Attributes
    ----------
        position : np.ndarray
            (N,3) positions
        velocity : np.ndarray
            (N,3) velocities
    """

    position: np.ndarray
    velocity: np.ndarray


class AbstractRK4Integrator(ABC):
    """RK4 integration of any system that can supply accelerations.

    Derive from this and implement motion_function; the integrator does not
    care what the forces mean.
    """

    state: State

    @abstractmethod
    def motion_function(self, state: State, t: float) -> np.ndarray:
        """Return the (N,3) accelerations of the system in `state` at time `t`."""

    def evaluate(self, initial: State, t: float) -> State:
        """Derivative of `initial` at `t`: d(position) is velocity, d(velocity)
        is the acceleration from the motion function."""
        return State(
            position=initial.velocity.copy(),
            velocity=self.motion_function(initial, t),
        )

    def evaluate_with_derivative(
        self, initial: State, t: float, dt: float, d: State
    ) -> State:
        """Step `initial` along the derivative `d` by `dt`, then take the
        derivative there."""
        stepped = State(
            position=initial.position + d.position * dt,
            velocity=initial.velocity + d.velocity * dt,
        )
        return State(
            position=stepped.velocity.copy(),
            velocity=self.motion_function(stepped, t + dt),
        )

    def integrate(self, t: float, dt: float) -> None:
        """Advance the state by `dt` using the four RK4 evaluations."""
        a = self.evaluate(self.state, t)
        b = self.evaluate_with_derivative(self.state, t, dt * 0.5, a)
        c = self.evaluate_with_derivative(self.state, t, dt * 0.5, b)
        d = self.evaluate_with_derivative(self.state, t, dt, c)
        dxdt = (a.position + 2.0 * (b.position + c.position) + d.position) / 6.0
        dvdt = (a.velocity + 2.0 * (b.velocity + c.velocity) + d.velocity) / 6.0
        self.state.position = self.state.position + dxdt * dt
        self.state.velocity = self.state.velocity + dvdt * dt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_rk4.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/rk4.py MassSpring/tests/test_rk4.py MassSpring/tests/__init__.py
git commit -m "feat(mass-spring): add the RK4 integrator core"
```

---

### Task 2: The mass spring chain

**Files:**
- Create: `MassSpring/mass_spring.py`
- Test: `MassSpring/tests/test_mass_spring.py`

**Interfaces:**
- Consumes: `rk4.State`, `rk4.AbstractRK4Integrator` (Task 1).
- Produces: `MassSpringChain(AbstractRK4Integrator)` with
  - `__init__(self, start: Vec3 = Vec3(-1,0,0), end: Vec3 = Vec3(1,0,0), num_masses: int = 2, rest_length: float = 1.0, k: float = 8.0, damping: float = 0.5)`
  - properties/setters: `set_k(float)`, `set_damping(float)`, `set_rest_length(float)`, `set_timestep(float)`, `set_gravity(bool)`, `set_gravity_strength(float)`, `set_num_masses(int)`, `set_start(Vec3)`, `set_end(Vec3)`, `set_fix_first(bool)`, `set_fix_last(bool)`
  - `positions -> np.ndarray` (N,3), `initial_positions -> np.ndarray` (N,3), `num_masses -> int`, `is_fixed(i: int) -> bool`
  - `update() -> None`, `reset() -> None`, `total_energy() -> float`

- [ ] **Step 1: Write the failing test**

Create `MassSpring/tests/test_mass_spring.py`:

```python
"""Tests for the mass spring chain.

The chain is the part that had to be redesigned from the C++ (where each
spring owned and integrated its own two endpoints, which cannot work once a
mass is shared by two springs), so these check the physics behaves rather
than that the code merely runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mass_spring import MassSpringChain  # noqa: E402
from ncca.ngl import Vec3  # noqa: E402


def _settled(chain: MassSpringChain, steps: int = 400) -> None:
    for _ in range(steps):
        chain.update()


def test_defaults_to_two_masses_like_the_original_demo():
    chain = MassSpringChain()
    assert chain.num_masses == 2
    assert chain.positions.shape == (2, 3)


def test_reset_spaces_masses_evenly_between_start_and_end():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(3, 0, 0), num_masses=4)
    np.testing.assert_allclose(
        chain.positions,
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        atol=1e-6,
    )


def test_a_fixed_mass_never_moves():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(3, 0, 0), num_masses=3)
    chain.set_fix_first(True)
    chain.set_gravity(True)
    anchor = chain.positions[0].copy()
    _settled(chain)
    np.testing.assert_allclose(chain.positions[0], anchor, atol=1e-9)


def test_a_chain_at_rest_length_with_no_gravity_stays_put():
    # three masses 1.0 apart, springs resting at exactly 1.0 -> no force
    chain = MassSpringChain(
        start=Vec3(0, 0, 0), end=Vec3(2, 0, 0), num_masses=3, rest_length=1.0
    )
    chain.set_gravity(False)
    before = chain.positions.copy()
    _settled(chain)
    np.testing.assert_allclose(chain.positions, before, atol=1e-6)


def test_damping_reduces_energy():
    chain = MassSpringChain(
        start=Vec3(0, 0, 0), end=Vec3(3, 0, 0), num_masses=2, rest_length=1.0
    )
    chain.set_gravity(False)
    chain.set_damping(0.8)
    start_energy = chain.total_energy()
    _settled(chain, 200)
    assert chain.total_energy() < start_energy


def test_a_stretched_spring_pulls_its_free_end_back():
    # two masses 3 apart but resting at 1.0: the free end must move toward the
    # fixed one, i.e. its x decreases
    chain = MassSpringChain(
        start=Vec3(0, 0, 0), end=Vec3(3, 0, 0), num_masses=2, rest_length=1.0
    )
    chain.set_gravity(False)
    chain.set_fix_first(True)
    for _ in range(5):
        chain.update()
    assert chain.positions[1][0] < 3.0


def test_a_top_pinned_chain_sags_under_gravity():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(2, 0, 0), num_masses=3)
    chain.set_fix_first(True)
    chain.set_gravity(True)
    _settled(chain, 50)
    assert chain.positions[-1][1] < 0.0


def test_an_unpinned_chain_falls_under_gravity():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(1, 0, 0), num_masses=3)
    chain.set_gravity(True)
    _settled(chain, 20)
    # nothing is pinned, so the whole chain is in free fall
    assert np.all(chain.positions[:, 1] < 0.0)


def test_gravity_off_means_no_fall():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(1, 0, 0), num_masses=2)
    chain.set_gravity(False)
    _settled(chain, 50)
    assert chain.positions[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_coincident_masses_do_not_produce_nan():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(0, 0, 0), num_masses=3)
    _settled(chain, 10)
    assert np.all(np.isfinite(chain.positions))


def test_changing_the_mass_count_rebuilds_and_resets():
    chain = MassSpringChain(start=Vec3(0, 0, 0), end=Vec3(4, 0, 0), num_masses=2)
    chain.set_num_masses(5)
    assert chain.num_masses == 5
    assert chain.positions.shape == (5, 3)
    np.testing.assert_allclose(chain.positions[-1], [4, 0, 0], atol=1e-6)


def test_rejects_fewer_than_two_masses():
    chain = MassSpringChain()
    with pytest.raises(ValueError):
        chain.set_num_masses(1)


def test_rejects_a_non_positive_timestep():
    chain = MassSpringChain()
    with pytest.raises(ValueError):
        chain.set_timestep(0.0)


def test_pinning_a_moving_mass_stops_it_dead():
    chain = MassSpringChain(
        start=Vec3(0, 0, 0), end=Vec3(3, 0, 0), num_masses=2, rest_length=1.0
    )
    chain.set_gravity(False)
    for _ in range(5):
        chain.update()
    chain.set_fix_last(True)
    _settled(chain, 50)
    # it snaps back to its anchor and stays there rather than drifting on
    # carrying the velocity it had when it was pinned
    np.testing.assert_allclose(
        chain.positions[1], chain.initial_positions[1], atol=1e-9
    )
    np.testing.assert_allclose(chain.state.velocity[1], [0.0, 0.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_mass_spring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mass_spring'`

- [ ] **Step 3: Write minimal implementation**

Create `MassSpring/mass_spring.py`:

```python
"""A chain of masses joined by damped springs, integrated with RK4.

The C++ demo this comes from gives each spring its own two endpoints and
integrates the displacement between them. That works for one spring but
falls apart for a chain: a mass in the middle belongs to two springs, and
each would integrate its own copy of it. So here the state belongs to the
masses and the springs are only force producers -- every spring touching a
mass adds into the same acceleration slot.

Every mass weighs 1.0, so force and acceleration are the same number.
"""

import numpy as np
from ncca.ngl import Vec3
from rk4 import AbstractRK4Integrator, State

# Below this separation the spring direction x/|x| is meaningless, so two
# coincident masses simply push on each other with nothing.
_MIN_SEPARATION = 1e-8


class MassSpringChain(AbstractRK4Integrator):
    """N masses in a line joined by N-1 damped springs.

    At num_masses=2 this is exactly the single spring of the original demo.

    Attributes
    ----------
        num_masses : int
            how many masses the chain has (>= 2)
    """

    def __init__(
        self,
        start: Vec3 = Vec3(-1.0, 0.0, 0.0),
        end: Vec3 = Vec3(1.0, 0.0, 0.0),
        num_masses: int = 2,
        rest_length: float = 1.0,
        k: float = 8.0,
        damping: float = 0.5,
    ) -> None:
        self._start = start
        self._end = end
        self._num_masses = num_masses
        self._rest_length = rest_length
        self._k = k
        self._damping = damping
        self._timestep = 0.1
        self._gravity = False
        self._gravity_strength = 9.81
        self._fix_first = False
        self._fix_last = False
        self._t = 0.0
        self.reset()

    # ------------------------------------------------------------- accessors
    @property
    def num_masses(self) -> int:
        return self._num_masses

    @property
    def positions(self) -> np.ndarray:
        return self.state.position

    @property
    def initial_positions(self) -> np.ndarray:
        return self._initial_positions

    def is_fixed(self, index: int) -> bool:
        """Is the mass at `index` pinned?"""
        return bool(self._fixed_mask[index])

    # -------------------------------------------------------------- mutators
    def set_k(self, k: float) -> None:
        self._k = k

    def set_damping(self, damping: float) -> None:
        self._damping = damping

    def set_rest_length(self, length: float) -> None:
        self._rest_length = length

    def set_timestep(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"timestep must be positive, got {dt}")
        self._timestep = dt

    def set_gravity(self, enabled: bool) -> None:
        self._gravity = enabled

    def set_gravity_strength(self, strength: float) -> None:
        self._gravity_strength = strength

    def set_num_masses(self, count: int) -> None:
        """Rebuild the chain with `count` masses. This resets the sim -- the
        state arrays change shape, so there is nothing sensible to carry over."""
        if count < 2:
            raise ValueError(f"a chain needs at least 2 masses, got {count}")
        self._num_masses = count
        self.reset()

    def set_start(self, start: Vec3) -> None:
        self._start = start
        self.reset()

    def set_end(self, end: Vec3) -> None:
        self._end = end
        self.reset()

    def set_fix_first(self, fixed: bool) -> None:
        self._fix_first = fixed
        self._rebuild_fixed_mask()

    def set_fix_last(self, fixed: bool) -> None:
        self._fix_last = fixed
        self._rebuild_fixed_mask()

    # ----------------------------------------------------------------- sim
    def reset(self) -> None:
        """Space the masses evenly between start and end and stop them dead."""
        a = np.array(self._start.to_list(), dtype=np.float64)
        b = np.array(self._end.to_list(), dtype=np.float64)
        blend = np.linspace(0.0, 1.0, self._num_masses)[:, None]
        positions = a + (b - a) * blend
        self.state = State(
            position=positions,
            velocity=np.zeros_like(positions),
        )
        self._initial_positions = positions.copy()
        self._t = 0.0
        self._rebuild_fixed_mask()

    def update(self) -> None:
        """Advance the sim one timestep."""
        self.integrate(self._t, self._timestep)
        self._t += self._timestep
        self._apply_fixed()

    def total_energy(self) -> float:
        """Kinetic + spring potential, used by the tests to show damping bleeds
        energy out of the system."""
        kinetic = 0.5 * float(np.sum(self.state.velocity**2))
        d = np.diff(self.state.position, axis=0)
        extension = np.linalg.norm(d, axis=1) - self._rest_length
        potential = 0.5 * self._k * float(np.sum(extension**2))
        return kinetic + potential

    # ------------------------------------------------------------- internals
    def _rebuild_fixed_mask(self) -> None:
        mask = np.zeros(self._num_masses, dtype=bool)
        mask[0] = self._fix_first
        mask[-1] = self._fix_last
        self._fixed_mask = mask

    def _apply_fixed(self) -> None:
        """Pin the fixed masses back to their anchors and stop them.

        The accelerations are already zeroed for these, but a mass pinned
        while it was moving still carries velocity, and the position half of
        the integration would drift it away forever.
        """
        self.state.position[self._fixed_mask] = self._initial_positions[
            self._fixed_mask
        ]
        self.state.velocity[self._fixed_mask] = 0.0

    def motion_function(self, state: State, t: float) -> np.ndarray:
        """F = -k(|x|-d)(x/|x|) - bv per spring, accumulated onto both ends.

        The spring force law is the original demo's. Unlike the C++, this
        reads the spring length out of the state being evaluated rather than
        the object's own members, so the RK4 sub-steps actually see the
        positions they are asking about.
        """
        accel = np.zeros_like(state.position)
        if self._gravity:
            accel[:, 1] -= self._gravity_strength

        # displacement a->b for every spring, where a is mass i and b is i+1
        displacement = state.position[1:] - state.position[:-1]
        length = np.linalg.norm(displacement, axis=1)
        direction = np.zeros_like(displacement)
        real = length > _MIN_SEPARATION
        direction[real] = displacement[real] / length[real, None]

        relative_velocity = state.velocity[1:] - state.velocity[:-1]
        force = (
            -self._k * (length - self._rest_length)[:, None] * direction
            - self._damping * relative_velocity
        )
        # force is the force on the b end; a gets the equal and opposite. The
        # two slices never overlap, so a plain += is enough to accumulate.
        accel[1:] += force
        accel[:-1] -= force

        accel[self._fixed_mask] = 0.0
        return accel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/ -q`
Expected: PASS, 17 passed (3 from Task 1 + 14 here)

- [ ] **Step 5: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/mass_spring.py MassSpring/tests/test_mass_spring.py
git commit -m "feat(mass-spring): add the mass spring chain"
```

---

### Task 3: The OpenGL scene

**Files:**
- Create: `MassSpring/MassSpringScene.py`

**Interfaces:**
- Consumes: `MassSpringChain` (Task 2) — uses `.positions`, `.initial_positions`, `.num_masses`, `.is_fixed(i)`, `.update()`.
- Produces: `MassSpringScene(PySideEventHandlingMixin, QOpenGLWidget)` with
  - `__init__(self, chain: MassSpringChain, timer_interval: int = 20, parent=None)`
  - `start_sim_timer() -> None`, `stop_sim_timer() -> None`
  - Slots: `set_timer_duration(int)`, `toggle_sim(bool)`

There is no unit test for this task — it is GL drawing, verified by `--smoketest` in Task 4 (repo convention: rendering is smoke-tested, maths is unit-tested).

- [ ] **Step 1: Write the scene**

Create `MassSpring/MassSpringScene.py`:

```python
"""The OpenGL view of the mass spring chain.

Draws what the C++ NGLScene drew, generalised to N masses: a line through
the chain, a cube at each mass (red when pinned, green when free) and a
ghost sphere at each mass's start position so you can see how far the chain
has moved.
"""

import numpy as np
import OpenGL.GL as gl
from mass_spring import MassSpringChain
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget

_MASS_SCALE = 0.1
_FIXED_COLOUR = (1.0, 0.0, 0.0, 1.0)
_FREE_COLOUR = (0.0, 1.0, 0.0, 1.0)
_GHOST_COLOUR = (0.4, 0.4, 0.8, 1.0)
_LINE_COLOUR = (1.0, 1.0, 1.0, 1.0)


class MassSpringScene(PySideEventHandlingMixin, QOpenGLWidget):
    """Draws the chain and drives the sim timer."""

    def __init__(
        self, chain: MassSpringChain, timer_interval: int = 20, parent=None
    ) -> None:
        super().__init__(parent)
        self.chain = chain
        self.window_width = 1024
        self.window_height = 720
        self.transform = Transform()
        self._timer_interval = timer_interval
        self._timer_id = None
        self.setup_event_handling()
        self.start_sim_timer()

    # ------------------------------------------------------------ sim timer
    def start_sim_timer(self) -> None:
        if self._timer_id is None:
            self._timer_id = self.startTimer(self._timer_interval)

    def stop_sim_timer(self) -> None:
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

    @Slot(int)
    def set_timer_duration(self, interval: int) -> None:
        """Restart the timer at a new interval, if it is running."""
        self._timer_interval = interval
        if self._timer_id is not None:
            self.stop_sim_timer()
            self.start_sim_timer()

    @Slot(bool)
    def toggle_sim(self, running: bool) -> None:
        if running:
            self.start_sim_timer()
        else:
            self.stop_sim_timer()

    def timerEvent(self, event) -> None:
        self.chain.update()
        self.update()

    # ---------------------------------------------------------------- setup
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / max(self.height(), 1), 0.5, 150.0
        )
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        # cube and teapot are mesh defaults -- Prims.CUBE is not a create()
        # type and passing it to create raises ValueError.
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, w / max(h, 1), 0.5, 150.0)

    # -------------------------------------------------------------- drawing
    def _mouse_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        return rot_y @ rot_x

    def load_matrices_to_shader(self, global_tx: Mat4) -> None:
        """The ngl diffuse shader lights in VIEW space (it takes MV and a
        normalMatrix built from MV), so unlike the world-space PBR demos the
        normal matrix here correctly comes from MV."""
        ShaderLib.use(DefaultShader.DIFFUSE)
        M = global_tx @ self.transform.matrix()
        MV = self.view @ M
        MVP = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def _draw_chain_line(self, global_tx: Mat4) -> None:
        """One GL_LINE_STRIP through every mass, rebuilt each frame because the
        positions change every step."""
        points = self.chain.positions.astype(np.float32).reshape(-1)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", *_LINE_COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINE_STRIP)
        with vao as v:
            v.set_data(VertexData(points, self.chain.num_masses))
            v.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            v.set_num_indices(self.chain.num_masses)
            v.draw()

    def _draw_mass(
        self, position: np.ndarray, colour, prim: str, global_tx: Mat4
    ) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", *colour)
        self.transform.reset()
        self.transform.set_scale(_MASS_SCALE, _MASS_SCALE, _MASS_SCALE)
        self.transform.set_position(
            float(position[0]), float(position[1]), float(position[2])
        )
        self.load_matrices_to_shader(global_tx)
        Primitives.draw(prim)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_tx = self._mouse_global_tx()

        self._draw_chain_line(global_tx)
        for i in range(self.chain.num_masses):
            colour = _FIXED_COLOUR if self.chain.is_fixed(i) else _FREE_COLOUR
            self._draw_mass(self.chain.positions[i], colour, "cube", global_tx)
        # ghosts of where each mass started, as the original drew its targets
        for i in range(self.chain.num_masses):
            self._draw_mass(
                self.chain.initial_positions[i], _GHOST_COLOUR, "sphere", global_tx
            )
```

- [ ] **Step 2: Check it imports cleanly**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring && QT_QPA_PLATFORM=offscreen uv run python -c "import MassSpringScene; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/MassSpringScene.py
git commit -m "feat(mass-spring): add the OpenGL scene"
```

---

### Task 4: The GUI

**Files:**
- Create: `MassSpring/main.py` (then `chmod +x`)

**Interfaces:**
- Consumes: `MassSpringChain` (Task 2), `MassSpringScene` (Task 3), `ncca.ngl.widgets.Vec3Widget`.
- Produces: `MainWindow(QMainWindow)`; `main()` entry point.

`Vec3Widget` API (verified): `set_name(str)`, `set_value(Vec3)`, `get_value() -> Vec3`, `set_range(min, max)`, signal `valueChanged(Vec3)`.

- [ ] **Step 1: Write the GUI**

Create `MassSpring/main.py`:

```python
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
        self.chain = MassSpringChain(
            start=Vec3(-1.0, 1.0, 0.0), end=Vec3(1.0, 1.0, 0.0), num_masses=2
        )
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
        self.start_widget = Vec3Widget(value=Vec3(-1.0, 1.0, 0.0))
        self.start_widget.set_name("Start (A)")
        self.start_widget.set_range(-5.0, 5.0)
        self.end_widget = Vec3Widget(value=Vec3(1.0, 1.0, 0.0))
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
        self.k = self._spinbox(8.0, 0.0, 100.0, 0.1)
        self.damping = self._spinbox(0.5, 0.0, 10.0, 0.01)
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
        self.dt = self._spinbox(0.1, 0.001, 1.0, 0.001)
        self.dt.setDecimals(3)
        self.timer_value = QSpinBox()
        self.timer_value.setRange(1, 200)
        self.timer_value.setValue(20)
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
```

- [ ] **Step 2: Make it executable and smoke-test it**

Run:
```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring
chmod +x main.py
uv run main.py --smoketest 600
```
Expected: `SMOKETEST OK` and exit 0, with no traceback. If a Qt/GL error appears, re-run with `--debug` for the traceback.

- [ ] **Step 3: Check the chain actually moves (not a still image)**

Run:
```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring
uv run python -c "
from mass_spring import MassSpringChain
from ncca.ngl import Vec3
c = MassSpringChain(start=Vec3(0,1,0), end=Vec3(2,1,0), num_masses=5)
c.set_fix_first(True); c.set_gravity(True)
before = c.positions[-1].copy()
for _ in range(100): c.update()
print('start', before, '-> after', c.positions[-1])
assert c.positions[-1][1] < before[1], 'chain did not sag'
print('CHAIN SAGS OK')
"
```
Expected: `CHAIN SAGS OK`

- [ ] **Step 4: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/main.py
git commit -m "feat(mass-spring): add the GUI built from the PyNGL widgets"
```

---

### Task 5: Screenshot, README and root listing

**Files:**
- Create: `MassSpring/README.md`
- Create: `MassSpring/MassSpring.png`
- Modify: `README.md` (root — add the demo to the listing)

**Interfaces:**
- Consumes: the finished demo from Tasks 1-4.
- Produces: nothing code depends on.

- [ ] **Step 1: Capture a screenshot of the running demo**

The repo convention is a preview image per demo folder, shown by `RunDemos.py`. Capture it with a script rather than by hand, so it is reproducible — `QOpenGLWidget.grabFramebuffer()` reads the GL scene straight out of the running app.

Write `/tmp/shot.py`:

```python
"""Throwaway: run the demo, settle the chain, save the preview image."""

import sys

from main import MainWindow
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
fmt = QSurfaceFormat()
fmt.setSamples(4)
fmt.setMajorVersion(4)
fmt.setMinorVersion(1)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)

win = MainWindow()
win.masses.setValue(8)
win.gravity.setChecked(True)
win.show()


def grab():
    # let the chain swing out into a shape worth looking at
    for _ in range(60):
        win.chain.update()
    win.scene.repaint()
    image = win.scene.grabFramebuffer()
    image.save("MassSpring.png")
    print("saved", image.width(), image.height())
    app.quit()


QTimer.singleShot(800, grab)
sys.exit(app.exec())
```

Run it from the demo folder (it needs a real display — do NOT set `QT_QPA_PLATFORM=offscreen`):

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring
uv run python /tmp/shot.py
```
Expected: `saved <width> <height>` with both over 400, and `MassSpring.png` written in the demo folder.

Then look at the image before committing it — `Read` the file. It must show the chain hanging and sagging, not an empty grey frame. If it is blank, the grab happened before the first paint: raise the `singleShot` delay and try again.

- [ ] **Step 2: Write the README**

Create `MassSpring/README.md` (jon-writing-style: first person, British English, no marketing adjectives, links inline):

```markdown
# Mass Spring Chain (RK4)

![](MassSpring.png)

A PyNGL / PySide6 port of the C++ [MassSpring](https://github.com/NCCA/NGL9Demos)
demo I use in the lectures, using RK4 integration as described on
[gafferongames](http://gafferongames.com/game-physics/). The spring force is
the usual `F = -k(|x|-d)(x/|x|) - bv`.

The one change from the C++ is that the single spring has become a chain: set
Masses to 2 and it is the original demo, wind it up and you get a rope. Start
and End place the two ends and the masses in between are spaced evenly along
it, so the chain is a strict generalisation rather than a different demo.

To run it:

```bash
cd MassSpring
uv run main.py
```

Left mouse rotates; everything else is on the panel. Pin either end, turn on
gravity and watch it swing. The cubes are the masses (red when pinned, green
when free) and the ghost spheres show where each mass started.

## How it is put together

The C++ gives each spring its own two endpoints and integrates the
displacement between them. That works for one spring but not for a chain -- a
mass in the middle belongs to two springs and each would integrate its own
copy of it. So here the state belongs to the masses and the springs only
produce forces, which every spring touching a mass adds into the same
acceleration slot.

- `rk4.py` -- the integrator, straight from the C++ `AbstractRK4Integrator`,
  over `(N,3)` arrays rather than a single `Vec3`. No Qt, no OpenGL.
- `mass_spring.py` -- the chain and the spring force law.
- `MassSpringScene.py` -- the drawing.
- `main.py` -- the GUI, using `Vec3Widget` from `ncca.ngl.widgets` for the
  two ends.

Because the physics has no Qt or GL in it, the tests are headless and check
the maths rather than the pixels -- the integrator goes against the analytic
simple harmonic oscillator:

```bash
uv run pytest MassSpring/tests/
```
```

- [ ] **Step 3: Add the demo to the root README**

The root README lists demos as a three-column table per section. `MassSpring` belongs under `## Animation` (line ~98), which currently holds only `SkeletalAnimation`. Add this row directly after the `SkeletalAnimation` row, keeping the table's existing style exactly:

```markdown
| <a href="MassSpring"><img src="MassSpring/MassSpring.png" width="220"></a> | [MassSpring](MassSpring) | Damped mass spring chain with RK4 integration, from a single spring up to a rope |
```

Do not touch the `## Contents` list at the top — `Animation` is already an entry there, and this adds no new section.

- [ ] **Step 4: Verify the whole thing one last time**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run pytest MassSpring/tests/ -q
uv run ruff check MassSpring/ && uv run ruff format --check MassSpring/
cd MassSpring && uv run main.py --smoketest 600
```
Expected: 17 passed; ruff clean; `SMOKETEST OK`.

- [ ] **Step 5: Commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
git add MassSpring/README.md MassSpring/MassSpring.png README.md
git commit -m "docs(mass-spring): add the README, preview image and root listing"
```
