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

from rk4 import AbstractRK4Integrator, State


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


def _run(sim: _Oscillator, dt: float, steps: int) -> float:
    """Integrate `steps` steps and return the time actually reached.

    The caller compares against the analytic solution at *this* time, not at
    the idealised one it was aiming for -- a step count rarely lands exactly
    on pi or 2pi, and being out by a fraction of a step is the test's
    rounding, not the integrator's error.
    """
    for step in range(steps):
        sim.integrate(step * dt, dt)
    return steps * dt


def test_matches_the_analytic_oscillator_over_a_full_period():
    sim = _oscillator()
    dt = 0.001
    t = _run(sim, dt, round(2.0 * np.pi / dt))
    # x(t) = cos(t), v(t) = -sin(t); after a full period it is back home
    assert sim.state.position[0, 0] == pytest.approx(np.cos(t), abs=1e-6)
    assert sim.state.velocity[0, 0] == pytest.approx(-np.sin(t), abs=1e-6)


def test_matches_the_analytic_oscillator_at_half_a_period():
    sim = _oscillator()
    dt = 0.001
    t = _run(sim, dt, round(np.pi / dt))
    assert sim.state.position[0, 0] == pytest.approx(np.cos(t), abs=1e-6)
    assert sim.state.velocity[0, 0] == pytest.approx(-np.sin(t), abs=1e-6)


def test_integrates_every_mass_independently():
    sim = _Oscillator()
    sim.state = State(
        position=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        velocity=np.zeros((2, 3)),
    )
    dt = 0.001
    steps = round(np.pi / dt)
    for step in range(steps):
        sim.integrate(step * dt, dt)
    np.testing.assert_allclose(sim.state.position[0], [-1.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(sim.state.position[1], [0.0, -2.0, 0.0], atol=1e-4)
