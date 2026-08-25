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
