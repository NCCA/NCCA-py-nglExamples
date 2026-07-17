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
        start: Vec3 = Vec3(0.0, 2.0, 0.0),
        end: Vec3 = Vec3(0.0, -2.0, 0.0),
        num_masses: int = 2,
        rest_length: float = 1.0,
        k: float = 5.0,
        damping: float = 2.0,
    ) -> None:
        # These defaults are the C++ demo's own (its MainWindow.ui): a vertical
        # spring stretched to 4 against a resting length of 1, pinned at the
        # top. At num_masses=2 this really is that demo.
        self._start = start
        self._end = end
        self._num_masses = num_masses
        self._rest_length = rest_length
        self._k = k
        self._damping = damping
        self._timestep = 0.01
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
