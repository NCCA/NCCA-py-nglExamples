"""Headless tests for the Reynolds flocking maths that BoidsCompute.wgsl
transcribes onto the GPU. See boid_maths.py for the rule implementations
this exercises."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from boid_maths import BoidParams, step  # noqa: E402

# A set of radii/weights that switches off every rule bar the one under test,
# so each test isolates a single force.
BASE = BoidParams(
    rs=0.0,
    ra=0.0,
    rc=0.0,
    w_sep=0.0,
    w_align=0.0,
    w_coh=0.0,
    w_wall=0.0,
    v_min=0.0,
    v_max=100.0,
    dt=0.1,
    bounds=(100.0, 100.0, 100.0),
)


class TestLoneBoid:
    def test_keeps_velocity_when_no_neighbours_and_no_walls(self):
        """A single boid has nobody to flock with, so every rule term is
        zero and its velocity should be untouched (only position drifts)."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.5, -0.25]])
        params = replace(
            BASE, rs=1.0, ra=1.0, rc=1.0, w_sep=1.0, w_align=1.0, w_coh=1.0
        )

        new_positions, new_velocities = step(positions, velocities, params)

        np.testing.assert_allclose(new_velocities, velocities, atol=1e-6)
        np.testing.assert_allclose(new_positions, positions + velocities * params.dt)


class TestSeparation:
    def test_two_converging_boids_separate(self):
        """Two boids heading straight at each other, well inside the
        separation radius, should have their velocities steered apart by
        the time step() returns - a strong enough push even reverses their
        closing motion."""
        positions = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        params = replace(BASE, rs=1.0, w_sep=50.0, dt=0.1)

        _, new_velocities = step(positions, velocities, params)

        # boid 0 (left) is pushed further left (away from boid 1); boid 1 is
        # pushed further right (away from boid 0) - they now diverge.
        assert new_velocities[0, 0] < 0.0
        assert new_velocities[1, 0] > 0.0

    def test_symmetric_pair_produces_mirror_image_forces(self):
        """Positions and velocities mirrored in x must produce mirrored
        accelerations - the maths has no hidden left/right bias."""
        positions = np.array([[-1.0, 0.3, 0.0], [1.0, 0.3, 0.0]])
        velocities = np.array([[0.4, 0.0, 0.2], [-0.4, 0.0, 0.2]])
        params = replace(BASE, rs=3.0, w_sep=2.0, dt=0.1)

        _, new_velocities = step(positions, velocities, params)

        mirror = new_velocities[0] * np.array([-1.0, 1.0, 1.0])
        np.testing.assert_allclose(mirror, new_velocities[1], atol=1e-6)


class TestAlignment:
    def test_matches_hand_computed_two_boid_case(self):
        """dt=1, only alignment switched on: each boid should simply swap
        headings with the other (steer = neighbour's velocity - mine)."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        params = replace(BASE, ra=2.0, w_align=1.0, dt=1.0)

        _, new_velocities = step(positions, velocities, params)

        np.testing.assert_allclose(new_velocities[0], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(new_velocities[1], [1.0, 0.0, 0.0], atol=1e-6)


class TestSpeedClamp:
    def test_speed_is_clamped_to_v_max(self):
        """A lone, wall-free boid moving faster than v_max must be rescaled
        down to v_max while keeping its heading."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[5.0, 0.0, 0.0]])
        params = replace(BASE, v_min=0.0, v_max=2.0)

        _, new_velocities = step(positions, velocities, params)

        speed = np.linalg.norm(new_velocities[0])
        np.testing.assert_allclose(speed, 2.0, atol=1e-6)
        np.testing.assert_allclose(
            new_velocities[0] / speed, [1.0, 0.0, 0.0], atol=1e-6
        )

    def test_speed_is_clamped_to_v_min(self):
        """A lone, wall-free boid moving slower than v_min must be rescaled
        up to v_min while keeping its heading."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[0.05, 0.0, 0.0]])
        params = replace(BASE, v_min=0.5, v_max=2.0)

        _, new_velocities = step(positions, velocities, params)

        speed = np.linalg.norm(new_velocities[0])
        np.testing.assert_allclose(speed, 0.5, atol=1e-6)
