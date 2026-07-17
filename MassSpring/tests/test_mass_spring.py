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
