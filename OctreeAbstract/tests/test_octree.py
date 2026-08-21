import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from octree import BoundingBox, Octree, ParticleSystem, bounds_lines  # noqa: E402


def test_bounding_box_subdivides_from_low_to_high_octants():
    bounds = BoundingBox.from_scalar(-2.0, 2.0)

    children = bounds.subdivide()

    assert children[0] == BoundingBox(-2, 0, -2, 0, -2, 0)
    assert children[7] == BoundingBox(0, 2, 0, 2, 0, 2)


def test_sphere_touching_a_box_is_in_bounds():
    bounds = BoundingBox.from_scalar(-1.0, 1.0)

    assert bounds.intersects_sphere(np.array([2.0, 0.0, 0.0]), 1.0)
    assert not bounds.intersects_sphere(np.array([2.01, 0.0, 0.0]), 1.0)


def test_bounds_lines_contains_twelve_independent_edges():
    lines = bounds_lines(2.0)

    assert lines.shape == (24, 3)
    assert np.all(np.abs(lines) == 2.0)


def test_octree_rejects_non_positive_height():
    with pytest.raises(ValueError, match="height"):
        Octree(0, BoundingBox.from_scalar(-1.0, 1.0))


def test_boundary_sphere_is_inserted_into_overlapping_leaves():
    tree = Octree(2, BoundingBox.from_scalar(-1.0, 1.0))

    tree.insert(5, np.zeros(3, dtype=np.float32), 0.1)

    leaves = tree.leaf_indices_for_object(5)
    assert len(leaves) == 8


def test_candidate_pairs_are_unique_across_overlapping_leaves():
    tree = Octree(2, BoundingBox.from_scalar(-1.0, 1.0))
    tree.insert(0, np.zeros(3, dtype=np.float32), 0.2)
    tree.insert(1, np.array([0.1, 0.0, 0.0], dtype=np.float32), 0.2)

    assert tree.candidate_pairs() == {(0, 1)}


def test_particles_reflect_from_the_simulation_walls():
    system = ParticleSystem(
        positions=np.array([[0.9, 0.0, 0.0]], dtype=np.float32),
        velocities=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
        colours=np.ones((1, 3), dtype=np.float32),
        radii=np.array([0.2], dtype=np.float32),
        bounds=BoundingBox.from_scalar(-1.0, 1.0),
        tree_height=2,
    )

    system.step()

    assert system.positions[0, 0] == pytest.approx(0.8)
    assert system.velocities[0, 0] == pytest.approx(-0.2)


def test_overlapping_particles_are_steered_apart_without_changing_speed():
    system = ParticleSystem(
        positions=np.array([[-0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32),
        velocities=np.array([[0, 0.1, 0], [0, -0.1, 0]], dtype=np.float32),
        colours=np.ones((2, 3), dtype=np.float32),
        radii=np.array([0.2, 0.2], dtype=np.float32),
        bounds=BoundingBox.from_scalar(-1.0, 1.0),
        tree_height=2,
    )
    speeds = np.linalg.norm(system.velocities, axis=1).copy()

    system.resolve_particle_collisions()

    assert system.velocities[0, 0] < 0.0
    assert system.velocities[1, 0] > 0.0
    assert np.allclose(np.linalg.norm(system.velocities, axis=1), speeds)


def test_seeded_grid_factory_is_repeatable():
    first = ParticleSystem.grid(size=3, seed=12)
    second = ParticleSystem.grid(size=3, seed=12)

    assert first.count == 27
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.velocities, second.velocities)
    assert np.array_equal(first.colours, second.colours)


def test_opengl_entry_point_imports():
    path = Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("octree_opengl_main", path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.bounds_lines)
