"""A small perfect octree and particle simulation for the octree demos."""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


def bounds_lines(extent: float = 10.0) -> np.ndarray:
    """Returns the twelve edges of a centred cube as a line list."""
    corners = np.array(
        [
            [-extent, -extent, -extent],
            [extent, -extent, -extent],
            [extent, extent, -extent],
            [-extent, extent, -extent],
            [-extent, -extent, extent],
            [extent, -extent, extent],
            [extent, extent, extent],
            [-extent, extent, extent],
        ],
        dtype=np.float32,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return np.asarray([corners[index] for edge in edges for index in edge])


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned limits for an octree node."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @classmethod
    def from_scalar(cls, minimum: float, maximum: float) -> "BoundingBox":
        return cls(minimum, maximum, minimum, maximum, minimum, maximum)

    def subdivide(self) -> tuple["BoundingBox", ...]:
        """Returns the eight children in the same order as the C++ source."""
        mid_x = (self.min_x + self.max_x) * 0.5
        mid_y = (self.min_y + self.max_y) * 0.5
        mid_z = (self.min_z + self.max_z) * 0.5
        children = []
        for index in range(8):
            x = (mid_x, self.max_x) if index & 1 else (self.min_x, mid_x)
            y = (mid_y, self.max_y) if index & 2 else (self.min_y, mid_y)
            z = (mid_z, self.max_z) if index & 4 else (self.min_z, mid_z)
            children.append(BoundingBox(x[0], x[1], y[0], y[1], z[0], z[1]))
        return tuple(children)

    def intersects_sphere(self, position: np.ndarray, radius: float) -> bool:
        return not (
            position[0] - radius > self.max_x
            or position[0] + radius < self.min_x
            or position[1] - radius > self.max_y
            or position[1] + radius < self.min_y
            or position[2] - radius > self.max_z
            or position[2] + radius < self.min_z
        )


@dataclass
class _TreeNode:
    bounds: BoundingBox
    height: int
    children: tuple["_TreeNode", ...] = ()
    objects: list[int] = field(default_factory=list)


class Octree:
    """A perfect octree which stores a sphere in every leaf it overlaps."""

    def __init__(self, height: int, bounds: BoundingBox) -> None:
        if height <= 0:
            raise ValueError("octree height must be positive")
        self.root = self._create_node(height, bounds)
        self._leaves: list[_TreeNode] = []
        self._collect_leaves(self.root)
        self._object_leaves: dict[int, list[int]] = {}

    def _create_node(self, height: int, bounds: BoundingBox) -> _TreeNode:
        if height == 1:
            return _TreeNode(bounds, height)
        children = tuple(
            self._create_node(height - 1, child) for child in bounds.subdivide()
        )
        return _TreeNode(bounds, height, children)

    def _collect_leaves(self, node: _TreeNode) -> None:
        if node.height == 1:
            self._leaves.append(node)
            return
        for child in node.children:
            self._collect_leaves(child)

    def clear(self) -> None:
        for leaf in self._leaves:
            leaf.objects.clear()
        self._object_leaves.clear()

    def insert(self, object_index: int, position: np.ndarray, radius: float) -> None:
        memberships: list[int] = []
        for leaf_index, leaf in enumerate(self._leaves):
            if leaf.bounds.intersects_sphere(position, radius):
                leaf.objects.append(object_index)
                memberships.append(leaf_index)
        self._object_leaves[object_index] = memberships

    def leaf_indices_for_object(self, object_index: int) -> tuple[int, ...]:
        return tuple(self._object_leaves.get(object_index, ()))

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for leaf in self._leaves:
            pairs.update(combinations(sorted(set(leaf.objects)), 2))
        return pairs


class ParticleSystem:
    """CPU particle motion with octree broad-phase collision checks."""

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        colours: np.ndarray,
        radii: np.ndarray,
        bounds: BoundingBox,
        tree_height: int = 4,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float32)
        self.velocities = np.asarray(velocities, dtype=np.float32)
        self.colours = np.asarray(colours, dtype=np.float32)
        self.radii = np.asarray(radii, dtype=np.float32)
        count = len(self.positions)
        if not (
            self.positions.shape == (count, 3)
            and self.velocities.shape == (count, 3)
            and self.colours.shape == (count, 3)
            and self.radii.shape == (count,)
        ):
            raise ValueError("particle arrays have incompatible shapes")
        self.bounds = bounds
        self.tree = Octree(tree_height, bounds)

    @property
    def count(self) -> int:
        return len(self.positions)

    @classmethod
    def grid(cls, size: int = 10, seed: int | None = None) -> "ParticleSystem":
        if size <= 0:
            raise ValueError("grid size must be positive")
        rng = np.random.default_rng(seed)
        axis = np.linspace(-9.5, 9.5, size, dtype=np.float32)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(
            np.float32
        )
        velocities = rng.uniform(-1.0, 1.0, (len(positions), 3)).astype(np.float32)
        lengths = np.linalg.norm(velocities, axis=1)
        lengths[lengths == 0.0] = 1.0
        velocities *= (0.05 / lengths)[:, None]
        colours = rng.random((len(positions), 3), dtype=np.float32)
        radii = rng.uniform(0.05, 0.35, len(positions)).astype(np.float32)
        return cls(
            positions,
            velocities,
            colours,
            radii,
            BoundingBox.from_scalar(-10.0, 10.0),
        )

    def step(self) -> None:
        self.positions += self.velocities
        self._resolve_walls()
        self.resolve_particle_collisions()

    def _resolve_walls(self) -> None:
        minimums = (self.bounds.min_x, self.bounds.min_y, self.bounds.min_z)
        maximums = (self.bounds.max_x, self.bounds.max_y, self.bounds.max_z)
        for axis, (minimum, maximum) in enumerate(zip(minimums, maximums, strict=True)):
            low = self.positions[:, axis] - self.radii < minimum
            high = self.positions[:, axis] + self.radii > maximum
            self.positions[low, axis] = minimum + self.radii[low]
            self.positions[high, axis] = maximum - self.radii[high]
            self.velocities[low, axis] = np.abs(self.velocities[low, axis])
            self.velocities[high, axis] = -np.abs(self.velocities[high, axis])

    def _rebuild_tree(self) -> None:
        self.tree.clear()
        for index, (position, radius) in enumerate(
            zip(self.positions, self.radii, strict=True)
        ):
            self.tree.insert(index, position, float(radius))

    def resolve_particle_collisions(self) -> None:
        self._rebuild_tree()
        forces = np.zeros_like(self.velocities)
        for first, second in self.tree.candidate_pairs():
            difference = self.positions[first] - self.positions[second]
            distance = float(np.linalg.norm(difference))
            overlap = float(self.radii[first] + self.radii[second]) - distance
            if overlap <= 0.0 or distance <= 1.0e-8:
                continue
            normal = difference / distance
            forces[first] += normal * overlap
            forces[second] -= normal * overlap

        speeds = np.linalg.norm(self.velocities, axis=1)
        changed = np.linalg.norm(forces, axis=1) > 0.0
        steered = self.velocities[changed] + forces[changed]
        lengths = np.linalg.norm(steered, axis=1)
        valid = lengths > 1.0e-8
        steered[valid] *= (speeds[changed][valid] / lengths[valid])[:, None]
        self.velocities[changed] = steered
