"""Headless tests for the transform-hierarchy maths in scene_graph.py.

These pin down the one thing this demo is teaching: how a node's world
matrix is built from its ancestors. Get the ``@`` order backwards and every
number below comes out wrong, so each test asserts an actual numeric
position rather than "it ran without crashing".
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Mat4
from scene_graph import Node


def _position(world: Mat4) -> np.ndarray:
    """World position of a node's local origin: row 3 of its world matrix.

    Transforming the origin (0, 0, 0, 1) by an affine matrix in the
    row-vector convention picks out exactly row 3, so this is a shortcut
    for ``(0, 0, 0, 1) @ world`` without needing a Vec4.
    """
    return np.array([world[3, 0], world[3, 1], world[3, 2]])


class TestTwoLevelTranslation:
    def test_translations_add_along_the_chain(self):
        """base at x=1, child local x=2 -> child world x=3."""
        base = Node("base", local=Mat4.translate(1.0, 0.0, 0.0))
        child = base.add(Node("child", local=Mat4.translate(2.0, 0.0, 0.0)))

        world = child.world_matrix(base.world_matrix())

        np.testing.assert_allclose(_position(world), [3.0, 0.0, 0.0], atol=1e-5)


class TestRotationSwingsChild:
    def test_parent_rotation_swings_child_offset(self):
        """base yaws 90 deg about y; a child offset (0,0,2) swings to (2,0,0)."""
        base = Node("base", local=Mat4.rotate_y(90.0))
        child = base.add(Node("child", local=Mat4.translate(0.0, 0.0, 2.0)))

        world = child.world_matrix(base.world_matrix())

        np.testing.assert_allclose(_position(world), [2.0, 0.0, 0.0], atol=1e-4)


class TestWalkOrderAndPropagation:
    def test_depth_first_order(self):
        """walk() visits parent before children, children before siblings."""
        root = Node("root")
        a = root.add(Node("a"))
        a.add(Node("b"))
        root.add(Node("c"))

        names = [node.name for node, _ in root.walk()]

        assert names == ["root", "a", "b", "c"]

    def test_grandchild_world_composes_the_whole_chain(self):
        """A grandchild's world matrix must fold in *every* ancestor, not
        just its immediate parent."""
        root = Node("root", local=Mat4.translate(1.0, 0.0, 0.0))
        a = root.add(Node("a", local=Mat4.translate(0.0, 1.0, 0.0)))
        a.add(Node("b", local=Mat4.translate(0.0, 0.0, 1.0)))

        worlds = {node.name: world for node, world in root.walk()}

        np.testing.assert_allclose(_position(worlds["b"]), [1.0, 1.0, 1.0], atol=1e-5)

    def test_walk_default_parent_is_identity(self):
        """Calling walk() with no argument starts from an identity world,
        so a root node's own local matrix is its world matrix."""
        root = Node("root", local=Mat4.translate(5.0, 0.0, 0.0))

        _, world = next(root.walk())

        np.testing.assert_allclose(_position(world), [5.0, 0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
