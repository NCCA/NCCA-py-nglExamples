"""A minimal transform hierarchy: the maths behind a scene graph.

This module is deliberately numpy/ncca-maths-only (no GL, no Qt) so the
composition rule can be tested headless, same pattern as
``RayPickingSelection/picking_maths.py`` and ``Blending/blend_scene.py``.

Every ``Node`` owns a *local* matrix -- its transform relative to its
parent -- and nothing else knows or cares about the rest of the tree. A
node's *world* matrix is simply its parent's world matrix combined with its
own local one:

    world = parent_world @ local

Matrices here follow the repo's row-vector convention (see the design
spec): a point transforms as ``row_vector @ matrix``, and ``A @ B`` applies
``B`` first, then ``A`` -- so ``local`` (this node's own placement) happens
before ``parent_world`` (everything above it in the tree) is applied. Walk
the whole graph and you get exactly what a matrix stack in a fixed-function
renderer used to do: push the parent's transform, multiply in the child's,
draw, pop.
"""

from collections.abc import Iterator

from ncca.ngl import Mat4


class Node:
    """One joint in a transform hierarchy.

    ``local`` is this node's transform relative to its parent -- typically
    built each frame from a joint angle, so the node itself stays dumb: it
    holds a matrix, not the logic that produced it. ``mesh`` names a
    ``Primitives`` shape to draw at this node (or None for a pure joint with
    no geometry of its own, e.g. a shoulder).
    """

    def __init__(
        self,
        name: str,
        local: Mat4 | None = None,
        mesh: str | None = None,
        colour: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.name = name
        self.local: Mat4 = local if local is not None else Mat4()
        self.mesh = mesh
        self.colour = colour
        self.children: list[Node] = []

    def add(self, child: "Node") -> "Node":
        """Attach child under this node and return it, so calls can chain
        (`arm = base.add(Node("arm")); hand = arm.add(Node("hand"))`)."""
        self.children.append(child)
        return child

    def world_matrix(self, parent_world: Mat4 | None = None) -> Mat4:
        """This node's world matrix given its parent's (identity if root)."""
        if parent_world is None:
            parent_world = Mat4()
        return parent_world @ self.local

    def walk(self, parent_world: Mat4 | None = None) -> Iterator[tuple["Node", Mat4]]:
        """Depth-first traversal yielding (node, world_matrix) pairs.

        This is the whole graph in one generator: visit self, then recurse
        into each child passing *this node's* world matrix down as their
        parent. A renderer just does ``for node, world in root.walk(): ...``
        and draws whatever mesh each node carries at its world matrix.
        """
        world = self.world_matrix(parent_world)
        yield self, world
        for child in self.children:
            yield from child.walk(world)
