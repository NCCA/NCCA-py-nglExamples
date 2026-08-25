# SceneGraph

![](SceneGraph.png)

A robot arm as a transform hierarchy, to bridge the gap between "one MVP matrix" demos and everything that comes after them. `scene_graph.py` is a numpy/ncca-maths-only `Node` class with no GL in it at all — `main.py` just walks the tree it builds and draws whatever comes out.

## The idea

Every joint is a `Node` holding a local matrix, a mesh name and a colour, nothing else. A node's world matrix is its parent's world matrix combined with its own local one:

```python
def world_matrix(self, parent_world=None):
    return (parent_world or Mat4()) @ self.local
```

Walk the tree depth-first, threading each node's world matrix down as the next `parent_world`, and you get exactly what `glPushMatrix`/`glPopMatrix` used to do in the fixed-function pipeline — except explicit, in Python, and unit-testable:

```python
for node, world in root.walk():
    draw(node.mesh, world)
```

Rotate the shoulder and the elbow, wrist and both claws swing with it, because they're all downstream of the shoulder's world matrix in the walk — nobody hand-propagates anything.

## The `@` order, and why it isn't obvious

This repo's maths is row-vector (`row_vec @ matrix`), and `Mat4.__matmul__` is written so `A @ B` applies `B` first, then `A` — same as `MVP = project @ view @ model` applying `model` first. So a child's world matrix is:

```python
world = parent_world @ local  # local first, then everything above it
```

and a single joint's own local matrix is built the same way: `Mat4.translate(*pivot) @ Mat4.rotate_z(angle)` rotates the node's local frame about its own origin first, then carries the whole thing out to the pivot point in the parent's space. Get this backwards — `local @ parent_world`, or rotate-then-translate the wrong way round — and the arm still "renders", it just swings from the wrong place, which is a much nastier bug to spot than a crash. That's why `tests/test_scene_graph.py` pins down actual numbers (a 90 degree parent rotation must swing a `(0,0,2)` child offset to `(2,0,0)`) rather than just checking the code runs.

Mesh size and offset (`MESH_TRANSFORM` in `main.py`) are deliberately *not* part of a node's local matrix — they're applied only when drawing, after `world`. A node's local matrix answers "where is this joint's pivot", nothing about how big the box hanging off it is. Fold scale into `local` instead and it also scales every child's translation, since `parent_world @ local` carries the scale down the chain — occasionally what you want (a shrinking hand shrinks its own fingers), never what you want by accident.

## Controls

| Key | Action |
| :-- | :-- |
| `1`..`5` | select a joint: base, upper arm, lower arm, left claw, right claw |
| `Left` / `Right` | rotate the selected joint +/-5 degrees |
| `P` | toggle a canned sine-driven waving animation |
| `Space` | reset to the resting pose |
| LMB / RMB / wheel | rotate / pan / zoom the camera, `Esc` quits |

The HUD shows which joint is selected and its current angle.

## Tests

```bash
uv run pytest SceneGraph/tests
```

`tests/test_scene_graph.py` checks the composition rule directly: two-level translation adds along the chain, a parent rotation swings a child's offset to the expected world position, `walk()` visits parent-before-children in depth-first order, and a grandchild's world matrix folds in every ancestor, not just its immediate parent.

## Reference

- [Scene graph (Wikipedia)](https://en.wikipedia.org/wiki/Scene_graph) — the general idea this demo is a minimal instance of.
- `RayPickingSelection/picking_maths.py` and `Blending/blend_scene.py` in this repo — the same "numpy-only maths module, GL-free, unit tested" pattern.
