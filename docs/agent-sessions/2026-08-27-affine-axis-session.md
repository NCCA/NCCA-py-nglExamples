# 2026-08-27 -- AffineTransforms axis gizmo pointed the wrong way

## Goal

The RGB axis in `AffineTransforms` was too big and the cylinders were oriented
wrongly — three fat cones and no visible shafts, sitting on top of the teapot
the demo exists to show.

## What was wrong

Two separate things, both from `axis.py` being a straight port of the C++
`NGL9Demos/AffineTransforms/Axis`.

**Orientation.** The C++ version's cylinder runs along z from the origin, so
the port rotated each shaft as if it started life pointing down +z. PyNGL's
`PrimData.cylinder` is aligned down **y** and centred on the origin, so those
rotations sent every shaft somewhere else: the X shaft was rotated about y,
which leaves a y-aligned cylinder exactly where it was, and the Y and Z shafts
swapped places. The cones came out right by luck — `PrimData.cone` really does
run from the origin along +z, matching the C++ layout.

**Size.** `draw_axis()` drew `Primitives.draw("cylinder")` and `"cone"`, i.e.
whatever the host demo had registered under those names. `main.py` registers
them at radius 0.5 for the object being transformed, so the axis inherited a
radius of half its own length. `main.py` had already worked around the symptom
by passing `scale=0.35` with a comment about the gizmo swallowing the teapot.

## The fix

`draw_axis()` now registers its own `_axis_shaft` (unit cylinder) and
`_axis_head` (unit cone) on first draw, and derives the shaft radius, head
radius and head length from `scale`, which is now documented as the half-length
of each axis. The proportions therefore hold at any size and don't depend on
what the demo has registered. `main.py` passes `scale=1.5`.

Before trusting the rotations I checked them numerically rather than by eye —
`Mat4` is row-vector, so `v @ tx.matrix()` is the transform a vertex sees:

```
shaft X [1. 0. 0.]   head +X [1. 0. 0.]   head -X [-1.  0.  0.]
shaft Y [0. 1. 0.]   head +Y [0. 1. 0.]   head -Y [ 0. -1.  0.]
shaft Z [0. 0. 1.]   head +Z [0. 0. 1.]   head -Z [ 0.  0. -1.]
```

## Files changed

- `AffineTransforms/axis.py` — rewritten shaft/head placement, own primitives
- `AffineTransforms/main.py` — `scale=1.5`, stale workaround comment removed
- `AffineTransforms/README.md` — the Notes bullet on `axis.py` was describing
  the old borrowed-primitives behaviour
- `AffineTransforms/AffineTransforms.png` — regenerated; the old one shows the
  broken gizmo

## Commands run

```bash
git worktree add .worktrees/affine-axis-fix -b agent/affine-axis-fix
uv run ruff format AffineTransforms/ && uv run ruff check AffineTransforms/
uv run pytest -q                      # 834 passed
cd AffineTransforms && uv run main.py --smoketest 800
```

The screenshot is composed in-process — `MainWindow.grab()` for the panel with
`scene.grabFramebuffer()` painted over the `QWindowContainer` — rather than
`screencapture`, which grabs the desktop region and can catch whatever is in
front of the window.
