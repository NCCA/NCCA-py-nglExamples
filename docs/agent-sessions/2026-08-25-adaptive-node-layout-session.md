# 2026-08-25 -- Adaptive MathNodeEditor layout

## Goal

Fix the four remaining Windows CI failures in `MathNodeEditor`, which were all
one root cause: sizes measured from font metrics laid out against positions
fixed in pixels.

## What was actually wrong

Node widths come from `QFontMetrics(...).horizontalAdvance(title)` in
`graphics_items.py`, whilst `examples/*.json` pins every node to an absolute
x/y. The palette was `setFixedWidth(240)` whilst its buttons size themselves to
their labels. Windows renders the default UI font wider than macOS, so the
columns ran into each other and the longer labels clipped.

Measuring first was worth the detour. The layouts had no margin at all: on
macOS, `normal_matrix_demo` and `triangle_normal_demo` already sat at a **0px**
gap, `mesh_pipeline_demo` and `mvp_mesh_demo` at 4px. The old test passed only
because it tolerated an overlap of up to a pixel. This was never a Windows bug
-- Windows was just the first font wide enough to eat the last pixel.

Vertical gaps were fine (32px at worst, mostly over 100), so only x needed
resolving. That kept the change much smaller than a full relayout.

## Files changed

- `MathNodeEditor/graphics_items.py` -- `COLUMN_GUTTER`, `SAME_COLUMN_TOLERANCE`,
  and `resolve_column_positions`
- `MathNodeEditor/canvas.py` -- `_resolve_column_positions`, called from
  `from_dict` once the nodes are named and before the wires are made
- `MathNodeEditor/palette.py` -- `_preferred_width`, chrome constants
- `MathNodeEditor/node_editor.py` -- scroll area follows the palette
- `MathNodeEditor/tests/test_node_editor.py` -- both tests rewritten, four unit
  tests for the resolver added

## How the resolver works

Sort by x; nodes within `SAME_COLUMN_TOLERANCE` (160) of each other are one
staggered column and keep their relative offsets, **unless** they overlap
vertically, in which case they are neighbours on the same row and get split.
Columns are then packed left to right from measured extents and never dragged
leftwards, so a roomy layout is returned untouched.

The 160 threshold comes from the authored data: within-column stagger runs
40-120, the nearest two columns ever sit is 200.

Measured against the painted `boundingRect`, not `self.width` -- the sockets
hang off both edges and it is those the eye reads as a collision. Getting this
wrong first time showed up as a uniform 32px gap where 48 was asked for.

## Making the tests fail on macOS

The point of the rewrite. Both tests now push the application font up 8pt
before building the window, so they fail here rather than only on the Windows
runner, and the overlap test asserts a real gutter instead of mere
non-overlap. The parametrize list is now derived from `examples/` -- the
hard-coded one had drifted and was missing four files, which is exactly why
`mesh_pipeline_demo` slipped through the first version of the fix.

Verified at three font sizes; minimum gap stays at 48 and the palette goes
240 -> 331 -> 600 with nothing clipped.

## Left alone

- The numeric spin boxes in value nodes use a fixed `NUMERIC_EDITOR_WIDTH = 92`
  and clip their digits at larger fonts -- same root cause, not in scope here.
- The resolver runs on every load, not just the examples, so a hand-placed
  graph saved with two nodes closer than the gutter reopens with them pushed
  apart. It only ever opens gaps, never closes them, but it does mean a saved
  layout is not always reproduced to the pixel.
- 20 `ruff check` findings (SIM102, UP040) exist across `MathNodeEditor` on
  main already; pre-commit only runs `--select I` plus the formatter, and both
  pass.

## Commands run

```bash
git worktree add .worktrees/adaptive-node-layout -b agent/adaptive-node-layout
QT_QPA_PLATFORM=offscreen uv run pytest --ignore=SkinnedMeshImport/tests/test_mesh_maths.py --ignore=PBR/HDRIBaker/tests/test_bake_ibl.py -q
uv run ruff check --select I MathNodeEditor/
uv run ruff format --check MathNodeEditor/
```

817 passed, up from 812 -- four resolver unit tests plus four newly covered
examples, less the two tests that were replaced.
