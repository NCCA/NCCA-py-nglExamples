# Agent session — 2026-07-09 — SpatialHash3D ESC quit fix

## Goal

When quitting the SpatialHash3D GUI app with ESC while the WebGPU widget had
focus, only the widget closed and the control-panel window stayed open. Both
should exit together.

## Root cause

`WebGPUScene3D.keyPressEvent` handled Escape with `self.close()`, which closes
just the embedded widget when it is hosted inside `WebGPU3DGui`'s splitter.

## Fix

`self.window().close()` instead — closes the top-level main window when
embedded; standalone behaviour is unchanged because the widget is its own
top-level window there.

## Files changed

- `WebGPUCompute/SpatialHash3D/WebGPU3D.py` — ESC handler closes `self.window()`

## Commands run

- Offscreen verification script (`QT_QPA_PLATFORM=offscreen`) instantiating
  `WebGPU3DGui`, sending ESC via `QTest.keyClick` to the WebGPU widget, and
  asserting the main window closed — failed before the fix, passes after.
- `uv run pytest -q` — 31 passed
- `uv run ruff check` / `ruff format --check` — clean

## Notes

Work done in the existing `.worktrees/spatialhash3d-spheres` worktree
(branch `agent/spatialhash3d-spheres`); the main checkout had uncommitted
user edits to the same files and was left untouched. Session run by Claude
Code (opencode export not applicable to this session).
