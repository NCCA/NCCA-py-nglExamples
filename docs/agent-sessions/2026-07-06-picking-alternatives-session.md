# Agent session — 2026-07-06 (picking alternatives)

## Goal

Implement alternatives to the colour-ID picking used by the selection demos,
as two new projects:

1. **`RayPickingSelection/`** (OpenGL) — object picking by CPU ray casting
   (unproject through `inverse(MVP)`, bounding-sphere broad phase, vectorised
   Möller–Trumbore narrow phase) and gizmo-handle picking by screen-space
   point/segment/polyline distance tests. Same scene, Maya-style gizmos and
   controls as `SelectionManipulator`, but no ID render pass and no
   `glReadPixels`.
2. **`WebGPUComputePicking/`** (WebGPU only, per request) — integer object
   IDs rendered to an `r32uint` target via a second fragment entry point,
   reduced on the GPU by a compute shader (9x9 block around the click,
   atomicMin parallel argmin of packed `dist²|id`), with a 4-byte readback
   instead of mapping the whole frame.

## Files changed (all new)

- `RayPickingSelection/picking_maths.py` — numpy-only ray + 2D distance
  maths (unit-testable headless).
- `RayPickingSelection/SelectionObject.py` — pickable objects with cached
  per-mesh triangle data and `intersect(origin, direction)`.
- `RayPickingSelection/ScreenGizmo.py` — Manipulator variant with
  `pick_handle()` screen-space hit tests (drag maths unchanged).
- `RayPickingSelection/main.py`, `README.md`,
  `tests/test_picking_maths.py` (24 tests).
- `WebGPUComputePicking/ObjectShader.wgsl` — shaded + `fragment_pick`
  (u32 ID) entry points.
- `WebGPUComputePicking/PickCompute.wgsl` — atomicMin pick-reduce kernel.
- `WebGPUComputePicking/ObjectPipeline.py` — shaded MSAA + single-sampled
  r32uint pipelines, `PickResolver` compute wrapper.
- `WebGPUComputePicking/SelectionObject.py`, `main.py`, `README.md`,
  `WebGPUWidget.py` (copied from `SelectionManipulatorWebGPU`).

## Commands run

- `git worktree add .worktrees/picking-alternatives -b agent/picking-alternatives`
- `uv run --group dev pytest RayPickingSelection/tests` — 24/24 pass.
- Headless GPU check (scratchpad): built both WebGPU pipelines + compute
  resolver, rendered a triangle with pick id 7 to a 64x64 r32uint target,
  resolved centre hit (7), corner miss (None) and near-edge slop hit — pass.
- Synthetic-click check (scratchpad, real window): `window.pick()` at every
  object's projected centre returns that object, empty corner returns None,
  gizmo X-arrow tip returns `('axis', Axis.X)` — pass. Caught and fixed one
  real bug: `_gizmo_scale` returned `np.float32`, which `Vec3 * scalar`
  rejects.
- Both demos launched interactively for several seconds — no errors logged.
- `uvx ruff check` (clean) and `uvx ruff format`.
