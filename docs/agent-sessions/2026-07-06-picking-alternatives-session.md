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

## Follow-up (same day): manipulators for WebGPUComputePicking

Added the full Maya-style gizmo to the compute demo (it was initially
selection-only):

- `WebGPUComputePicking/Manipulator.py` — port of the WebGPU manipulator
  with reserved *integer* pick IDs (`GIZMO_ID_BASE + 1..4`, top of the
  20-bit range) instead of reserved colours.
- `ObjectPipeline.py` — new `GizmoPipeline`: flat-colour pipeline for
  drawing plus an ID sibling writing a uniform u32 into the r32uint target.
- `PickCompute.wgsl` — priority packing: objects pack `distance² + 1`,
  gizmo IDs pack distance `0`, so a handle anywhere in the 9x9 block beats
  an object even under the exact click pixel (caught during design: with
  naive `d2 = 0` an object at the click centre would out-sort the handle).
- `main.py` — Q/W/E/R modes, gizmo draw/ID passes (one small pass per
  part), drag handling; `README.md` updated.

Verification: headless GPU test extended (gizmo ID pass + priority rule),
and a synthetic-click test against the real widget — all 5 objects, empty
miss, translate X arrow, centre cube (priority over the teapot behind it)
and rotate ring all resolve correctly. Ruff clean, 24/24 maths tests pass.
