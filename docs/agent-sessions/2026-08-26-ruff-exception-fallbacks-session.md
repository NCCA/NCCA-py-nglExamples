# 2026-08-26 -- Ruff event fallbacks

## Goal

Replace broad exception handling in the PySide mouse-event compatibility paths.

## Files changed

The 2D drawing and spatial-hash WebGPU demos now catch `AttributeError` when
an older PySide event does not provide `position()`. Other failures are no
longer hidden by the fallback.

## Commands run

```bash
git worktree add .worktrees/ruff-exceptions -b agent/ruff-exceptions agent/ruff-cleanup-2
uv run ruff check ... --select BLE001 --output-format concise
uv run python -m compileall -q 2DDrawingOpenGL SimpleComputeWebGPU WebGPUCompute/SpatialHash2D WebGPUCompute/SpatialHash3D
```

Compilation passed. The focused Ruff report now contains only the remaining
WebGPU initialisation and rendering boundaries in these modules.
