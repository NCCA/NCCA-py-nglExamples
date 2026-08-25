# 2026-08-25 -- Ruff class variables

## Goal

Mark class-level demo constants and shared registries as `ClassVar` so they are
not mistaken for mutable instance defaults.

## Files changed

The OpenGL and WebGPU affine, BVH, skinned-mesh, voxel, easing, interpolation,
and PBR demos now annotate their 13 class-level mutable values.

## Commands run

```bash
git worktree add .worktrees/ruff-cleanup-2 -b agent/ruff-cleanup-2
uv run ruff check . --select RUF012 --output-format concise
uv run python -m compileall -q AffineTransforms BVHViewer EasingFunctions Interpolation PBR/PBRTexture SkinnedMeshImport Voxels
```

The focused Ruff check and compilation both passed.
