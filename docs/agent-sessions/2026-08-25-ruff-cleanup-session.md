# 2026-08-25 -- Ruff cleanup

## Goal

Start reducing the errors reported by `uv run ruff check .` without changing the
behaviour of the demos.

## Files changed

The first batch changes 29 demo and test files. It removes redundant casts and
unpacked values, modernises type aliases and type hints, replaces a few simple
constructs, and removes invalid shebangs. It also uses `sys.exit()` and bare
`raise` where Ruff requested them.

## Commands run

```bash
git worktree add .worktrees/ruff-cleanup -b agent/ruff-cleanup
uv run ruff check .
uv run pytest GeometryTessellation/tests/test_tess_grid.py MassSpring/tests/test_rk4.py PBR/HDRIBaker/tests/test_bake_ibl.py SceneGraph/tests/test_scene_graph.py SkeletalAnimation/tests/test_skinning_maths.py
uv build
```

The targeted tests passed (41 tests). The full Ruff run reduced from 156 to 111
errors. `uv build` still fails because setuptools finds the repository's many
top-level demo packages; this was not caused by this lint batch.
