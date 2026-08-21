# 2026-08-21 session: Core Demos Phase 7

## Goal

Port the final three demos in the core NGL9Demos roadmap: `ResetLine`,
`MorphObj` and `OctreeAbstract`. Each demo has OpenGL and WebGPU entry points,
shared Python data or simulation code, tests, documentation and a real preview.

## Files changed

- `ResetLine/` — seeded blade geometry, primitive-restart OpenGL renderer,
  WebGPU line-list renderer, shaders, tests, README and preview.
- `MorphObj/` — the three Bruce pose assets, base-plus-delta mesh packing,
  OpenGL and WebGPU morph shaders, tests, README and preview.
- `OctreeAbstract/` — perfect octree, CPU particle simulation, instanced sphere
  renderers, shaders, tests, README and preview.
- `README.md` — catalogue rows for all three demos.
- `docs/superpowers/plans/2026-08-21-core-demos-phase7.md` — implementation
  plan.
- `docs/agent-sessions/2026-08-21-core-demos-phase7-session.jsonl` — Codex
  session export.
- `docs/agent-sessions/2026-08-21-core-demos-phase7-session.md` — this summary.

## Commands run

```bash
git status --short --branch
git merge --no-ff Version1.0 -m "merge: update core demos phase 7 base"
uv run pytest ResetLine/tests MorphObj/tests OctreeAbstract/tests -v
uv run ruff format ResetLine MorphObj OctreeAbstract
uv run ruff check ResetLine MorphObj OctreeAbstract
uv run ruff format --check ResetLine MorphObj OctreeAbstract
uv run python -m compileall -q ResetLine MorphObj OctreeAbstract
uv run --script ResetLine/main.py --rows 20 --cols 20 --smoketest 500
QT_QPA_PLATFORM=offscreen uv run --script ResetLine/main_webgpu.py --rows 20 --cols 20 --smoketest 700
uv run --script MorphObj/main.py --smoketest 700
QT_QPA_PLATFORM=offscreen uv run --script MorphObj/main_webgpu.py --smoketest 900
uv run --script OctreeAbstract/main.py --grid 4 --smoketest 900
QT_QPA_PLATFORM=offscreen uv run --script OctreeAbstract/main_webgpu.py --grid 4 --smoketest 1000
uv run pytest -q
uv run pytest -q --ignore=MathNodeEditor/tests/test_main.py
uv run ruff check .
uv build
git diff --check
git commit -m "feat(core-demos): implement phase 7 ports"
cp /Users/jmacey/.codex/sessions/2026/08/21/rollout-2026-08-21T11-03-15-01a023c6-531b-7443-994b-e02e03cb6c67.jsonl docs/agent-sessions/2026-08-21-core-demos-phase7-session.jsonl
```

I used red-green-refactor for the blade field, morph packing and controls, and
the octree simulation. The first combined run failed because none of the three
shared modules existed. Later failing tests covered the real OBJ normal-index
contract, OpenGL entry-point imports and the shared bounding-box line geometry.
All 23 Phase 7 tests now pass.

All six live smoketests pass and the changed Python is Ruff clean and
formatted. The preview images were captured from the real WebGPU frame buffers
and checked visually. The repository passes 773 tests when the existing
`MathNodeEditor/tests/test_main.py` duplicate-module collection clash is
excluded.

The repository baselines are unchanged: the unfiltered pytest run stops on the
duplicate `test_main.py` name, repository-wide Ruff reports 47 unrelated unused
imports, and `uv build` reaches setuptools but fails because this flat demo
collection has many top-level folders and no explicit package discovery. These
are not Phase 7 failures.

I reused the clean `.worktrees/core-demos-phase7` worktree and updated it from
`Version1.0` before editing. There is no `RTK.md` in this checkout or the nearby
paths checked, so I followed the supplied AGENTS.md instructions directly. The
work remains local on `agent/core-demos-phase7`; nothing was pushed.
