# 2026-08-21 session: Core Demos Phase 6 (ImageMaze)

## Goal

Port NGL9Demos' `ImageMaze` to PyNGLDemos with OpenGL and WebGPU entry
points. The demo reads a small PNG, draws every non-white pixel as a coloured
cube and lets a troll move through the white pixels.

## Files changed

- `docs/superpowers/plans/2026-08-21-core-demos-phase6.md` — implementation
  plan.
- `ImageMaze/maze_scene.py` — shared image coordinates, wall extraction,
  actor movement and camera direction.
- `ImageMaze/mesh_data.py` — coloured triangle and wireframe vertex data for
  WebGPU.
- `ImageMaze/tests/` — 15 pytest tests for the shared scene and mesh data.
- `ImageMaze/main.py` — OpenGL entry point.
- `ImageMaze/main_webgpu.py` and `ImageMaze/ImageMazeShader.wgsl` — WebGPU
  entry point and flat-colour shader.
- `ImageMaze/maps/` — the three PNG maps copied from the C++ demo.
- `ImageMaze/ImageMaze.png` — preview captured from the real WebGPU frame
  buffer.
- `ImageMaze/README.md` and the root `README.md` — running instructions,
  controls and catalogue entry.
- `docs/agent-sessions/2026-08-21-core-demos-phase6-session.jsonl` — Codex
  session export.
- `docs/agent-sessions/2026-08-21-core-demos-phase6-session.md` — this summary.

## Commands run

```bash
git status --short --branch
uv run pytest ImageMaze/tests -v
uv run ruff format ImageMaze
uv run ruff check ImageMaze
uv run ruff format --check ImageMaze
uv run python -m compileall -q ImageMaze
cd ImageMaze && uv run --script main.py --smoketest 700
cd ImageMaze && uv run --script main_webgpu.py --smoketest 1000
uv run python -c '<WebGPU framebuffer capture>'
uv run pytest -q
uv run pytest -q --ignore=MathNodeEditor/tests/test_main.py
uv run ruff check .
uv build
git diff --check
cp /Users/jmacey/.codex/sessions/2026/08/21/rollout-2026-08-21T10-04-34-01a02390-9a4b-7222-ab73-4d4144c140a1.jsonl docs/agent-sessions/2026-08-21-core-demos-phase6-session.jsonl
```

I used red-green-refactor for the image mapping, movement rules, actor camera
direction and WebGPU mesh conversion. The initial test run failed because the
shared modules did not exist; the wireframe test also failed before the edge
builder was added. All 15 ImageMaze tests now pass. Both live smoketests pass,
the changed Python is Ruff clean and formatted, bytecode compilation passes,
and the WebGPU preview was checked visually.

The repository-wide test command still has the existing duplicate
`test_main.py` collection clash between BVHViewer and MathNodeEditor. With the
MathNodeEditor file ignored, all 751 collected tests pass. Repository-wide Ruff
still reports 47 existing unused imports in unrelated demos. `uv build` still
fails because setuptools auto-discovers the repository's many top-level demo
folders as packages. These three baseline problems are outside ImageMaze; the
targeted checks are clean.

The existing local worktree `.worktrees/core-demos-phase6` was already clean
and on `agent/core-demos-phase6`, so I used it directly. Nothing was pushed, as
requested. There is no `RTK.md` in this checkout, so I followed the supplied
AGENTS.md instructions directly.
