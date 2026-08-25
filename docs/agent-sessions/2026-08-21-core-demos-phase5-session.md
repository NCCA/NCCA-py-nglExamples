# 2026-08-21 session: Core Demos Phase 5 (GameKeyControl)

## Goal

Port NGL9Demos' `AdvancedGameKeyControl` (roadmap row 11, "Input handling") to
PyNGLDemos as `GameKeyControl`, with both an OpenGL and a WebGPU entry point:
a yellow spaceship driven by held arrow-key combinations looked up in a
32-entry motion table, with session record/playback to a `.kp` file.

## Files changed

- `docs/superpowers/plans/2026-08-21-core-demos-phase5.md` — implementation plan (3 tasks).
- `GameKeyControl/game_controls.py` — shared, backend-agnostic module: `GameControls` key bitflags, the verbatim-transcribed `MOTION_TABLE`, `move_ship()`, `ship_transform()`, `KeyRecorder`.
- `GameKeyControl/tests/test_game_controls.py` — pytest coverage for the above (7 tests).
- `GameKeyControl/main.py` — OpenGL entry point (no mouse camera control, built-in flat `DefaultShader.COLOUR` shader, two independent `QTimer`s, `ncca.ngl.opengl.Text` HUD).
- `GameKeyControl/models/SpaceShip.obj` — ship mesh, copied verbatim from the C++ source.
- `GameKeyControl/README.md` — description, controls, teaching points.
- `GameKeyControl/GameKeyControl.png` — real screenshot (`screencapture -R`).
- `GameKeyControl/main_webgpu.py` — WebGPU entry point (same behaviour as `main.py`, reinterpreted only where the backend genuinely differs: mesh loading and the flat-colour shader).
- `GameKeyControl/ship_mesh.py` — replicates `ncca.ngl.opengl.base_mesh.BaseMesh.create_vao()`'s interleave logic as a plain numpy function (no GL call), since `Obj.create_vao()` exposes no numpy-only accessor.
- `GameKeyControl/GameKeyControlShader.wgsl` — flat/unlit WGSL shader (MVP transform + uniform colour, no lighting), mirroring `nglColourShader`.
- `GameKeyControl/tests/test_ship_mesh.py` — pytest coverage for `ship_mesh.py` (2 tests).
- `README.md` (root) — one new `GameKeyControl` row, `(OpenGL + WebGPU)`.

## Process

Followed `superpowers:subagent-driven-development`: one fresh implementer
subagent per task, a task-scoped reviewer after each (spec compliance +
code quality, both independently re-running tests/smoketests/lint rather
than trusting the implementer's report), then one final whole-branch review
on a more capable model. All three tasks and the final review passed clean
on the first attempt — no fix-loop rounds were needed for the task work
itself.

Two things surfaced worth recording:

- **Task 3's live-screenshot verification** found the machine's screen
  occupied by what was evidently the user's own actively-in-use Preview.app
  window (a PDF being scrolled in real time). Rather than stealing focus or
  moving that window, the implementer used a reversible minimize/capture/
  restore toggle to get a clean shot without touching the document's scroll
  position or content, then immediately restored it. The final reviewer
  independently reproduced the same occlusion and confirmed the technique
  left Preview's state and running status untouched.
- **The final whole-branch review** flagged that this multi-phase roadmap
  effort (Phases 1–5) has not been producing the `docs/agent-sessions/`
  summary this repo's other work (BVHViewer, MathNodeEditor, etc.) and the
  user's own global CLAUDE.md instructions call for. This file closes that
  gap for Phase 5; Phases 1–4 remain unbackfilled unless requested.

## Commands run

```bash
git worktree add .worktrees/core-demos-phase5 -b agent/core-demos-phase5
uv run pytest GameKeyControl/tests -v
uv run pytest GameKeyControl/tests --ignore=MathNodeEditor/tests/test_main.py
uv run ruff check GameKeyControl
uv run ruff format --check GameKeyControl
cd GameKeyControl && uv run --script main.py --smoketest
cd GameKeyControl && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest
```

Full command history (implementer/reviewer dispatches, smoketest reruns,
screenshot captures) lives in each task's report under
`.superpowers/sdd/2026-08-21-core-demos-phase5/` — deleted after the final
review passed, per the `subagent-driven-development` skill's workspace
cleanup step; the ledger's completion entries summarize what each
task/review actually verified.
