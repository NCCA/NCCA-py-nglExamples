# 2026-08-21 session: Merge Core Demos Phase 6

## Goal

Merge `agent/core-demos-phase6` into `Version1.0` locally and check the
ImageMaze demo again on the merged branch.

## Files changed

- `ImageMaze/`, the root `README.md`, Phase 6 plan and Phase 6 session record
  were merged from `agent/core-demos-phase6`.
- `docs/agent-sessions/2026-08-21-core-demos-phase6-merge-session.jsonl` —
  exported Codex session.
- `docs/agent-sessions/2026-08-21-core-demos-phase6-merge-session.md` — this
  summary.

## Commands run

```bash
git status --short --branch
git merge --no-ff agent/core-demos-phase6 -m "merge: integrate core demos phase 6"
uv run pytest ImageMaze/tests -q
uv run ruff check ImageMaze
uv run ruff format --check ImageMaze
uv run python -m compileall -q ImageMaze
cd ImageMaze && uv run --script main.py --smoketest 700
cd ImageMaze && uv run --script main_webgpu.py --smoketest 1000
git diff --check
cp /Users/jmacey/.codex/sessions/2026/08/21/rollout-2026-08-21T10-04-34-01a02390-9a4b-7222-ab73-4d4144c140a1.jsonl docs/agent-sessions/2026-08-21-core-demos-phase6-merge-session.jsonl
```

The merge completed without conflicts. All 15 ImageMaze tests pass, both
renderers pass their live smoketests, and the ImageMaze Python files are Ruff
clean and formatted.

`SkinnedMeshImport/models/Walk.fbx` was already untracked in the target
worktree. I left it alone as requested. Nothing was pushed.
