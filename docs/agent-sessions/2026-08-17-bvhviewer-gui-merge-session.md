# 2026-08-17 session: merge BVHViewer GUI

## Goal

Merge the finished BVHViewer GUI, playback range and FPS controls into
`Version1.0`, then check the result from the merged branch.

## Files changed

- No source files changed during the merge; `Version1.0` moved from `e1bdbb8`
  to `cab7df3`
- `docs/agent-sessions/2026-08-17-bvhviewer-gui-merge-session.jsonl` — exported
  Codex session
- `docs/agent-sessions/2026-08-17-bvhviewer-gui-merge-session.md` — this merge
  record

## Commands run

```bash
git status --short --branch
git merge-base --is-ancestor Version1.0 agent/bvhviewer-playback-range
git log --oneline Version1.0..agent/bvhviewer-playback-range
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run --active pytest BVHViewer/tests -q
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run --active ruff check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run --active ruff format --check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run --active python -m compileall -q BVHViewer
git merge --ff-only agent/bvhviewer-playback-range
git diff --check
```

The feature branch and merged `Version1.0` branch both passed all 51
BVHViewer tests, Ruff, the formatting check and bytecode compilation. The
history was a direct descendant of `Version1.0`, so the merge was a clean
fast-forward with no conflict resolution.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
