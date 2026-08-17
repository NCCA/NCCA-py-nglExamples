# 2026-08-17 session: merge BVHViewer main setup

## Goal

Fast-forward the finished BVHViewer main setup work into `Version1.0` and
check it again from the merged branch.

## Files changed

- No source files changed during the merge; `Version1.0` moved from `be061be`
  to `b6c682a`
- `docs/agent-sessions/2026-08-17-bvhviewer-main-merge-session.jsonl` — exported
  Codex session
- `docs/agent-sessions/2026-08-17-bvhviewer-main-merge-session.md` — this merge
  record

## Commands run

```bash
git status --short --branch
git merge-base --is-ancestor Version1.0 agent/bvhviewer-main
git merge --ff-only agent/bvhviewer-main
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run pytest BVHViewer/tests -q
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run ruff check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run ruff format --check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-merge-uv-cache uv run python -m compileall -q BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-main-uv-cache uv build
git diff --check
```

The merged branch passed all 23 BVHViewer tests, Ruff, format checking and
bytecode compilation. The root build still stops at the existing setuptools
flat-layout package discovery error; it is not caused by the BVHViewer change.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
