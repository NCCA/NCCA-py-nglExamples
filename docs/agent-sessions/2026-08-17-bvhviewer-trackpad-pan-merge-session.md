# 2026-08-17 session: merge BVHViewer trackpad pan

## Goal

Fast-forward the tested BVHViewer trackpad pan change into `Version1.0` and
check it again from the merged branch.

## Files changed

- `Version1.0` moved from `efcf299` to `b38e5d7`
- `docs/agent-sessions/2026-08-17-bvhviewer-trackpad-pan-merge-session.jsonl`
  — exported Codex merge session
- `docs/agent-sessions/2026-08-17-bvhviewer-trackpad-pan-merge-session.md`
  — this merge record

## Commands run

```bash
git status --short --branch
git -C .worktrees/bvhviewer-fn-middle-click status --short --branch
git merge-base --is-ancestor Version1.0 agent/bvhviewer-fn-middle-click
git log --oneline Version1.0..agent/bvhviewer-fn-middle-click
git merge --ff-only agent/bvhviewer-fn-middle-click
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv run pytest BVHViewer/tests -q
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv run ruff check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv run ruff format --check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv run python -m compileall -q BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv run BVHViewer/main.py --smoketest 700
UV_CACHE_DIR=/private/tmp/bvhviewer-fn-middle-uv-cache uv build
cp /Users/jmacey/.codex/sessions/2026/08/17/rollout-2026-08-17T15-35-33-01a01026-2f6d-7c51-9905-09cdccbad20f.jsonl docs/agent-sessions/2026-08-17-bvhviewer-trackpad-pan-merge-session.jsonl
git diff --check
```

The merge was a direct fast-forward with no conflicts. The merged branch
passed all 69 BVHViewer tests, Ruff, format checking, bytecode compilation and
the live Qt/OpenGL smoke test.

The root `uv build` command still stops at the existing setuptools flat-layout
package discovery error. It is unrelated to the BVHViewer change.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
