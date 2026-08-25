# 2026-08-17 session: merge BVHViewer WebGPU

## Goal

Fast-forward the tested BVHViewer WebGPU version into `Version1.0` and check it
again from the merged branch.

## Files changed

- `Version1.0` moved from `36484e8` to `38b8d13`
- `docs/agent-sessions/2026-08-17-bvhviewer-webgpu-merge-session.jsonl`
  — exported Codex merge session
- `docs/agent-sessions/2026-08-17-bvhviewer-webgpu-merge-session.md`
  — this merge record

## Commands run

```bash
git status --short --branch
git merge-base --is-ancestor Version1.0 agent/bvhviewer-webgpu
git log --oneline Version1.0..agent/bvhviewer-webgpu
git merge --ff-only agent/bvhviewer-webgpu
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv run --active pytest BVHViewer/tests -q
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv run --active ruff check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv run --active ruff format --check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv run --active python -m compileall -q BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv run --active python BVHViewer/main_webgpu.py --smoketest 1000
UV_CACHE_DIR=/private/tmp/bvhviewer-webgpu-uv-cache uv build
git worktree add .worktrees/bvhviewer-webgpu-merge-record -b agent/bvhviewer-webgpu-merge-record
cp /Users/jmacey/.codex/sessions/2026/08/17/rollout-2026-08-17T17-58-02-01a010a8-a2bf-7103-a90d-fbd827dda201.jsonl docs/agent-sessions/2026-08-17-bvhviewer-webgpu-merge-session.jsonl
git diff --check
```

The feature merge was a direct fast-forward with no conflicts. The merged
branch passed all 77 BVHViewer tests, Ruff, format checking, bytecode compilation
and the live Qt/WebGPU smoke test.

The root `uv build` command still stops at the existing setuptools flat-layout
package discovery error. It finds the demo folders as top-level packages, so
this is unrelated to the BVHViewer merge.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
