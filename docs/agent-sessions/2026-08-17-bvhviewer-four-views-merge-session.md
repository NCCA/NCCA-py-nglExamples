# 2026-08-17 session: merge BVHViewer four views

## Goal

Merge the Maya-style BVHViewer four-view layout into `Version1.0` and check it
again from the merged branch.

## Files changed

- `docs/agent-sessions/2026-08-17-bvhviewer-four-views-merge-session.jsonl` —
  exported Codex session
- `docs/agent-sessions/2026-08-17-bvhviewer-four-views-merge-session.md` — this
  summary

The feature files and their first session record were merged from
`agent/bvhviewer-four-views` at commit `a4f8f6a`.

## Commands run

```bash
git status --short --branch
git merge --no-ff agent/bvhviewer-four-views -m "merge(bvhviewer): add four-view layout"
uv run pytest BVHViewer/tests -q
uv run ruff check BVHViewer/main.py BVHViewer/tests/test_viewport_camera.py
uv run ruff format --check BVHViewer/main.py BVHViewer/tests/test_viewport_camera.py
.venv/bin/python -m compileall -q BVHViewer
git diff --check
uv run python -c '<four-view merge Qt/OpenGL smoke test>'
uv build
git worktree add .worktrees/bvhviewer-four-views-merge-record -b agent/bvhviewer-four-views-merge-record
cp /Users/jmacey/.codex/sessions/2026/08/17/rollout-2026-08-17T13-55-07-01a00fca-3c64-75c0-8a15-f8803c7ec1b6.jsonl docs/agent-sessions/2026-08-17-bvhviewer-four-views-merge-session.jsonl
```

The merge produced `db14e7f`. All 63 BVHViewer tests pass on `Version1.0`, as
do Ruff, formatting, bytecode compilation and the live four-view OpenGL smoke
test.

`uv build` still fails because setuptools treats the repository's many
top-level demo folders as packages. This is the same existing flat-layout
configuration problem recorded by the feature session and is unrelated to the
BVHViewer merge.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
