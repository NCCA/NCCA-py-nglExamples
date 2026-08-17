# 2026-08-17 session: remove BVHViewer worktrees

## Goal

Remove the old BVHViewer worktrees after their changes were merged into
`Version1.0`, whilst keeping the branches and commits available.

## Worktrees removed

- `.worktrees/bvhviewer-main`
- `.worktrees/bvhviewer-gui`
- `.worktrees/bvhviewer-playback-range`

All three worktrees were clean and their branch tips were already ancestors of
`Version1.0`. The `agent/bvhviewer-main`, `agent/bvhviewer-gui` and
`agent/bvhviewer-playback-range` branches were not deleted.

## Files changed

- `docs/agent-sessions/2026-08-17-bvhviewer-worktree-cleanup-session.jsonl` —
  exported Codex session
- `docs/agent-sessions/2026-08-17-bvhviewer-worktree-cleanup-session.md` — this
  summary

## Commands run

```bash
git status --short --branch
git worktree list --porcelain
git merge-base --is-ancestor agent/bvhviewer-main Version1.0
git merge-base --is-ancestor agent/bvhviewer-gui Version1.0
git merge-base --is-ancestor agent/bvhviewer-playback-range Version1.0
git worktree remove .worktrees/bvhviewer-main
git worktree remove .worktrees/bvhviewer-gui
git worktree remove .worktrees/bvhviewer-playback-range
git worktree add .worktrees/bvhviewer-cleanup-record -b agent/bvhviewer-cleanup-record Version1.0
UV_CACHE_DIR=/private/tmp/bvhviewer-cleanup-uv-cache uv run --active pytest BVHViewer/tests -q
git diff --check
```

No source files changed. The BVHViewer suite still passed all 51 tests before
committing the record. The temporary cleanup-record worktree is removed after
this record is merged.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
