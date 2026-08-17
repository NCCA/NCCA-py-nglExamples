# 2026-08-17 session: remove old worktrees

## Goal

Remove the old linked worktrees whilst keeping their branches and commits
available.

## Worktrees removed

- `.worktrees/bvhviewer-main`
- `.worktrees/bvhviewer-gui`
- `.worktrees/bvhviewer-playback-range`
- `.worktrees/architecture-rewrite`
- `.worktrees/math-node-examples`
- `.worktrees/math-node-hardening`
- `.worktrees/math-node-icons`
- `.worktrees/math-node-icons-merge-session`
- `.worktrees/math-node-main-window`
- `.worktrees/math-node-names`
- `.worktrees/math-transform-node`
- `.worktrees/readme-rewrite`
- `.worktrees/scifiui-webgpu`
- `.worktrees/texture-pack-parser`
- `.worktrees/texture-parser-docs`

The three BVHViewer worktrees were checked as clean and merged before removal.
The final audit showed that all the previously linked worktree directories had
been removed. Their branch refs were not deleted, so the committed work remains
available.

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
git branch --format='%(refname:short) %(objectname:short)'
git worktree list
```

No source files changed. The BVHViewer suite still passed all 51 tests before
committing the record. The temporary cleanup-record worktree is removed after
this record is merged.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
