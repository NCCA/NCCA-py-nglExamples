# 2026-08-17 session: BVHViewer GUI

## Goal

Turn BVHViewer into a desktop GUI with a file menu and the sort of frame
timeline and transport controls used in Maya or Houdini.

## Files changed

- `BVHViewer/main.py` — now has a `QMainWindow` shell around the OpenGL
  viewport, File, Playback and View menus, keyboard shortcuts, BVH file
  loading, a playback timer and a status bar
- `BVHViewer/timeline.py` — adds the frame scrubber, editable current-frame
  field, clip range and frame-rate display, and first, previous, play/pause,
  next and last transport buttons
- `BVHViewer/bvh.py` — adds clamped seeking for timeline input
- `BVHViewer/bvh_scene.py` — adds single-character replacement, scene seeking
  and frame-count queries
- `BVHViewer/tests/test_bvh.py`, `BVHViewer/tests/test_bvh_scene.py`,
  `BVHViewer/tests/test_main.py` and `BVHViewer/tests/test_timeline.py` — cover
  seeking, scene replacement, timeline signals, file-menu actions, initial
  clip loading and transport behaviour
- `BVHViewer/README.md` — updates the run command and documents the GUI and
  shortcuts
- `docs/agent-sessions/2026-08-17-bvhviewer-gui-session.jsonl` — exported
  Codex session
- `docs/agent-sessions/2026-08-17-bvhviewer-gui-session.md` — this summary

## Commands run

```bash
git status --short --branch
git worktree add .worktrees/bvhviewer-gui -b agent/bvhviewer-gui
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active pytest BVHViewer/tests/test_bvh.py BVHViewer/tests/test_bvh_scene.py -q
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active pytest BVHViewer/tests/test_timeline.py -q
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active pytest BVHViewer/tests -q
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active ruff check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active ruff format --check BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active python -m compileall -q BVHViewer
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active python main.py --smoketest 700
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv build
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active pytest -q
UV_CACHE_DIR=/private/tmp/bvhviewer-gui-uv-cache uv run --active pytest --import-mode=importlib -q
git diff --check
```

The work followed red-green-refactor: the first model/scene run had seven
expected failures and the first timeline/window run had five. The final
BVHViewer run passed all 39 tests, Ruff, the formatting check, bytecode
compilation and the timed Qt/OpenGL smoke test.

The repository-wide pytest command still stops during collection because
several demo test folders contain an un-packaged `test_main.py`. Importlib mode
gets past that collision but then exposes existing `PBR/HDRIBaker` path import
errors. `uv build` also reaches the existing setuptools failure caused by the
repository's many top-level demo directories. Neither failure involves the
BVHViewer changes.

There is no `RTK.md` in this checkout, so I followed the supplied AGENTS.md
instructions directly.
