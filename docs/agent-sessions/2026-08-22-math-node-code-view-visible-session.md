# 2026-08-22 session: Visible code view

## Goal

Show the Python code dock by default while keeping the View action and saved
window geometry working.

## Files changed

- `MathNodeEditor/node_editor.py` — show the dock on startup and use a width
  which fits existing saved windows.
- `MathNodeEditor/tests/test_node_editor.py` — cover the visible default.

## Commands run

```bash
git worktree add .worktrees/math-node-code-view-visible -b agent/math-node-code-view-visible agent/math-node-code-view-helper
QT_QPA_PLATFORM=offscreen uv run --group dev pytest MathNodeEditor/tests/test_node_editor.py -k visible_read_only
QT_QPA_PLATFORM=offscreen uv run --group dev pytest MathNodeEditor/tests
uv run --group dev ruff check MathNodeEditor
uv run --group dev ruff format --check MathNodeEditor
uv run python -m compileall -q MathNodeEditor
QT_QPA_PLATFORM=offscreen uv run python MathNodeEditor/main.py --smoketest 500
uv build
```

The MathNodeEditor suite passes with 266 tests. Ruff, compileall and the
smoketest pass. `uv build` still stops at the existing flat-layout
package-discovery error.
