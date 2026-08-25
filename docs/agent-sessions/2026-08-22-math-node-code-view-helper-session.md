# 2026-08-22 session: Code view helper cleanup

## Goal

Only emit the generated component-multiply helper when an Output branch uses a
regular Multiply node.

## Files changed

- `MathNodeEditor/math_graph.py` — emit `_component_multiply` on first use.
- `MathNodeEditor/tests/test_math_graph.py` — check the helper is omitted from
  branches which do not need it.

## Commands run

```bash
git worktree add .worktrees/math-node-code-view-helper -b agent/math-node-code-view-helper agent/math-node-code-view
uv run --group dev pytest MathNodeEditor/tests/test_math_graph.py -k generate_python
QT_QPA_PLATFORM=offscreen uv run --group dev pytest MathNodeEditor/tests
uv run --group dev ruff check MathNodeEditor
uv run --group dev ruff format --check MathNodeEditor
uv run python -m compileall -q MathNodeEditor
QT_QPA_PLATFORM=offscreen uv run python MathNodeEditor/main.py --smoketest 500
uv build
```

The MathNodeEditor suite passes with 266 tests, Ruff and compileall pass, and
the smoketest prints `SMOKETEST OK`. `uv build` still stops at the existing
flat-layout package-discovery error.
