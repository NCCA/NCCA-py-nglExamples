# 2026-08-26 -- Math graph exception types

## Goal

Use `TypeError` when a graph operation is called with the wrong kind of node.

## Files changed

`MathGraph` now raises `TypeError` for invalid node kinds, and the graph tests
check that public behaviour.

## Commands run

```bash
git worktree add .worktrees/ruff-math-graph -b agent/ruff-math-graph agent/ruff-exceptions
uv run pytest MathNodeEditor/tests/test_math_graph.py
uv run ruff check MathNodeEditor/math_graph.py --select TRY004 --output-format concise
```

The graph tests passed (86 tests) and the focused Ruff check is clean.
