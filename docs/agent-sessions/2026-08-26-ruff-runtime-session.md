# 2026-08-26 -- Ruff runtime checks

## Goal

Make the demo runners' subprocess failure policy explicit and avoid blocking an
async notebook cell.

## Commands run

```bash
uv run ruff check RunDemos.py smoketest_all.py Notebooks/TestCanvas.py --select PLW1510,B018,ASYNC251 --output-format concise
uv run python -m compileall -q RunDemos.py smoketest_all.py Notebooks/TestCanvas.py
```

The focused Ruff check and compilation passed.
