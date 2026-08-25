# 2026-08-17 session: MathNodeEditor ARCHITECTURE.md

## Goal

Write a document explaining the structure and architecture of
`MathNodeEditor` and how to add new nodes, as a readable companion to the
"Adding a new operation" checklist comment already living in
`math_graph.py` from the previous session.

Before starting, found an unrelated uncommitted change on `Version1.0` to
`MathNodeEditor/README.md` (a wording/whitespace tweak, not from this
session) — asked and was told to keep and commit it as-is before branching.

## Files changed

- `MathNodeEditor/README.md` — committed the pending wording/whitespace
  fix, then added a "Developer notes" section linking to the new doc.
- `MathNodeEditor/ARCHITECTURE.md` (new) — module map (what each file owns
  and depends on); the calculation graph's table-driven `Operation` design
  in `math_graph.py`; the visual layer (`BaseNodeItem` subclasses,
  `PortItem`/`ConnectionItem`, `node_visuals.py`'s single source of truth
  for icons/colours, `mesh_view.py`'s shared-state-vs-per-surface-VAO
  split); `canvas.py`'s role gluing the graph model to graphics items,
  wire-dragging, `update_outputs()`, and JSON save/load; the node
  creation UI in `palette.py`/`node_editor.py`; and prose versions of the
  "adding a new operation" and "adding a whole new node kind" checklists,
  pointing back at the code comment as the authoritative version.

## Commands run

```bash
uv run pytest MathNodeEditor/tests -q   # 259 passed, both before and after the doc-only change
```

No lint/format run — `ruff` doesn't touch Markdown, and the pre-commit
hook skipped both files (Python-only) when committing.
