# 2026-08-25 -- UTF-8 read_text fix

## Goal

Work out why five tests failed on the Windows CI runner whilst passing on Linux
and macOS, and fix the Unicode one.

## What was actually wrong

Three of the five failures were unrelated to Unicode (see "Left alone" below).
The Unicode one came down to a bare `Path.read_text()`, which decodes using the
locale encoding -- UTF-8 on Linux and macOS, but
[cp1252 on Windows](https://docs.python.org/3/library/functions.html#open).

`MathNodeEditor/tests/test_main.py` walks every module in the demo folder with
`ast.parse`. One of them, `node_visuals.py`, holds the maths symbols used to
draw the node icons, and the superscript minus in there (U+207B) encodes as
`E2 81 BB`. cp1252 leaves `0x81` undefined, so the read raised before the test
got anywhere near checking imports. CI reported the bad byte at position 2285
and I saw 2230 locally, which is just the CRLF checkout on Windows shifting the
offset.

`RunDemos.py` loaded each demo's README the same way. 61 of the 75 top-level
READMEs contain non-ASCII, so on Windows the launcher would have shown mojibake
for most of them, and raised outright on `BVHViewer/README.md`.

## Files changed

- `MathNodeEditor/tests/test_main.py` -- `read_text(encoding="utf-8")`
- `RunDemos.py` -- same, in `_load_readme`

## Left alone

The other four Windows failures are one problem, not two, and are a design
question rather than a bug to patch:

- `test_long_quaternion_palette_label_is_not_clipped`
- `test_teaching_example_nodes_do_not_overlap` for the normal matrix,
  quaternion rotation and quaternion slerp examples

Node widths come from `QFontMetrics(...).horizontalAdvance(title)` in
`graphics_items.py`, but the example graphs in `MathNodeEditor/examples/` store
hard-coded absolute positions. The palette does `setFixedWidth(240)` whilst its
buttons size themselves to their text. Windows' default UI font is wider than
the macOS one, so nodes grow into their neighbours and the long button labels
clip. Measured on macOS the "Quaternion from Axis Angle" button wants 179px
against 204px available; Windows wants 346px. So the test was passing on 25px of
slack rather than because the layout was sound.

Fixing it properly means letting the palette follow its `sizeHint()` and laying
the examples out from measured widths instead of baked coordinates. Left for a
separate branch.

## Commands run

```bash
git worktree add .worktrees/fix-read-text-encoding -b agent/fix-read-text-encoding
QT_QPA_PLATFORM=offscreen uv run pytest --ignore=SkinnedMeshImport/tests/test_mesh_maths.py --ignore=PBR/HDRIBaker/tests/test_bake_ibl.py -q
uv run ruff check MathNodeEditor/tests/test_main.py RunDemos.py
uv run ruff format --check MathNodeEditor/tests/test_main.py RunDemos.py
```

808 passed. To check the fix without a Windows box, decode the files as cp1252
on purpose -- `node_visuals.py` and `BVHViewer/README.md` are the two that blow
up.
