# 2026-08-24 session: Documentation and README style pass

## Goal

Check every README in the repository is still accurate and reads in my own
voice rather than the default AI documentation register, and fix whatever
had drifted.

## What I found

The corpus was in better shape than I expected — 99 of the 114 READMEs came
back clean. A first scan flagged `WebGPUComputePicking/README.md` for nine
emoji, but those turned out to be the box-drawing arrows in its ASCII
pipeline diagram, so it needed no change. Two real problems were left: a
handful of American spellings in prose, and `BVHViewer` missing from the
root catalogue table and from its own README's screenshot line.

Three things I deliberately left alone, because they only look like
mistakes:

- The American spellings in `ColourSelectionOpenGL/README.md` are inside a
  verbatim quote of `SelectObject.py`, so they have to match the code.
- `Optimized Spatial Hashing`, `Modeling a Class of Fuzzy Objects` and
  `Computer Synthesized Pictures` are cited paper titles.
- `PBR/PBRTexture/README.md`'s `## Future work` is specific to that demo
  rather than filler, so it stays.

Also worth recording: BSD `sed` on macOS does not understand `\b`, so a
`s/\bmaximize\b/` substitution silently changed nothing and looked like it
had worked. Literal patterns fixed it.

## Files changed

- `README.md` — added the `BVHViewer` row to the Animation table.
- `BVHViewer/README.md` — embedded `BVHViewer.png` after the title, matching
  every other demo; `maximize` to `maximise`.
- `AffineTransforms/README.md`, `ShadedGrid/README.md` — `visualization` /
  `visualizes` to `visualisation` / `visualises`.
- `LoadShaderFromJSon/README.md` — `initialization` to `initialisation`,
  `artifact` to `artefact` (the spelling the rest of the repo uses).
- `RayPickingSelection/README.md`, `SelectionManipulator/README.md` —
  `center cube` to `centre cube`, which is what both files' own controls
  tables already said.
- `SkinnedMeshImport/README.md` — `maximize` to `maximise`.
- `SkeletalAnimation/README.md` — `meter` to `metre`.
- `Billboards/README.md` — dropped "Robust to".
- `CurveDemos/README.md` — dropped "comprehensive".

## Commands run

```bash
git worktree add .worktrees/docs-style-pass -b agent/docs-style-pass
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

787 tests pass, Ruff reports no problems, and 320 files are already
formatted. Every demo folder is now linked from the root README (80 of 80),
no README references a missing image, and no README points at a script or
shader file that has since been renamed.
