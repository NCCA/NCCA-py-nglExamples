# 2026-07-17 — writing-style and consistency audit

## Goal

Audit every README.md and all Python code comments against the jon-writing-style guidelines, plus a general consistency and stray-file review. Produce a plan of what to change (agent work vs Jon's decisions) — no fixes applied this session.

## Files changed

- `docs/superpowers/plans/2026-07-17-style-consistency-audit.md` — the audit findings and execution plan (new, uncommitted)
- `docs/agent-sessions/2026-07-17-style-audit-session.md` — this record

No demo code or README was modified.

## How it was done

Three parallel read-only survey agents: one over all 84 READMEs, one over the Python comments/docstrings, one on consistency/stray files (that one hit a session limit part-way and its remaining checks were finished inline: stray-file triage, root-README cross-check, shebang/executable audit, worktree and agent-branch state, pytest collection).

## Commands run (read-only)

- `git status` / `git log` / `git worktree list` / `git branch` + `merge-base --is-ancestor` per agent branch
- `find` / `grep` sweeps for READMEs, shebangs, banner comments, spellings
- `diff WebGPUMultiGeo/WebGPUMultiGeo.py WebGPUMultiGeo/WebGPUMultiGeo_updated.py`
- `uv run pytest --collect-only -q` (354 tests collect cleanly)

## Headline findings

- `DefferedLighting/README.md` has SimpleWebGPU's title and body; `GUIDemos/NGLWidgetsOpenGL` duplicates its sibling's README and screenshot.
- Leaked agent-process text in GeometryTessellation, PBR/IBL and three stale "headless agent" screenshot TODOs.
- 21 per-folder READMEs never inline their existing screenshot; ~9 American spellings in prose; a few AI bold-bullet READMEs (SpatialHash3D worst).
- Template GL demos share copy-pasted boilerplate docstrings/comments (~20 files with the same trivial `__init__` docstring, ~24 with "# Clear the color and depth buffers").
- Docstring style split: 68 Google-style files vs 5 numpydoc (the stated house style) — needs a decision before sweeping.
- Stray files triaged: Obj2Numpy and Blending/tests look commit-ready; `WebGPUMultiGeo_updated.py`, `Particles/ParticleQuads/uv.lock`, `PBR/HDRIBaker/test.npz` look deletable; `.worktrees/readme-references` is an orphaned remote-session worktree with nothing unmerged; all 16 agent/* branches are merged.
