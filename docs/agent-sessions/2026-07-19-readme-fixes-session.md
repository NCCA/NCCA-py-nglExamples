# Agent session — 2026-07-19 — README fixes (style audit branch 1)

## Goal

Execute Parts 1 and 2 of `docs/superpowers/plans/2026-07-17-style-consistency-audit.md` — the prose-only README branch: fix the outright errors (copied/duplicated content, wrong titles, typos) and de-AI the worst READMEs (leaked agent process notes, template features lists, American spellings, missing screenshots and run commands, `##` document titles).

## Branch

`agent/style-audit-readmes`, worktree `.worktrees/style-audit-readmes`, two commits:

1. `docs(readme): fix wrong, duplicated or garbled content and remove leaked agent notes`
2. `docs(readme): style sweeps — British spellings, inline screenshots, uv run, title levels`

## Files changed

44 READMEs. Full rewrites: DefferedLighting, GUIDemos/NGLWidgetsOpenGL, ShowMipmap, WebGPUCompute/SpatialHash3D, WebGPUShadows, WebGPUMultiGeo. Section removals: GeometryTessellation (ShaderLib API check), PBR/IBL (screenshot TODO, task-brief commentary, sandbox note), MarchingCubes / SceneGraph / Billboards (stale screenshot TODOs). The rest are typo, spelling, screenshot, run-command and title-level fixes per the plan.

Found in passing and fixed: ShadowMapping/README.md claimed WebGPUShadows does a manual non-comparison depth read and shares PCF/artefact toggles — its shader actually uses `textureSampleCompare` and its only keys are `1`/`Esc`.

## Commands run

- `uv run pytest -q` — 349 passed (baseline and after; the plan's 354 includes untracked test files not on this branch)
- `uv run ruff format --check .` — clean
- `uv run ruff check .` — 47 pre-existing errors, identical on the base commit, all in Python files this branch does not touch (branch 2's remit)
- AI-tell grep (`comprehensive|showcasing|seamless|task brief|headless agent|sandbox` over `*.md`) — clean apart from a factual description of Pomax's Bézier primer and a project-brief title
