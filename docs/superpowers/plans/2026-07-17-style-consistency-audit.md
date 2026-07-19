# Writing-style and consistency audit — what needs fixing

Date: 2026-07-17. Three sweeps were run over the whole repo: every README.md (84 files) against the [jon-writing-style guidelines], every Python file's comments and docstrings, and a consistency / stray-file audit. This plan lists what came out of it, split into work an agent can do unattended and things Jon needs to decide or do himself. Nothing has been changed yet.

The short version: the recent maths-first demos (MassSpring, SkeletalAnimation, SceneGraph, Billboards, GimbalLock and friends) are in good shape and can serve as the style reference. Almost all the problems are in the older template-derived GL demos (copy-pasted boilerplate comments, Google-style docstrings, American spellings) and in a handful of AI-generated READMEs that leaked process notes or the wrong content entirely.

## Part 1 — outright errors (agent, do first)

These are wrong, not just off-style.

1. `DefferedLighting/README.md` — the title and body are a verbatim copy of `SimpleWebGPU/README.md` ("# SimpleWebGPU Example ... generate a new pipeline per shader type"). Only the References section is about deferred shading. Rewrite the title and body to describe the G-buffer / lighting-pass demo. Note TODO.md says this demo is "not working at all", so the README should be honest about its state.
2. `GUIDemos/NGLWidgetsOpenGL/README.md` — title and body duplicate `GUIDemos/PySideGUIOpenGL/README.md`, and its screenshot `PySideGUI.png` is a copy of the sibling demo's. Rewrite to describe the NGL-widgets variant. (New screenshot needs Jon — see Part 5.)
3. `VertexArrayObject/MultiBufferVAO/README.md` — title is `# Boid`; should be MultiBufferVAO.
4. `ShowMipmap/README.md` — lines 5–12 are raw OpenGL reference text pasted verbatim, line 14 is the unfinished sentence "To set the levels." Rewrite as a short demo note; title `# ShowMipMap` should match the folder name.
5. Typos: `MassSpring/README.md` ("pixelsm" line 63, garbled clause line 16, trailing comma in heading line 46), `FontRendering/README.md:31` ("With and optional"), `VertexArrayObject/BoidShaded/README.md:7` ("Normnals"), `FBODemos/SimpleFBO/README.md:5` ("texuture"), `VAOPrimitives/README.md` ("build in" ×2, "bunny buddah"), `SimplePyNGL/README.md:3` ("These examples is", "the build in shaders"), `NormalMapping/README.md:4` (C++ `ngl::Obj` should be `ngl.Obj`, and missing blank line after the image).

## Part 2 — de-AI the READMEs (agent)

1. Remove leaked agent-process passages:
   - `GeometryTessellation/README.md` lines 61–91 — the whole "ShaderLib API check (read this before writing similar demos)" section ("Per the task brief...", "No library edit was needed or made").
   - `PBR/IBL/README.md` — line 4 "(screenshot TODO — cannot be captured headlessly)" (IBL.png exists), lines 27–30 the task-brief commentary ("the six new demos in the batch", "the brief explicitly allows..."), and lines 78–85 the "Sandbox note" about the agent's Qt platform plugin.
   - Stale "headless agent can't capture" screenshot TODOs in `MarchingCubes/README.md:5-6`, `SceneGraph/README.md:5`, `Billboards/README.md:5` — the images all exist and are referenced.
2. Rewrite `WebGPUCompute/SpatialHash3D/README.md` (153 lines) — the worst AI-template offender: "showcasing", "comprehensive controls", a Features section of bold-bullet pairs, American spellings (Visualization, Colors, colored, neighboring), bare `python` run commands, and unrequested "Performance Considerations" / "Extensions" sections. Cut to a short note in the house style; `uv run` commands; reference `SpatialHash3D.png`.
3. Flatten the bold-nested bullet hierarchies in `WebGPUMultiGeo/README.md` ("How It Works" section) and `2DDrawingOpenGL/README.md` ("User Controls") into short prose or plain lists, and rewrite `WebGPUShadows/README.md` (thin AI stub, "This has two pipeline...") to match its OpenGL sibling `ShadowMapping/README.md`.
4. British spellings in prose (leave code identifiers alone): `ShadowMapping/README.md:7` artifact→artefact, `Particles/ParticleQuads/README.md:7` vectorized, `PBR/PBRTexture/README.md:32` grayscale, `Interpolation/README.md:19` popularized, `ColourObj/README.md:6` modeling, `PointCloud/README.md:6` unitizes, `ColourSelectionOpenGL/README.md:9,11` color/colors, `2DDrawingOpenGL/README.md:12` Initialization, plus the SpatialHash3D ones above.
5. Add the missing inline screenshot (`![](Foo.png)`) to the 21 per-folder READMEs whose image exists but is never shown: WebGPUComputePicking, WebGPUMultiGeo, SpatialHash3D, SpatialHash2D, SimpleComputeWebGPU, Voxels, ShowMipmap, VAOPrimitives, WebGPUShadows, SimpleWebGPU, SimpleTexture, DefferedLighting, SimplePyNGL, and all eight VertexArrayObject sub-demos.
6. Add a fenced `uv run <path>` block where the run instructions are missing or use bare `python` / `./file.py`: SpatialHash3D, SpatialHash2D, WebGPUMultiGeo, WebGPUShadows, 2DDrawingOpenGL and any others found in passing.
7. Promote `##` document titles to `#` in: ColourSelectionOpenGL, GUIDemos/QMLOverlayApp, GUIDemos/QMLWebGPUOverlay, GUIDemos/PySideGUIOpenGL, GUIDemos/NGLWidgetsOpenGL, BlankWebGPU, ShadingModels, SimpleTexture.

## Part 3 — comment and docstring cleanup (agent, after the Part 6 decisions)

The template-derived GL demos share copy-pasted boilerplate; these are mechanical, repo-wide find-and-fix passes.

1. Delete the trivial-constructor docstring "Initializes the main window and sets up default scene parameters." from the ~20 `main.py` files that carry it (Lights, BlankPySide6NGL, KleinBottle, ColourObj, PBR/SimplePBR, PBR/HDRI, ImageHeightMap, CurveDemos, FrustumCull, VAOPrimitives, Interpolation, Voxels, SimplePyNGL/PySideSimpleNGL.py, OpenGLPrimRestart/FastPyNGLVersion.py, ...), plus "Initializes the FrameBufferObject." in `Voxels/FrameBufferObject.py:55` and `PostProcessChain/FrameBufferObject.py:55`, and `RunDemos.py:58`.
2. Delete the copy-pasted what-narration comments across the same template family (~12–24 files each): "# Create a QSurfaceFormat object...", "# Create the main window", "# Update the projection matrix to match the new aspect ratio.", "# Call the base class implementation...", "# Set up the camera's view matrix.", "# Clear the color and depth buffers from the previous frame" (~24 files, also American), the trailing "# Exit the application" / "# Switch to wireframe rendering" / view/projection trailing comments, and the "It's crucial to update the viewport..." filler in `Voxels/main.py:216` and `FrustumCull/main.py:216`.
3. `FrustumCull/main.py` — the one file where missing type hints are a clear pattern: the Qt event handlers (lines 219–380) and the App class (406/410) have no annotations. Add them.
4. Docstring convention sweep (pending decision D1): 68 files use Google-style `Args:`/`Returns:`; only 5 use the numpydoc style the house guidelines specify. If D1 says numpydoc, convert — it's mechanical but touches most of the repo, so it should be its own branch/PR. Style collisions worth fixing regardless of D1: `MarchingCubes/marching_cubes.py` vs its sibling `mc_tables.py`, and `PBR/HDRI/main.py` vs `exr_loader.py`.
5. Leave alone: the shader `// ------` banner comments (they match the NCCA C++ standard), the wheel-event "120 = one standard wheel notch" comments (legitimate why-comments), and the exemplary files — `MassSpring/mass_spring.py`, `SkeletalAnimation/skinning_maths.py`, `SceneGraph/scene_graph.py`, `Billboards/billboard_maths.py`, `RayMarchingSDF/sdf_maths.py`, `EasingFunctions/easing.py`, `GimbalLock/rotation_maths.py` — which are the reference for how the rest should read.

## Part 4 — housekeeping the stray files (agent, per decisions below)

State of the untracked files, checked individually:

| Path | What it is | Suggested action |
| :--- | :--- | :--- |
| `Obj2Numpy/` | Complete demo (executable .py, README, png), already linked from the root README | Commit |
| `Blending/tests/test_blend_sort.py` | Finished headless test, collects under pytest | Commit |
| `docs/superpowers/plans/2026-07-11-*.md`, `specs/2026-07-11-*.md` | Planning docs in the established pattern | Commit |
| `WebGPUMultiGeo/WebGPUMultiGeo_updated.py` | Stale older iteration despite the name — imports a local `WebGPUWidget` module that doesn't exist in the folder, no type hints | Delete (also the stray `err` file in that folder) |
| `Particles/ParticleQuads/uv.lock` | Stray per-folder lockfile; the repo uses one root env | Delete |
| `PBR/HDRIBaker/test.npz` | 35 MB test bake | Delete |
| `2DDrawingOpenGL/Compute.wgsl` | 2D particle compute shader referenced by nothing in the folder | Decision D3 |
| `PBR/HDRIBaker/BBridge.npz`, `TableMountain.npz` | 32 MB bakes each | Decision D2 |
| `SimplePyNGL/WithQuat.py` | Work in progress, not mentioned by the README | Decision D4 |
| `Notebooks/Canvas.py`, `WebGPUTriangle.html` | Scratch experiments | Decision D5 |

Also: `.worktrees/readme-references/` is an orphaned worktree from a remote session (its `.git` file points at a dead `/sessions/...` path; no matching branch exists and its READMEs are identical to the working tree). Nothing unmerged in it — `git worktree prune` and delete the directory. All 16 `agent/*` branches are merged into Version1.0 and can be deleted. There is also a broken ref `refs/stale.readme-references.lock.old` to clean up.

## Part 5 — things only Jon can do

1. Screenshots that need a real GL run: a genuine `NGLWidgetsOpenGL` screenshot (it currently shows the sibling demo's), and a sanity pass over previews after any README rewrites.
2. `DefferedLighting` is a misspelling of DeferredLighting — renaming the folder means touching the root README, RunDemos discovery and git history, so it's a judgement call whether it's worth it. The demo is also on TODO.md as broken, so fixing the demo itself is a separate (larger) task.
3. TODO.md grooming — several entries look stale against the current tree (worth a five-minute pass).
4. Trim review of the two long teaching READMEs, `GeometryTessellation` (209 lines) and `UBOStorageBuffers` (197) — cutting teaching content is judgement, not mechanics. An agent can draft the trim for approval if wanted.

## Part 6 — decisions needed before the agent passes run

- **D1 — docstring convention.** The house style is numpydoc; 68 files consistently use Google-style. Convert the lot (one mechanical PR, my recommendation since the style guide is explicit), or accept Google as the de-facto house style and only hold new code to numpydoc?
- **D2 — .npz bake policy.** `ibl_maps.npz` (9.4 MB) is already tracked; `BBridge`/`TableMountain` (32 MB each) are not. Commit them, git-ignore `*.npz` and treat bakes as generated artefacts, or move to git-lfs?
- **D3 — `2DDrawingOpenGL/Compute.wgsl`.** Nothing references it; it looks like it belongs to a WebGPU particle demo. Delete, or move to wherever it was meant to live?
- **D4 — `SimplePyNGL/WithQuat.py`.** Finish and document it, or delete?
- **D5 — `Notebooks/`.** The only top-level folder with no README, no screenshot and no root-README entry. Give it a one-line README saying it's scratch space, or leave it out of the catalogue deliberately?
- **D6 — the `# ----` section-banner comments** in SelectionManipulator, Instancing and RayPickingSelection. They echo the NCCA C++ banner style so my recommendation is to leave them; say the word if they should go.

## Suggested execution order

1. Jon answers D1–D6 (five minutes).
2. Agent branch 1: Parts 1 + 2 (README fixes) — all prose, no code risk.
3. Agent branch 2: Part 3 (comments/docstrings) — run `uv run pytest` and `uv run ruff format --check .` after; the changes are comment-only so tests must stay green.
4. Agent branch 3: Part 4 housekeeping (commits + deletions per decisions).
5. Jon: Part 5 items at leisure.

Each branch as a worktree per the repo rules, conventional commits, one PR each so the diffs stay reviewable.

## Verification

- `uv run pytest` — 354 tests currently collect and should still pass untouched (Parts 2–3 touch no logic).
- `uv run ruff check .` and `uv run ruff format --check .` clean.
- `uv run RunDemos.py` — launcher still discovers every demo and shows the rewritten READMEs/previews.
- Re-grep the tree for the AI tells: `grep -rniE "comprehensive|showcasing|seamless|task brief|headless agent|sandbox" --include="*.md"` should come back empty (bar legitimate uses), and the American-spelling greps likewise in prose.

[jon-writing-style guidelines]: the summary is: first person, British English, short plain READMEs (title, what/why, how to run with uv, links, inline screenshot), numpydoc docstrings, sparse why-comments, no marketing adjectives or bold-bullet hierarchies.
