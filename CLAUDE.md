# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A collection of ~70 standalone teaching demos for **PyNGL** (`ncca.ngl`, NCCA's Python graphics library — [source](https://github.com/NCCA/PyNGL), [PyPI](https://pypi.org/project/ncca-ngl/), [docs](https://ncca.github.io/PyNGL/)). Each top-level folder is an independent demo covering one OpenGL, WebGPU, or maths topic (cameras, VAOs, picking, shadows, compute shaders, particles, etc.) — there is no shared application code between demos, only the shared `ncca.ngl` dependency.

`RunDemos.py` is a PySide6 GUI launcher that scans the repo for executable demo scripts and lets you browse/run them with a README + preview image per demo.

## Commands

This project uses **uv** exclusively.

```bash
uv sync                          # install/update the environment (also runs automatically via .envrc/direnv)
uv run RunDemos.py               # launch the demo browser GUI
uv run <path/to/demo.py>         # run a single demo directly, e.g. uv run Camera/main.py
uv run pytest                    # run all tests
```

Most demo scripts are also directly executable (`chmod +x`, shebang `#!/usr/bin/env -S uv run --script`), so `./Camera/main.py` works too.

Tests are sparse and live per-demo in a `tests/` subfolder when present (e.g. `RayPickingSelection/tests/`) — they are headless unit tests for pure-Python maths (ray/triangle intersection, screen-space picking, etc.), not GUI/GPU tests.

## Architecture

### Demo folder structure

Each demo folder is self-contained. New demos are discovered automatically by `RunDemos.py` — there is no registration step, just drop a folder in the root with an executable `.py` file.

### Two rendering backends

- **OpenGL demos**: use `PySide6.QtOpenGL.QOpenGLWindow` (or PySDL3 in a few cases) + `PyOpenGL` (`OpenGL.GL`), driven through `ncca.ngl.opengl` (`ShaderLib`, `Primitives`, `DefaultShader`). Math types (`Mat3`, `Mat4`, `Vec3`, `Quaternion`, `look_at`, `perspective`) come from `ncca.ngl` directly.
- **WebGPU demos**: built on `wgpu-py`, using a custom `WebGPUWidget` (PySide6 widget wrapping a wgpu surface) plus per-demo "Pipeline" classes (e.g. `TeapotPipeline`, `FloorPipeline`) that each own a render/compute pipeline and its `.wgsl` shader. There is a `PipelineFactory`-style pattern in newer demos; see the `pyngl-webgpu` skill for details.

### Shared dependency: `ncca.ngl`

The actual library code is **not** in this repo — it's an external editable dependency pinned in `pyproject.toml` under `[tool.uv.sources]` to a local path (`/Users/jmacey/teaching/Code/PyNGL`). When a demo needs a library-level fix rather than a demo-level fix, the change belongs in that separate PyNGL repo, not here.

### `[tool.uv.workspace]`

`members=["*"]` / `exclude=["*"]` — every top-level folder is technically a workspace member but excluded, meaning each demo runs with `uv run` against the single root environment rather than per-folder venvs.

## Working conventions

- When editing PyNGL demo code (Qt/OpenGL, ShaderLib, VAOs, Primitives, Obj meshes, Vec/Mat/Quaternion math), consult the **pyngl** skill; for the WebGPU stack specifically, consult the **pyngl-webgpu** skill.
- Keep changes scoped to a single demo folder unless explicitly asked to change the shared launcher (`RunDemos.py`) or repo-wide config.
- `NGLDebug.log` files and `__pycache__` dirs appear inside demo folders at runtime — these are git-ignored, not part of the demo source.

Ensure a screen shot of the demo running is included in folder along with the README, and update the root README.md file with a link to the demo's folder
