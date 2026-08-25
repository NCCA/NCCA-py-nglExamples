# 2026-07-15 — WebGPU Render To Texture rebuild

## Goal

Fix `FBODemos/WebGPURenderToTexture/`, which rendered the teapot into the top-left
corner of the window and never actually rendered to a texture. Make it work like the
OpenGL `FBODemos/SimpleFBO/` demo: render the teapot to a texture in a first pass, then
apply that texture to a new scene (plane + sphere) in a second pass.

## Root cause

The demo did not do render-to-texture at all. `TeapotPipeline.paint()` drew the teapot
straight into the widget's full-size colour buffer through a fixed `set_viewport(0, 0,
512, 512, ...)` call, so on a retina Mac (buffer = window x device-pixel-ratio) the
teapot landed in a small patch in one corner. The projection aspect was also taken from
the window, not the target. The README was a leftover copy of BlankWebGPU's.

## Changes

Worktree `agent/webgpu-rtt`, all under `FBODemos/WebGPURenderToTexture/`:

- `TeapotPipeline.py` — now owns its own 1024x1024 offscreen render target (multisampled
  colour resolved to a single-sample texture, plus a multisampled depth buffer). `paint()`
  takes no external views and renders into those targets; the resolved texture is exposed
  as `texture_view`.
- `ScenePipeline.py` (new) — second-pass pipeline drawing a `PrimData.triangle_plane` and
  `PrimData.sphere`, both textured with the teapot `texture_view`, one MVP uniform + bind
  group per object.
- `SceneShader.wgsl` (new) — MVP + `texture_2d` + sampler textured shader.
- `main.py` — two-pass `paintWebGPU` (teapot -> offscreen texture, then plane + sphere ->
  widget buffer at full window size), separate square projection for the teapot vs
  window-aspect projection for the scene, `resizeWebGPU` updates the scene projection, and
  standard LMB-rotate / RMB-pan / wheel-zoom mouse controls added.
- `README.md` — rewritten to describe the two-pass demo.
- `WebGPURenderToTexture.png` — new screenshot of the working demo.

Root `README.md` already linked the demo with the matching screenshot path, so no change
there.

## Commands run

```bash
uv run FBODemos/WebGPURenderToTexture/main.py --smoketest 500   # SMOKETEST OK
uv run ruff check --select I --fix FBODemos/WebGPURenderToTexture/
uv run ruff format FBODemos/WebGPURenderToTexture/
uv run ruff check FBODemos/WebGPURenderToTexture/               # All checks passed
uv run pytest -q                                               # 270 passed
```

Visual verification was done by grabbing a rendered frame: the teapot render (teal
background + gold teapot) is applied as a texture to the plane and sphere and fills the
window with the correct aspect ratio, matching SimpleFBO.
