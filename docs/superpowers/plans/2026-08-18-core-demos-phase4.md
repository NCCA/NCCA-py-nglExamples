# Core Demos Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port NGL9Demos' "Texture/shader infrastructure" trio — `TexelFetch`, `LoadShaderFromJSon`, `TextureCompressor` — to PyNGLDemos. `TexelFetch` and `LoadShaderFromJSon` each get an OpenGL entry point (a faithful, direct port) and a WebGPU entry point (a *reinterpretation*: same teaching point, different mechanism, since neither TBOs nor JSON-driven `ShaderLib` loading exist on the WebGPU side). `TextureCompressor` ships OpenGL-only, as one focused demo distilled from the C++ original's four sub-tools.

**Architecture:** Three independent top-level demo folders (`TexelFetch/`, `LoadShaderFromJSon/`, `TextureCompressor/`) — none are a multi-sub-demo family, so none use the `Parent/<SubDemo>/` pattern; each is self-contained with its own `main.py` (+ `main_webgpu.py` where applicable), shaders, and `README.md`, matching the `ShadedGrid/`/`Spotlight/`/`MatrixStack/` precedent (single-topic folder, no shared cross-demo module needed — verified there is no maths worth factoring into a `sys.path.insert`-shared module for any of these three; TexelFetch's grid/noise, LoadShaderFromJSon's shader-assembly, and TextureCompressor's DXT1 codec are each single-demo concerns).

- **TexelFetch:** a 200×200 grid of `GL_POINTS`, each point's height read every frame from a buffer of `sin`/`cos` values via `texelFetch(samplerBuffer, gl_VertexID)` on OpenGL. WebGPU has no Texture Buffer Object or `texelFetch`; the reinterpretation reads the same per-vertex height data from a WGSL `var<storage, read> array<f32>` indexed by `@builtin(vertex_index)` — the same "feed raw per-vertex data to a shader outside the normal vertex-attribute path" lesson, WebGPU's native mechanism for it.
- **LoadShaderFromJSon:** a JSON manifest lists, per shader stage, an ordered list of GLSL files to concatenate into one compiled stage — a Phong-lit, 6-octave-simplex-noise-shaded teapot. `ncca.ngl`'s `ShaderLib` has no `load_from_json`; the OpenGL port parses the JSON in Python and drives `ShaderLib`'s existing low-level per-stage API directly (a pattern this repo already uses for tessellation in `GeometryTessellation/tess_main.py` and for a from-scratch program in `Lights/main.py` — same primitives, applied to JSON-declared stages instead of a fixed tuple). The WebGPU reinterpretation parses an equivalent WGSL-part JSON manifest and concatenates WGSL fragments into one `device.create_shader_module` source string — same "assemble one shader from several JSON-declared files" teaching point, applied to WebGPU's single-module-multi-entry-point shape.
- **TextureCompressor:** OpenGL-only (see Global Constraints and Task 5 for why). Ships a from-scratch DXT1 (S3TC) encoder/decoder pair and a viewer, distilled from the C++ original's four sub-tools (see Task 5's scope note).

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, numpy, `uv run --script`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md` (Phase 4 = roadmap rows 8–10: `TexelFetch`, `LoadShaderFromJSon`, `TextureCompressor`; grouping section "4. Texture/shader infrastructure").

## Global Constraints

- No edits to `/Users/jmacey/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder; where `ncca.ngl` has no matching helper (TBO creation, DXT-compressed texture upload), raw `OpenGL.GL` calls are used directly, matching this repo's established practice of not modifying the library for a one-demo need.
- Work happens in branch `agent/core-demos-phase4`, worktree at `.worktrees/core-demos-phase4`. This plan was authored without creating that worktree (per the authoring session's directive) — create it first: `git worktree add .worktrees/core-demos-phase4 -b agent/core-demos-phase4`.
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) using the `QTimer.singleShot(...)` pattern from `VAOPrimitives/main.py` (OpenGL) or `Blending/BlendingWebGPU.py`'s `main()` (WebGPU).
- OpenGL entry points: `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)`, calling `self.setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01, zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` in `__init__`. GL 4.1 core profile via the standard `QSurfaceFormat` block (copy from `VAOPrimitives/main.py`'s `__main__`).
- WebGPU entry points: `class WebGPUScene(WebGPUWidget)` importing `from ncca.ngl.webgpu import WebGPUWidget` directly. `self.msaa_sample_count = 4`, call `get_default_device()`, build pipelines/scene, then `self._create_render_buffer()`. Mouse/keyboard handlers hand-copied from `Blending/BlendingWebGPU.py` (no mixin for `QWidget`).
- Maths convention: numpy/PyNGL row-vector convention — points transform as `row_vec @ M`, translation lives in row 3. Matrix composition order matches the C++ source exactly, never reordered.
- **Smoketest verification command differs by backend, established repeatedly across Phases 1–3:** OpenGL entry points (`main.py`) must be verified WITHOUT `QT_QPA_PLATFORM=offscreen` on this machine (that mode segfaults for every `QOpenGLWindow` demo here — pre-existing environment limitation, not a code bug). WebGPU entry points (`main_webgpu.py`) must be verified WITH it.
- **`closeEvent`-stops-the-animation-timer requirement — applies to every task in this phase.** `TexelFetch` (both backends) and `LoadShaderFromJSon` (both backends) all animate via a repeating `QTimer` (20ms, matching the C++'s `startTimer(20)`), so all four MUST override `closeEvent` to call `<timer_attr>.stop()` before `super().closeEvent(event)` — the real, repeatedly-fixed crash from Phase 2 (`Spotlight`/`ShadedGrid`): a queued timer tick firing a GL/GPU call after window-close teardown. `TextureCompressor` (Task 5) has no repeating timer (static image viewer, redraws only on key/file-load) and must explicitly NOT have this override.
- **WebGPU per-draw uniform buffer pool** (the queue-timeline aliasing bug, found and fixed multiple times in Phases 1–3: `MatrixStack`, `LookAtDemos`, `SpherePlane`) does **not** apply to either WebGPU task in this phase — `TexelFetch` draws one `GL_POINTS`-equivalent batch per frame from one shared MVP uniform (no per-object draws), and `LoadShaderFromJSon` draws exactly one teapot per frame. Each task below states this explicitly so nobody adds unneeded pool machinery.
- `PrimData.sphere(radius, precision)` is a real pure-numpy sphere generator usable on WebGPU — not needed by any task in this phase (TexelFetch is a point grid, LoadShaderFromJSon uses the baked `teapot`), noted here only so nobody reaches for the wrong octahedron-substitute habit out of Phase 1–3 muscle memory.
- `ruff check` and `ruff format --check` must pass.
- README.md per demo folder (description, controls, teaching points, `![](<Demo>.png)` reference — screenshot itself expected missing, deferred to Jon).
- Root `README.md` gets one row per demo (`TexelFetch`, `LoadShaderFromJSon`, `TextureCompressor`), added in each demo's first task (matching the Phase 2/3 precedent of README rows landing inside feature commits).
- One commit per task.

---

## Task 1: TexelFetch (OpenGL)

**Files:**
- Create: `TexelFetch/main.py`
- Create: `TexelFetch/VertexShader.glsl`
- Create: `TexelFetch/FragmentShader.glsl`
- Create: `TexelFetch/README.md`
- Modify: `README.md` (root — add the TexelFetch row)

**Source:** `NGL9Demos/TexelFetch/src/NGLScene.cpp`, `include/NGLScene.h`, `shaders/{Vertex,Fragment}Shader.glsl`.

**Design notes:**
- Grid: `gridDimension = 10.0`, `gridStep = 0.1` → 200 samples per axis, nested loop `z` outer / `x` inner in `[-10, 10)`, giving exactly 40 000 points (`m_gridSize` in the C++). Reproduce the exact sample ordering with:
  ```python
  _GRID_DIM = 10.0
  _GRID_STEP = 0.1
  _GRID_N = int(round(2 * _GRID_DIM / _GRID_STEP))  # 200
  _COORDS = (np.arange(_GRID_N, dtype=np.float32) * _GRID_STEP) - _GRID_DIM
  ```
  `Z, X = np.meshgrid(_COORDS, _COORDS, indexing="ij")` — `Z` varies slowest (axis 0), `X` fastest (axis 1), matching the C++'s outer-`z`/inner-`x` loop when both are `.ravel()`'d (C order). The XZ vertex buffer and the Y texture-buffer data MUST use this identical ordering — `gl_VertexID` in the shader indexes both in lock-step.
- Y data (height): `y = sin(X + offset) + cos(X - offset)`, recomputed and re-uploaded every 20ms tick; `offset` starts at 0.0 and += 0.01 per tick (matches `updateTextureBuffer()`'s `static float offset`). The *initial* build in `buildTextureBuffer()` before the first tick uses `y = sin(X)` only (no `offset`/`cos` term yet) — reproduce that exact initial-vs-per-tick asymmetry, it's a one-frame cosmetic detail from the source, not worth "fixing".
- TBO: no `ncca.ngl` helper exists for Texture Buffer Objects (confirmed via grep — `Texture`/`ShaderLib` have nothing TBO-shaped) — use raw `OpenGL.GL` calls, matching this repo's precedent of dropping to raw GL when the library has no equivalent for a one-demo feature. `GL_TEXTURE_BUFFER`, `GL_R32F`, and `glTexBuffer` are all present in this environment's PyOpenGL (confirmed via direct import).
- Shader: single-file-per-stage, so the ordinary `ShaderLib.load_shader("TexelShader", "VertexShader.glsl", "FragmentShader.glsl")` is sufficient — no low-level JSON-style assembly needed here (that's Task 3).
- Camera: `look_at(Vec3(0,1,4), Vec3(0,0,0), Vec3(0,1,0))`. Projection: `perspective(45, w/h, 0.01, 150)` initially, `perspective(45, w/h, 0.05, 150)` in `resizeGL` (note: **not** 350 — that far-plane value belongs to `LoadShaderFromJSon`, don't cross-contaminate the two demos' numbers). Background clear colour `(0.4, 0.4, 0.4, 1.0)`.
- The C++ `initializeGL` calls `ngl::ShaderLib::use("nglColourShader")` + sets a `Colour` uniform before building `TexelShader` — this is dead code with zero visible effect (that shader/uniform is never used again; `paintGL` only ever uses `"TexelShader"`, whose fragment shader outputs a hardcoded `vec4(1.0)` regardless). Omit it; note the omission in a one-line comment so nobody "restores" it later thinking it was missed.
- Point rendering: `gl.glPointSize(4)` before the draw call, `gl.glDrawArrays(gl.GL_POINTS, 0, _GRID_N * _GRID_N)`.
- Keys: Escape/W (wireframe)/S (fill)/F (fullscreen)/N (windowed) — W/S are visually inert on `GL_POINTS` geometry in the source too (no triangles to outline), port them anyway for fidelity, matching the source's own vestigial behaviour.
- 20ms repeating `QTimer` → **needs `closeEvent`**.

- [ ] **Step 1: Write the shaders**

Create `TexelFetch/VertexShader.glsl` (verbatim port of the source, `#version 410 core` to match this repo's GL 4.1 core profile convention rather than the source's `410 core` — same version, no change needed):

```glsl
#version 410 core
layout(location=0) in vec2 xz;
uniform mat4 MVP;
uniform samplerBuffer yPosSampler;
void main()
{
  float ypos = texelFetch(yPosSampler, gl_VertexID).r;
  gl_Position = MVP * vec4(xz.x, ypos, xz.y, 1.0);
}
```

Create `TexelFetch/FragmentShader.glsl`:

```glsl
#version 410 core
layout(location=0) out vec4 fragColour;
void main()
{
  fragColour = vec4(1.0);
}
```

- [ ] **Step 2: Write main.py**

Create `TexelFetch/main.py` following this repo's standard `MainWindow(PySideEventHandlingMixin, QOpenGLWindow)` skeleton (copy the `__main__`/`QSurfaceFormat`/argparse block from `VAOPrimitives/main.py`). Key pieces:

```python
_GRID_DIM = 10.0
_GRID_STEP = 0.1
_GRID_N = int(round(2 * _GRID_DIM / _GRID_STEP))  # 200
_COORDS = (np.arange(_GRID_N, dtype=np.float32) * _GRID_STEP) - _GRID_DIM


def _build_xz() -> np.ndarray:
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    return np.stack([x.ravel(), z.ravel()], axis=1).astype(np.float32)


def _build_y(offset: float | None) -> np.ndarray:
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    if offset is None:
        return np.sin(x).ravel().astype(np.float32)
    return (np.sin(x + offset) + np.cos(x - offset)).ravel().astype(np.float32)
```

In `initializeGL`:
- `self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))`
- Build the XZ VAO (attribute 0, 2 floats, `GL_POINTS`); store `self.vertex_count = _GRID_N * _GRID_N`.
- Build the TBO:
  ```python
  y = _build_y(None)
  self.ypos_buffer = gl.glGenBuffers(1)
  gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.ypos_buffer)
  gl.glBufferData(gl.GL_TEXTURE_BUFFER, y.nbytes, y, gl.GL_STATIC_DRAW)
  self.tbo_texture = gl.glGenTextures(1)
  gl.glActiveTexture(gl.GL_TEXTURE0)
  gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.tbo_texture)
  gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R32F, self.ypos_buffer)
  ```
- `ShaderLib.load_shader("TexelShader", "VertexShader.glsl", "FragmentShader.glsl")`, `ShaderLib.use("TexelShader")`, `ShaderLib.set_uniform("yPosSampler", 0)`.
- `self.offset = 0.0`; `self.animation_timer = QTimer(self); self.animation_timer.timeout.connect(self._on_tick); self.animation_timer.start(20)`.

`_on_tick`: rebuild `y = _build_y(self.offset)`, re-upload (`glBindBuffer` + `glBufferData`, same call as init), `self.offset += 0.01`, `self.update()`.

`paintGL`: clear, `ShaderLib.use("TexelShader")`, compute `MVP = self.project @ self.view @ self.mouse_global_tx` (row-vector convention, same composition order as the C++'s `m_project * m_view * m_mouseGlobalTX`), `set_uniform("MVP", MVP)`, bind the TBO texture + VAO, `gl.glPointSize(4)`, `gl.glDrawArrays(gl.GL_POINTS, 0, self.vertex_count)`.

`resizeGL`: `self.project = perspective(45.0, w / h, 0.05, 150.0)`.

`closeEvent`: `self.animation_timer.stop(); super().closeEvent(event)`.

`keyPressEvent`: Escape/W/S/F/N as usual.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x TexelFetch/main.py
cd TexelFetch && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. Run WITHOUT `QT_QPA_PLATFORM=offscreen` (segfaults `QOpenGLWindow` on this machine).

- [ ] **Step 4: Write README.md and add the root README row**

Create `TexelFetch/README.md` — description ("demonstrates copying raw per-vertex data to a shader via a Texture Buffer Object and `texelFetch`"), controls, `![](TexelFetch.png)`.

Add a row to root `README.md` in an appropriate section (a "Shaders" or similar textures/shader-infrastructure section — check existing section headers first and place it with kindred demos, e.g. near `ShadedGrid`/`Spotlight` if a general OpenGL-techniques section exists, otherwise create a small new section), following the exact existing row format:
```markdown
| <a href="TexelFetch"><img src="TexelFetch/TexelFetch.png" width="220"></a> | [TexelFetch](TexelFetch) | Texture Buffer Object + texelFetch: animated height grid |
```

- [ ] **Step 5: Commit**

```bash
git add TexelFetch/main.py TexelFetch/VertexShader.glsl TexelFetch/FragmentShader.glsl TexelFetch/README.md README.md
git commit -m "feat(texel-fetch): add the OpenGL TBO/texelFetch demo"
```

---

## Task 2: TexelFetch (WebGPU)

**Files:**
- Create: `TexelFetch/main_webgpu.py`
- Create: `TexelFetch/TexelFetchShader.wgsl`

**Design notes — the reinterpretation:** WebGPU has no Texture Buffer Object and no `texelFetch`. The equivalent mechanism is a **read-only storage buffer** indexed by `@builtin(vertex_index)` in the vertex shader — same lesson (per-vertex data fed to the shader outside the normal vertex-attribute path, updated on the CPU every frame), WebGPU's native tool for it. This is a genuinely different mechanism from a texture sampler, which is exactly what the roadmap spec calls for ("WebGPU side: storage-buffer read, reinterpreted").

- Vertex buffer: XZ positions only (`_build_xz()` from Task 1, reused verbatim — same grid, same ordering).
- Storage buffer: Y heights (`_build_y()` from Task 1, reused verbatim), created once with `wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST`, updated per-tick via `queue.write_buffer(self.y_buffer, 0, y)` (cheaper than recreating the buffer, matches this repo's established per-frame-update pattern, e.g. `ShadedGrid/main_webgpu.py`).
- One shared uniform buffer holds `MVP` only (no per-draw pool needed — single draw call per frame).
- **Known, documented rendering difference:** WebGPU's `point-list` topology has no `gl_PointSize`/point-size-control equivalent — points always rasterize at 1 physical pixel, unlike the OpenGL sibling's `glPointSize(4)`. This is a genuine WebGPU platform limitation (not a simplification of the demo's data/mechanism), state it in the README's WebGPU section so it's not mistaken for a bug.
- 20ms repeating `QTimer` → **needs `closeEvent`**.

- [ ] **Step 1: Write the WGSL shader**

Create `TexelFetch/TexelFetchShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> y_buf: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) xz: vec2<f32>, @builtin(vertex_index) vidx: u32) -> VertexOutput {
    var out: VertexOutput;
    let ypos = y_buf[vidx];
    out.position = uniforms.mvp * vec4<f32>(xz.x, ypos, xz.y, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `TexelFetch/main_webgpu.py` following the standard `WebGPUScene(WebGPUWidget)` shape (mouse/keyboard handlers hand-copied from `Blending/BlendingWebGPU.py`; camera/perspective/grid-build logic shared with Task 1's `_build_xz`/`_build_y` — either duplicate the two small functions inline, since they're a few lines each and this pair of files has no established shared-module precedent for such small helpers, or `sys.path.insert`-import them from `main.py` if that reads cleaner; either is fine, pick one and be consistent).

- `_create_pipeline`: one render pipeline, `point-list` primitive topology, vertex buffer layout = 2×f32 at shader location 0 (xz), bind group layout = binding 0 uniform (`mvp`), binding 1 storage read-only (`y_buf`).
- `_create_scene`: build `xz` (Task 1's `_build_xz()`), `create_buffer_with_data(data=xz, usage=wgpu.BufferUsage.VERTEX)`; build initial `y = _build_y(None)`, `self.y_buffer = self.device.create_buffer_with_data(data=y, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)`; `self.uniform_buffer = self.device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)`; one bind group binding both.
- `_on_tick`: recompute `y = _build_y(self.offset)`, `self.device.queue.write_buffer(self.y_buffer, 0, y)`, `self.offset += 0.01`, `self.update()`.
- `paintWebGPU`: write `MVP` to `self.uniform_buffer` via `queue.write_buffer`, begin render pass, set pipeline/bind group/vertex buffer, `render_pass.draw(self.vertex_count)`.
- `animation_timer` at 20ms, same as Task 1.
- `closeEvent` stops `self.animation_timer`.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x TexelFetch/main_webgpu.py
cd TexelFetch && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `TexelFetch/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reinterprets the OpenGL Texture Buffer Object + `texelFetch`
mechanism: WebGPU has neither, so the per-vertex height values instead live
in a read-only storage buffer (`var<storage, read> array<f32>`), indexed by
`@builtin(vertex_index)` in the vertex shader — the same "raw per-vertex
data fed to the shader outside normal vertex attributes" lesson, WebGPU's
native tool for it. One platform limitation to note: WebGPU's `point-list`
topology has no point-size control, so points render at 1 pixel here versus
the OpenGL version's `glPointSize(4)`.
```

- [ ] **Step 5: Commit**

```bash
git add TexelFetch/main_webgpu.py TexelFetch/TexelFetchShader.wgsl TexelFetch/README.md
git commit -m "feat(texel-fetch): add the WebGPU storage-buffer reinterpretation"
```

---

## Task 3: LoadShaderFromJSon (OpenGL)

**Files:**
- Create: `LoadShaderFromJSon/main.py`
- Create: `LoadShaderFromJSon/shaders/shaders.json`
- Create: `LoadShaderFromJSon/shaders/version.glsl`
- Create: `LoadShaderFromJSon/shaders/common.glsl`
- Create: `LoadShaderFromJSon/shaders/noise3D.glsl`
- Create: `LoadShaderFromJSon/shaders/PhongVertex.glsl`
- Create: `LoadShaderFromJSon/shaders/PhongFragment.glsl`
- Create: `LoadShaderFromJSon/README.md`
- Modify: `README.md` (root — add the LoadShaderFromJSon row)

**Source:** `NGL9Demos/LoadShaderFromJSon/src/NGLScene.cpp`, `shaders/shaders.json`, `shaders/{version,common,noise3D,PhongVertex,PhongFragment}.glsl`.

**Design notes:**
- `ncca.ngl`'s `ShaderLib` has no `load_from_json` (confirmed by reading the full `shader_lib.py` — only `load_shader`, which takes exactly one file per stage). This repo already has precedent for driving `ShaderLib`'s low-level per-stage API directly when `load_shader`'s single-file-per-stage shape doesn't fit: `GeometryTessellation/tess_main.py`'s `load_tess_program()` (a fixed 4-stage tuple) and `Lights/main.py` (from-scratch two-stage build). This task's loader is the same pattern, generalized to read the stage list — and, per stage, the list of files to concatenate — from JSON instead of a hardcoded tuple.
- JSON schema (verbatim from the source, reproduce exactly — this is the actual teaching content of the demo):
  ```json
  {
    "ShaderProgram": {
      "name": "NoiseShader",
      "debug": false,
      "Shaders": [
        {"type": "Vertex", "name": "NoiseShaderVertex", "path": ["shaders/version.glsl", "shaders/common.glsl", "shaders/PhongVertex.glsl"]},
        {"type": "Fragment", "name": "NoiseShaderFragment", "path": ["shaders/version.glsl", "shaders/common.glsl", "shaders/noise3D.glsl", "shaders/PhongFragment.glsl"]}
      ]
    }
  }
  ```
- Loader (in `main.py`, or a small `load_shader_from_json.py` module alongside it — either is fine, this is a one-demo concern, no shared-module precedent needed):
  ```python
  import json

  _TYPE_TO_SHADERTYPE = {"Vertex": ShaderType.VERTEX, "Fragment": ShaderType.FRAGMENT}


  def load_shader_from_json(json_path: Path) -> str:
      data = json.loads(json_path.read_text())
      program = data["ShaderProgram"]
      program_name = program["name"]
      ShaderLib.create_shader_program(program_name)

      base_dir = (
          json_path.parent.parent
      )  # JSON's "shaders/..." paths are relative to the demo folder
      ok = True
      for stage in program["Shaders"]:
          shader_name = stage["name"]
          shader_type = _TYPE_TO_SHADERTYPE[stage["type"]]
          source = "\n".join((base_dir / p).read_text() for p in stage["path"])
          ShaderLib.attach_shader(shader_name, shader_type)
          ShaderLib.load_shader_source_from_string(shader_name, source)
          ok = ShaderLib.compile_shader(shader_name) and ok
          ShaderLib.attach_shader_to_program(program_name, shader_name)

      if not (ShaderLib.link_program_object(program_name) and ok):
          logger.error(
              f"Failed to build shader program {program_name!r} from {json_path}"
          )
      return program_name
  ```
  (`ShaderLib.load_shader_source_from_string` is confirmed present on both `_ShaderLib` and the underlying `Shader` class — this is exactly what makes the multi-file concatenation possible; `load_shader_source` only takes a single file path.)
- `common.glsl` declares the `Materials`/`Lights` GLSL structs and the `uniform Lights light; uniform Materials material;` (plus `uniform float time; uniform float repeat;`) — this is why the C++ (and this port) can use dotted uniform names like `"light.position"`/`"material.diffuse"`. This repo already sets dotted struct uniforms this way elsewhere (`SkinnedMeshImport/main.py`, `ShadedGrid/main.py`) — same convention, no adaptation needed.
- Camera: `look_at(Vec3(0,1,2), Vec3(0,0,0), Vec3(0,1,0))`. Projection: `perspective(45, w/h, 0.05, 350)` (this demo's far plane is 350 — do not confuse with `TexelFetch`'s 150). Background clear colour `(0.4, 0.4, 0.4, 1.0)`.
- Uniforms set once in `initializeGL` (verbatim from the source):
  - `light.position = (-2.0, 5.0, 2.0, 0.0)`, `light.ambient = (0,0,0,1)`, `light.diffuse = (1,1,1,1)`, `light.specular = (0.8,0.8,0.8,1)`
  - `material.ambient = (0.274725, 0.1995, 0.0745, 0.0)`, `material.diffuse = (0.75164, 0.60648, 0.22648, 0.0)`, `material.specular = (0.628281, 0.555802, 0.3666065, 0.0)`, `material.shininess = 51.2`
  - `viewerPos = (0.0, 1.0, 2.0)` (same as the camera `from`)
  - `time = 0.0`, `repeat = 0.01`
  - `Normalize` uniform is never set in the source (GLSL default `false`) — leave unset; the teapot's normals stay unit-length through the MV/normal-matrix transform anyway since there's no scale in the model transform, so this has no visible effect. Don't "fix" it by setting `Normalize = True`.
- **Faithfully-preserved source quirk:** `repeat`'s *shader* uniform starts at `0.01` (set in `initializeGL`), but the *key-adjustment* variable that `1`/`2` increment/decrement is a separate `static float repeat = 0.1f` local to `keyPressEvent` in the C++, decoupled from the init value. The first `1`/`2` press therefore jumps the shader's `repeat` uniform from `0.01` straight to `0.09`/`0.11` — a visible discontinuity in the noise UV scale. Port this exactly: track a `self.repeat = 0.1` instance attribute (matching the key-handler's static, not the init uniform) used only by the `1`/`2` handlers, separate from the `0.01` passed to `set_uniform` in `initializeGL`. This is a real, if minor, quirk in the original — preserve it rather than "fixing" the apparent mismatch, per this project's no-simplification-of-the-source policy.
- `time` increments by `0.01` every 20ms tick (matches `timerEvent`).
- Draw: `Primitives.load_default_primitives()` registers `"teapot"` (`Prims.TEAPOT`); `paintGL` computes `MV = self.view @ M`, `MVP = self.project @ MV`, `normal_matrix = inverse(transpose(MV))` (as a `Mat3`), uploads `MV`, `MVP`, `normalMatrix`, `M`, then `Primitives.draw("teapot")`.
- Keys: Escape/W/S/F/N plus `1`/`2` (repeat -/+ 0.01, clamped nowhere — matches the source, which has no clamp either).
- 20ms repeating `QTimer` → **needs `closeEvent`**.

- [ ] **Step 1: Copy the shader source files verbatim**

Create the five files under `LoadShaderFromJSon/shaders/` with the exact content from `NGL9Demos/LoadShaderFromJSon/shaders/{version,common,noise3D,PhongVertex,PhongFragment}.glsl` — this is the demo's actual teaching content (the JSON-driven concatenation, the struct-based light/material uniforms, the 6-octave simplex noise) and must not be altered. `version.glsl` is one line (`#version 410 core`); `common.glsl` declares `Materials`/`Lights` and the `light`/`material`/`time`/`repeat` uniforms; `noise3D.glsl` is Ashima Arts' `snoise(vec3)` (MIT-licensed, keep the header comment); `PhongVertex.glsl`/`PhongFragment.glsl` are the Phong lighting + noise-colormap stages.

- [ ] **Step 2: Write shaders.json**

Create `LoadShaderFromJSon/shaders/shaders.json` with the exact schema shown in the design notes above.

- [ ] **Step 3: Write main.py**

Create `LoadShaderFromJSon/main.py` with the `load_shader_from_json()` helper above, wired into the standard `MainWindow(PySideEventHandlingMixin, QOpenGLWindow)` skeleton with all the camera/uniform/timer/key details from the design notes. In `initializeGL`, call it and use the returned name for every subsequent `ShaderLib` call:

```python
program_name = load_shader_from_json(Path(__file__).parent / "shaders" / "shaders.json")
ShaderLib.use(program_name)
```

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x LoadShaderFromJSon/main.py
cd LoadShaderFromJSon && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback (a shader link failure would `exit()` before reaching the smoketest timer — a fast, loud way to catch a bad JSON path or concatenation order).

- [ ] **Step 5: Write README.md and add the root README row**

Create `LoadShaderFromJSon/README.md` — description ("shader program stages assembled at runtime from files listed in a JSON manifest; renders a Phong-lit teapot shaded with layered simplex noise"), controls, `![](LoadShaderFromJSon.png)`.

Add to root `README.md`, same section as `TexelFetch`:
```markdown
| <a href="LoadShaderFromJSon"><img src="LoadShaderFromJSon/LoadShaderFromJSon.png" width="220"></a> | [LoadShaderFromJSon](LoadShaderFromJSon) | JSON-driven shader assembly: noise-shaded teapot |
```

- [ ] **Step 6: Commit**

```bash
git add LoadShaderFromJSon/main.py LoadShaderFromJSon/shaders LoadShaderFromJSon/README.md README.md
git commit -m "feat(load-shader-from-json): add the OpenGL JSON-driven shader demo"
```

---

## Task 4: LoadShaderFromJSon (WebGPU)

**Files:**
- Create: `LoadShaderFromJSon/main_webgpu.py`
- Create: `LoadShaderFromJSon/shaders_webgpu/shaders.json`
- Create: `LoadShaderFromJSon/shaders_webgpu/common.wgsl`
- Create: `LoadShaderFromJSon/shaders_webgpu/noise3D.wgsl`
- Create: `LoadShaderFromJSon/shaders_webgpu/phong_noise.wgsl`

**Design notes — the reinterpretation:** wgpu-py pipelines aren't built from `ShaderLib`-style multi-file GLSL stages at all — a WGSL module is one string handed to `device.create_shader_module(code=...)`, typically containing both `@vertex`/`@fragment` entry points. There's no native "JSON shader descriptor" concept on this stack. The reinterpretation keeps the *same* lesson (a shader program assembled at runtime by concatenating several JSON-declared source files) applied to WGSL's shape: a small JSON manifest lists WGSL fragment files in order, Python concatenates them into one source string, and *that* string becomes the `create_shader_module` code — same mechanism as `LoadShaderFromJSon`'s OpenGL sibling (JSON → ordered file list → concatenation → compile), just with one manifest entry (one combined module) instead of two (separate vertex/fragment stages), because that's how WGSL modules are shaped.

- `shaders_webgpu/shaders.json`:
  ```json
  {
    "ShaderModule": {
      "name": "NoiseShader",
      "path": ["common.wgsl", "noise3D.wgsl", "phong_noise.wgsl"]
    }
  }
  ```
- Loader in `main_webgpu.py`:
  ```python
  def load_wgsl_from_json(json_path: Path, device) -> "wgpu.GPUShaderModule":
      data = json.loads(json_path.read_text())
      module = data["ShaderModule"]
      source = "\n".join((json_path.parent / p).read_text() for p in module["path"])
      return device.create_shader_module(code=source)
  ```
- `common.wgsl`: WGSL structs mirroring `common.glsl`'s `Materials`/`Lights`, plus the transform/lighting uniform structs (std140-style layout with explicit padding — follow `ShadedGrid/main_webgpu.py`'s `_TRANSFORM_DTYPE`/`_MATERIAL_DTYPE`/`_LIGHT_DTYPE` numpy-dtype pattern for the Python-side matching layout, and its WGSL struct-padding convention, since that file already solved "GLSL light/material struct → WGSL uniform struct" for this exact material/light shape in this repo):
  ```wgsl
  struct Material {
      ambient: vec4<f32>,
      diffuse: vec4<f32>,
      specular: vec4<f32>,
      shininess: f32,
      _pad0: vec3<f32>,
  };
  struct Light {
      position: vec3<f32>,
      _pad0: f32,
      ambient: vec4<f32>,
      diffuse: vec4<f32>,
      specular: vec4<f32>,
  };
  struct Transform {
      m: mat4x4<f32>,
      mvp: mat4x4<f32>,
      normal_matrix: mat4x4<f32>,
      viewer_pos: vec3<f32>,
      _pad0: f32,
  };
  struct Params {
      time: f32,
      repeat: f32,
      _pad0: vec2<f32>,
  };

  @group(0) @binding(0) var<uniform> transform: Transform;
  @group(0) @binding(1) var<uniform> material: Material;
  @group(0) @binding(2) var<uniform> light: Light;
  @group(0) @binding(3) var<uniform> params: Params;
  ```
- `noise3D.wgsl`: a mechanical WGSL translation of `noise3D.glsl`'s `snoise(vec3<f32>) -> f32` (Ashima Arts simplex noise) — same algorithm, WGSL syntax (`fn`/`vec3<f32>`/`vec4<f32>`, explicit `floor`/`step`/`abs` calls which all exist in WGSL with the same names). This is a direct, well-defined port (no design judgement calls, just syntax translation of a ~70-line well-known public-domain function) — translate it faithfully rather than reimplementing noise differently; keep the same corner/gradient logic so the noise pattern matches the OpenGL sibling's visually.
- `phong_noise.wgsl`: vertex stage computes `fragment_normal`/`eye_direction`/`v_position`/`light_dir`/`half_vector`/`uv` exactly as `PhongVertex.glsl` does (using `transform`'s `m`/`mvp`/`normal_matrix`, `light.position`, `params.repeat`); fragment stage ports `pointLight()` and the 6-octave noise-layering `main()` from `PhongFragment.glsl` unchanged (same octave weights `1, 0.5, 0.25, 0.125, 0.0625, 0.03125`, same `time`-offset pattern per octave, same `vec3(1.0, 0.5, 0.0)` hot colormap base).
- Camera/projection: identical to the OpenGL sibling (`eye=(0,1,2)`, `to=(0,0,0)`, `up=(0,1,0)`, `perspective(45, w/h, 0.05, 350, PerspMode.WebGPU)`).
- Mesh: `PrimData.primitive(Prims.TEAPOT.value)` — the baked teapot mesh, matching this repo's established WebGPU mesh-loading pattern.
- One shared uniform buffer set per struct (`transform`, `material`, `light`, `params`) — no per-draw pool needed (one teapot, one draw call per frame).
- `params.time` incremented `+0.01` per 20ms tick, written via `queue.write_buffer`; `params.repeat` starts at `0.1` and is nudged `±0.01` by keys `1`/`2` — **this WebGPU sibling does NOT need to reproduce the OpenGL port's `0.01`-vs-`0.1` initial-value quirk** (that quirk is an artifact of the C++'s two separate places setting the value; the WebGPU reinterpretation has one clean place to initialize it, so just start `repeat` at `0.1` and don't manufacture a discontinuity that has no equivalent WebGPU-side cause). Document this one intentional smoothing in the README's WebGPU section.
- 20ms repeating `QTimer` → **needs `closeEvent`**.

- [ ] **Step 1: Write the WGSL shader parts and JSON manifest**

Create `shaders_webgpu/shaders.json`, `common.wgsl`, `noise3D.wgsl`, `phong_noise.wgsl` per the design notes.

- [ ] **Step 2: Write main_webgpu.py**

Create `LoadShaderFromJSon/main_webgpu.py` with `load_wgsl_from_json()`, the standard `WebGPUScene(WebGPUWidget)` shape, the four uniform buffers, the teapot mesh/vertex buffer, `1`/`2` key handling for `repeat`, `animation_timer` at 20ms for `time`, and `closeEvent`.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x LoadShaderFromJSon/main_webgpu.py
cd LoadShaderFromJSon && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback (a WGSL syntax error in the concatenated `noise3D.wgsl`/`phong_noise.wgsl` port would surface here as a `create_shader_module` failure — check the exact wgpu-py error output carefully if this fails, since it's the one step in this phase most likely to need a syntax-translation fix).

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `LoadShaderFromJSon/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reinterprets the JSON-driven shader assembly for WGSL's
shape: wgpu-py has no native "load a shader program from JSON" path (a
WGSL module is one source string with both vertex and fragment entry
points, unlike GLSL's separate compiled stages), so `shaders_webgpu/shaders.json`
lists WGSL fragment files that get concatenated into one module source
before `device.create_shader_module` — same "assemble a shader from
JSON-declared files" lesson, applied to WebGPU's single-module shape. One
intentional smoothing: the OpenGL version's `repeat` uniform has a visible
jump on the first `1`/`2` keypress, an artifact of the C++ source setting
its initial value in two disconnected places; this version starts `repeat`
at `0.1` cleanly since there's no equivalent second initialization site here.
```

- [ ] **Step 5: Commit**

```bash
git add LoadShaderFromJSon/main_webgpu.py LoadShaderFromJSon/shaders_webgpu LoadShaderFromJSon/README.md
git commit -m "feat(load-shader-from-json): add the WebGPU JSON-driven WGSL-assembly reinterpretation"
```

---

## Task 5: TextureCompressor (OpenGL only)

**Files:**
- Create: `TextureCompressor/dxt_texture.py`
- Create: `TextureCompressor/compress_texture.py`
- Create: `TextureCompressor/main.py`
- Create: `TextureCompressor/TextureVertex.glsl`
- Create: `TextureCompressor/TextureFragment.glsl`
- Create: `TextureCompressor/textures/source.png` (generated by Step 1, then committed as a real file)
- Create: `TextureCompressor/textures/texture.cmptx` (generated by Step 3, then committed as a real binary asset)
- Create: `TextureCompressor/README.md`
- Modify: `README.md` (root — add the TextureCompressor row)

**Source:** `NGL9Demos/TextureCompressor/{Compressor,PackTool,DXTViewer,PackViewer}/` (four sub-tools), `TextureCompressor/README.md`.

**Why OpenGL-only:** DXT/S3TC compressed-texture formats (`GL_COMPRESSED_RGBA_S3TC_DXT{1,3,5}_EXT`, `glCompressedTexImage2D`) are a desktop-GL feature with no WebGPU equivalent in this stack — wgpu-py exposes GPU texture-compression formats (BC1–BC7) only via pre-compressed data upload on backends/adapters that support them, which is a different, less teachable surface, and matches this phase's roadmap entry ("GL-only ... DXT/S3TC desktop-GL feature", same reasoning as Phase 2's `ShadedGrid`). Confirmed via the local environment: `OpenGL.GL.EXT.texture_compression_s3tc` exposes the three DXT internal-format constants and `OpenGL.GL.glCompressedTexImage2D` is present and callable with the standard 8-argument signature (`target, level, internalformat, width, height, border, imageSize, data`) — no PyOpenGL gap blocks this.

**Scope decision — which sub-tool(s) become the Python demo:** the C++ source is actually four separate programs: `Compressor` (CLI, image → single `.cmptx` file via SDL2_image + the `squish` library), `PackTool` (CLI, multiple images → one `.pack` file), `DXTViewer` (GL viewer, one `.cmptx` file on a screen quad), `PackViewer` (GL viewer + a reusable `TexturePack` class, arrow-keys through a `.pack` file's contents). This repo's demos are interactive viewers, not CLI utilities, and per Jon's standing rule (`Use the same folder just give each a unique demo name` / prior phases) a single focused demo beats reproducing four separate tools. **Decision: port `DXTViewer`, not `PackViewer`.** `DXTViewer`'s single-file format is self-contained and simpler to reproduce faithfully (`PackViewer`'s `.pack`/`TexturePack` machinery is a second, separate file-format and class layered on top, whose main teaching value — multi-texture-per-material lookup — is arguably closer to the *already-shipped* `PBR/PBRTexture/texture_pack.py` + `texture_pack_parser.py`'s JSON-driven texture-pack pattern than to a new compression demo; adding it here would mostly duplicate that, not teach DXT compression). Squish and SDL2_image aren't available from Python, so `Compressor`'s CLI is also reinterpreted rather than ported 1:1: `compress_texture.py` is a from-scratch numpy DXT1 encoder (see Step 2) writing the *same* `ngl::cmptx` binary layout the C++ `DXTTexture::load()` reads — same file format, Python-native encoder in place of linking `libsquish`. **DXT3/DXT5 (the source's alpha-carrying variants) are out of scope** — this demo only ships DXT1 (opaque, 4-colour blocks), since the teaching point (block-based endpoint/palette compression) doesn't need all three variants and a from-scratch DXT3/DXT5 encoder would roughly double this task's size for no proportional teaching gain; document this scope cut in the README.

**File format (verbatim from the C++, reproduce exactly — read/write must round-trip with `DXTTexture::load()`'s layout):**
```
10 bytes  ASCII "ngl::cmptx" (no trailing NUL)
int32 LE  width
int32 LE  height
uint32 LE internalformat (GL_COMPRESSED_RGBA_S3TC_DXT1_EXT's numeric value)
int32 LE  compression enum (DXT1=0, DXT3=1, DXT5=2 — this demo only ever writes 0)
uint32 LE size (byte length of the compressed data that follows)
<size> bytes raw DXT1-compressed block data
```

**DXT1 encoder — from scratch, since no Python DXT/S3TC library is available in this environment.** A simple, correct (not optimal — no cluster-fit like `libsquish`) per-block principal-axis encoder is sufficient for a teaching demo: pick the two most extreme colours along the block's dominant colour axis as the two 565-packed endpoints, derive the standard 4-colour DXT1 palette, assign each of the 16 texels to its nearest palette colour. This produces spec-correct DXT1 data any GPU decodes properly, with visible (and pedagogically useful — that's the point of the demo) block-compression artefacts on hard edges.

- [ ] **Step 1: Write `dxt_texture.py`**

Create `TextureCompressor/dxt_texture.py` — the encoder, the `.cmptx` reader/writer, and a synthetic test-pattern generator (used by Step 3 to avoid depending on any binary asset from another demo folder — keeps `TextureCompressor/` fully self-contained, per this repo's per-demo-folder convention):

```python
"""DXT1 (S3TC) block compression and the ngl::cmptx file format.

A from-scratch encoder/decoder pair -- no Python S3TC/squish library is
available in this environment, and the C++ original's `squish` dependency
isn't portable here. This trades libsquish's cluster-fit quality for a
simple principal-axis endpoint choice: still spec-correct DXT1 data (any
GPU decodes it normally), with more visible block artefacts on hard edges
than a production encoder -- which is fine, even useful, for a demo whose
whole point is to make block compression visible.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from OpenGL.GL.EXT.texture_compression_s3tc import GL_COMPRESSED_RGBA_S3TC_DXT1_EXT

_MAGIC = b"ngl::cmptx"
_DXT1 = 0


def _pack_rgb565(rgb: np.ndarray) -> int:
    r, g, b = (int(c) for c in rgb)
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _unpack_rgb565(value: int) -> np.ndarray:
    r = (value >> 11) & 0x1F
    g = (value >> 5) & 0x3F
    b = value & 0x1F
    return np.array([r << 3, g << 2, b << 3], dtype=np.float32)


def _compress_block(block: np.ndarray) -> bytes:
    """block: (16, 3) uint8 RGB texels of one 4x4 block, row-major."""
    pixels = block.astype(np.float32)
    mean = pixels.mean(axis=0)
    centred = pixels - mean
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centred @ axis
    c0 = np.clip(pixels[np.argmax(proj)], 0, 255).astype(np.uint8)
    c1 = np.clip(pixels[np.argmin(proj)], 0, 255).astype(np.uint8)

    r0 = _pack_rgb565(c0)
    r1 = _pack_rgb565(c1)
    if r0 < r1:
        r0, r1 = r1, r0

    e0 = _unpack_rgb565(r0)
    e1 = _unpack_rgb565(r1)
    palette = np.stack([e0, e1, (2 * e0 + e1) / 3.0, (e0 + 2 * e1) / 3.0])
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    indices = dists.argmin(axis=1).astype(np.uint32)

    index_bits = 0
    for i, idx in enumerate(indices):
        index_bits |= int(idx) << (2 * i)

    return struct.pack("<HHI", r0, r1, index_bits)


def compress_dxt1(rgb: np.ndarray) -> bytes:
    """rgb: (height, width, 3) uint8. height and width must be multiples of 4."""
    height, width, _ = rgb.shape
    if height % 4 or width % 4:
        raise ValueError("DXT1 compression requires dimensions that are multiples of 4")
    blocks = bytearray()
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            block = rgb[by : by + 4, bx : bx + 4].reshape(16, 3)
            blocks += _compress_block(block)
    return bytes(blocks)


def write_cmptx(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    data = compress_dxt1(rgb)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<ii", width, height))
        f.write(struct.pack("<I", int(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT)))
        f.write(struct.pack("<i", _DXT1))
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def read_cmptx(path: Path) -> tuple[int, int, int, bytes]:
    """Returns (width, height, internal_format, data)."""
    with open(path, "rb") as f:
        magic = f.read(10)
        if magic != _MAGIC:
            raise ValueError(f"{path} is not an ngl::cmptx file")
        width, height = struct.unpack("<ii", f.read(8))
        (internal_format,) = struct.unpack("<I", f.read(4))
        f.read(
            4
        )  # compression enum -- unused on read, DXT1 is the only variant this demo writes
        (size,) = struct.unpack("<I", f.read(4))
        data = f.read(size)
    return width, height, internal_format, data


def make_test_pattern(size: int = 256) -> np.ndarray:
    """A synthetic RGB test image with sharp edges and gradients -- deliberately
    a mix even DXT1's coarse 4-colour-per-block palette will visibly struggle
    with in places, so the compression artefacts this demo exists to show are
    actually visible. Self-generated so this folder has no dependency on any
    binary asset from another demo folder.
    """
    y, x = np.mgrid[0:size, 0:size]
    checker = (((x // 16) + (y // 16)) % 2) * 255
    gradient_r = (x * 255 // size).astype(np.uint8)
    gradient_g = (y * 255 // size).astype(np.uint8)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(checker > 0, gradient_r, 255 - gradient_r)
    rgb[..., 1] = gradient_g
    rgb[..., 2] = checker.astype(np.uint8)
    return rgb
```

- [ ] **Step 2: Write `compress_texture.py`**

Create `TextureCompressor/compress_texture.py` — a small CLI mirroring the C++ `Compressor` tool's spirit (image file(s) in, `.cmptx` file(s) out), built on `dxt_texture.py` and Pillow (already a transitive dependency via `ncca.ngl.Image` — confirm the import path used elsewhere in this repo, e.g. `PIL.Image` or `ncca.ngl.image.Image`, and use whichever this repo already depends on for reading arbitrary PNG/JPG files into a numpy RGB array; pad non-multiple-of-4 dimensions by cropping down to the nearest multiple of 4, matching a common, simple DXT1 constraint rather than adding padding logic the C++ didn't have either):

```python
#!/usr/bin/env -S uv run --script
"""Compress an image to the ngl::cmptx DXT1 format used by main.py's viewer.

A from-scratch reinterpretation of the NGL9Demos Compressor CLI tool: same
job (image in, compressed .cmptx file out) and the same output file format,
without linking libsquish (unavailable from Python here) -- see
dxt_texture.py for the encoder.

Usage:
    ./compress_texture.py input.png [-o output.cmptx]
"""

import argparse
from pathlib import Path

import numpy as np
from dxt_texture import write_cmptx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    rgb = np.array(img)
    h, w, _ = rgb.shape
    h4, w4 = h - (h % 4), w - (w % 4)
    rgb = rgb[:h4, :w4]

    output = args.output or args.input.with_suffix(".cmptx")
    write_cmptx(output, rgb)
    print(f"wrote {output} ({w4}x{h4}, DXT1, {output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

Run it once to generate the bundled sample asset (do this locally, then commit both files as real assets):
```bash
cd TextureCompressor
python3 -c "from dxt_texture import make_test_pattern; from PIL import Image; Image.fromarray(make_test_pattern()).save('textures/source.png')"
python3 compress_texture.py textures/source.png -o textures/texture.cmptx
cd ..
```

- [ ] **Step 3: Write the texture shader**

Create `TextureCompressor/TextureVertex.glsl` and `TextureCompressor/TextureFragment.glsl` — a minimal NDC-space textured-quad pair (verbatim shape from `DXTViewer/src/ScreenQuad.cpp`'s shaders: position passthrough in `[-1,1]`, UV interpolated), **without** the source's `.bgr` swizzle in the fragment shader (`fragColour.rgb = texture(tex, vertUV).bgr` in the C++ — that swizzle exists because the C++ `squish` library's compressed output stores channels in BGR order; this from-scratch Python encoder in Step 1 compresses straight RGB, so read `.rgb` here, not `.bgr` — using `.bgr` against RGB-order data would swap the red/blue channels for no reason. Note this deviation with a one-line comment in the fragment shader.):

```glsl
#version 410 core
layout (location = 0) in vec3 inVert;
layout (location = 1) in vec2 inUV;
out vec2 vertUV;
void main()
{
    gl_Position = vec4(inVert, 1.0);
    vertUV = inUV;
}
```

```glsl
#version 410 core
layout (location = 0) out vec4 fragColour;
uniform sampler2D tex;
in vec2 vertUV;
// Note: no .bgr swizzle here (unlike the C++ source) -- this demo's DXT1
// encoder (dxt_texture.py) compresses straight RGB, not squish's BGR
// output, so a plain .rgb read is correct.
void main()
{
    fragColour = texture(tex, vertUV);
}
```

- [ ] **Step 4: Write main.py**

Create `TextureCompressor/main.py` — the `DXTViewer` port: `MainWindow(PySideEventHandlingMixin, QOpenGLWindow)`, no repeating timer (static viewer, matches the source having none — **do not add a `closeEvent` override**, per this phase's Global Constraints).

- `initializeGL`: build a screen-space NDC quad VAO (two triangles, position+UV, same layout as `SimpleTexture`'s or `DXTViewer`'s `ScreenQuad`), `ShaderLib.load_shader("Texture", "TextureVertex.glsl", "TextureFragment.glsl")`, `ShaderLib.set_uniform("tex", 0)`, then load the default `textures/texture.cmptx` via:
  ```python
  width, height, internal_format, data = read_cmptx(
      Path(__file__).parent / "textures" / "texture.cmptx"
  )
  self.tex_id = gl.glGenTextures(1)
  gl.glActiveTexture(gl.GL_TEXTURE0)
  gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
  gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
  gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
  gl.glCompressedTexImage2D(
      gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, len(data), data
  )
  ```
  (`glCompressedTexImage2D` needs the explicit `imageSize` argument in this PyOpenGL build — confirmed via `OpenGL.GL.VERSION.GL_1_3`'s wrapper, which only annotates `data`'s array size as unchecked, it does not auto-fill `imageSize` the way `glTexImage2D` auto-fills from `format`/`type`.)
- `paintGL`: clear, bind texture + VAO, draw 6 vertices as `GL_TRIANGLES` — no MVP needed (NDC-space screen quad, same as the source's `ScreenQuad`).
- `keyPressEvent`: Escape/F/N as usual, plus `O` — open a `QFileDialog.getOpenFileName` for a `.cmptx` file, `read_cmptx()` it, delete the old GL texture (`glDeleteTextures`) and re-upload, matching `DXTViewer::reload()`. Also support an optional `--file` CLI argument (argparse) that pre-loads a specific `.cmptx` at startup instead of the bundled default — a reasonable, non-simplifying addition matching the C++ main's "first command-line argument" behaviour (`NGLScene(_fname)`).
- No `closeEvent` override (no repeating timer).

- [ ] **Step 5: Make executable and smoke-test**

```bash
chmod +x TextureCompressor/main.py TextureCompressor/compress_texture.py
cd TextureCompressor && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. Run WITHOUT `QT_QPA_PLATFORM=offscreen`.

Also verify the round-trip independently of the GUI (catches an encoder/format bug without needing a live GL context):
```bash
cd TextureCompressor
python3 -c "
from pathlib import Path
from dxt_texture import read_cmptx
w, h, fmt, data = read_cmptx(Path('textures/texture.cmptx'))
assert w == 256 and h == 256, (w, h)
assert len(data) == w * h // 2, len(data)  # DXT1 = 0.5 bytes/pixel
print('round-trip OK', w, h, fmt, len(data))
"
cd ..
```

- [ ] **Step 6: Write README.md and add the root README row**

Create `TextureCompressor/README.md` covering: what DXT1/S3TC block compression is and why it exists (fixed 4 bytes/block regardless of content → predictable GPU memory footprint + can be sampled compressed, unlike JPEG); the `ngl::cmptx` file format; that this port ships `compress_texture.py` (a from-scratch Python DXT1 encoder replacing the C++ original's `libsquish` dependency) + `main.py` (the viewer, ported from `DXTViewer`); the scope cut (DXT1 only, no DXT3/DXT5 alpha variants; `PackViewer`'s multi-texture `.pack` format not ported, since its teaching value overlaps the already-shipped `PBR/PBRTexture` texture-pack demo); controls (`O` load a new `.cmptx` file, Escape/F/N); `![](TextureCompressor.png)`.

Add to root `README.md`:
```markdown
| <a href="TextureCompressor"><img src="TextureCompressor/TextureCompressor.png" width="220"></a> | [TextureCompressor](TextureCompressor) | DXT1/S3TC compressed-texture viewer + a from-scratch Python encoder |
```

- [ ] **Step 7: Commit**

```bash
git add TextureCompressor/dxt_texture.py TextureCompressor/compress_texture.py TextureCompressor/main.py TextureCompressor/TextureVertex.glsl TextureCompressor/TextureFragment.glsl TextureCompressor/textures/source.png TextureCompressor/textures/texture.cmptx TextureCompressor/README.md README.md
git commit -m "feat(texture-compressor): add the OpenGL DXT1 viewer and encoder"
```

---

## Final steps (after all 5 tasks)

- [ ] **Run full verification**

```bash
uv run ruff check TexelFetch LoadShaderFromJSon TextureCompressor
uv run ruff format --check TexelFetch LoadShaderFromJSon TextureCompressor
uv run pytest
```
Expected: ruff clean; pytest passes (no new test files in this phase — none of the three demos have a pure-maths core substantial enough to warrant one, unlike `Collisions`/`ViewToWorldTransform`; the DXT1 round-trip check in Task 5 Step 5 is a manual verification, not a pytest suite, since it needs a real image round-trip rather than isolated pure-function assertions — note this in the final report rather than silently skipping test coverage).

- [ ] **Confirm all 3 root README rows are present and correctly formatted**

Each demo task above adds its own row inline; this step is just a final visual check that all 3 rows exist, in a sensible section, with no duplicates, and that the WebGPU-note sections were appended to the right READMEs (`TexelFetch`, `LoadShaderFromJSon` — not `TextureCompressor`, which has no WebGPU sibling).

- [ ] **Report to Jon**

List the 3 `.png` screenshots that still need capturing (`TexelFetch.png`, `LoadShaderFromJSon.png`, `TextureCompressor.png`) under their respective folders. Flag the known risk areas for a human/reviewer to double-check interactively, since no automated smoketest exercises them:
- `TexelFetch`'s TBO re-upload every 20ms (does the wave actually animate, not just draw once) — both backends.
- `LoadShaderFromJSon`'s noise shading (does the teapot visibly shift/ripple over time, does `1`/`2` visibly change the noise UV scale, including the intentional first-press jump on the OpenGL side) — both backends, and specifically the hand-translated `noise3D.wgsl` port on the WebGPU side, the one piece of this phase most likely to have a subtle syntax or corner-case bug that still compiles but looks visually wrong.
- `TextureCompressor`'s DXT1 encoder quality (visible blockiness is expected and correct — it's not cluster-fit — but confirm colours are right, i.e. no red/blue channel swap from the removed `.bgr` swizzle) and the `O` file-open dialog with a hand-picked external `.dds`/`.cmptx`-shaped test file if Jon wants to stress it beyond the bundled sample.
