# UG2 Demos Implementation Plan — Instancing, SceneGraph, StencilOutline, Billboards

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Read first:** `docs/superpowers/specs/2026-07-11-new-teaching-demos-design.md` — all
skeletons, maths conventions, WebGPU widget rules, and the definition of done live there
and are NOT repeated per task. Reference implementations: `Blending/`, `OITransparency/`.

**Goal:** Four 2nd-year fundamentals demos. Audience: students who have seen VAOs,
basic shaders and the MVP pipeline but nothing else.

---

## Task 1: Instancing (OpenGL + WebGPU)

The single biggest gap in the repo. Show that one draw call can render thousands of
objects, and *why* that is faster than a Python loop of draw calls.

**Files:**
- `Instancing/main.py` (GL), `Instancing/InstancingWebGPU.py`, `Instancing/WebGPUWidget.py` (copy)
- `Instancing/instance_layout.py` — numpy-only: generates per-instance data
- `Instancing/shaders/InstanceVertex.glsl`, `InstanceFragment.glsl`, `Instancing/InstanceShader.wgsl`
- `Instancing/tests/test_instance_layout.py`, `Instancing/README.md`

**Scene:** N cubes (own numpy cube geometry, pos+normal interleaved — reuse/adapt the
`quad()` generator style) arranged by `instance_layout.golden_spiral(n, radius)` and/or
`grid(n)` returning a float32 array of per-instance records: offset vec3, uniform scale
float, colour vec4 (8 floats — deliberately NOT a mat4 so one attribute slot suffices).
Whole field slowly rotates via a `rotation` uniform + `startTimer(16)`.

**Controls:** `I` toggle instanced vs naive loop; `+`/`-` double/halve N (default 4096,
clamp 1..65536); HUD shows mode, N, and rolling frame-time in ms (time `paintGL` with
`time.perf_counter`, average over 30 frames) — the frame time IS the teaching point.

**GL specifics:**
- One VBO for cube vertices (locations 0,1), second VBO for instance data with
  `glVertexAttribPointer` at locations 3 (offset+scale as vec4) and 4 (colour vec4) and
  `glVertexAttribDivisor(loc, 1)`.
- Draw: `glDrawArraysInstanced(GL_TRIANGLES, 0, 36, n)`. Naive mode: loop n times
  setting an `offsetScale`/`Colour` uniform and drawing 36 verts (same shader with an
  `instanced` bool uniform, or a second tiny shader — prefer the bool for one code path).
- Vertex shader: `pos = inVert * instScale + instOffset` then MVP. Do NOT pass a mat4
  attribute (uses 4 locations, needs 4 divisor calls — mention in README as an aside).

**WebGPU specifics:**
- Second vertex buffer with `"step_mode": "instance"` and shader locations 3/4;
  `render_pass.draw(36, n)`. Naive mode: n small uniform-buffer bind groups is slow to
  build — instead draw with `draw(36, 1, 0, i)` per instance using
  `@builtin(instance_index)`… which still reads the instance buffer, so the honest naive
  comparison is a loop of `draw(36, 1, first_instance=i)` calls. Verify `first_instance`
  is honoured by wgpu-py; if not, fall back to n separate draws with a dynamic uniform
  offset, and note whichever route was taken in the README.

**Tests:** layout functions return the right shape/dtype, spiral radii monotonic,
grid centred on origin, colours in [0,1].

**Pitfalls:** attribute divisor left set pollutes later VAOs (bind VAO before setting);
frame-time must be measured over whole frames, not just the draw call, or the naive
Python-loop overhead (the actual lesson) is missed.

- [ ] instance_layout.py + tests green
- [ ] GL demo: both modes render identically at N=4096, HUD timing works
- [ ] WebGPU demo: both modes, matching visuals
- [ ] README (explain divisor, step_mode, and why mat4-per-instance is avoided), root README row, smoketests, ruff, commit

---

## Task 2: SceneGraph (OpenGL)

Transform hierarchies: the gap between "single MVP" demos and everything else.

**Files:**
- `SceneGraph/main.py`, `SceneGraph/scene_graph.py` (numpy/ncca-maths only, no GL),
  `SceneGraph/tests/test_scene_graph.py`, `SceneGraph/README.md`
- Shaders: none — use `DefaultShader.DIFFUSE` with per-node `Colour`.

**scene_graph.py:** a minimal `Node`:
```python
class Node:
    def __init__(self, name, local=Mat4(), mesh=None, colour=(1,1,1)):  # mesh = Primitives name
        self.children: list[Node] = []
    def add(self, child) -> Node: ...
    def world_matrix(self, parent_world=Mat4()) -> Mat4:   # parent_world @ self.local
    def walk(self, parent_world=Mat4()):                    # yields (node, world) depth-first
```
Composition rule (row-vector repo convention, see spec): child world =
`parent_world @ local`; a node's local = `T @ R` composed explicitly with Mat4 calls.
Keep it dumb and readable — this file is lecture material.

**Scene:** a robot arm: base (cylinder or scaled cube) → upper arm → lower arm → two
claw fingers, each node a scaled `Primitives` cube/sphere. Joint angles stored per node
name in a dict; rebuilding `local` each frame from the current angles.

**Controls:** `1..5` select joint (HUD shows selection), `Left/Right` rotate selected
joint ±5°, `P` play a canned waving animation (sin-driven angles via `startTimer`),
`Space` reset pose.

**Tests:** two-level translation composes (base at x=1, child local x=2 → world x=3);
rotation of parent swings child position as expected (90° about y maps child offset
(0,0,2) to (2,0,0) within fp tolerance); walk order and parent_world propagation.

**Pitfalls:** the `@`-order convention — write the test FIRST against
`Blending/blend_scene.view_space_z` style numpy checks; scale in a parent node also
scales child translations (feature, mention in README).

- [ ] scene_graph.py + tests green
- [ ] GL demo with joint selection, animation, HUD
- [ ] README (matrix-stack story, DFS traversal), root README row (new "Transforms & Hierarchy" section or Geometry section), smoketest, ruff, commit

---

## Task 3: StencilOutline (OpenGL)

The depth/stencil buffer made visible: Maya-style selection outlines.

**Files:**
- `StencilOutline/main.py`, `StencilOutline/shaders/` (reuse Blending-style vert/frag pair
  plus the same vertex with a `uniform float outlineScale` fattening along the normal),
  `StencilOutline/README.md`. No tests folder (no pure maths).

**Setup:** `format.setStencilBufferSize(8)` in the `QSurfaceFormat` block — without this
the whole demo silently does nothing (put a loud comment there).

**Scene:** teapot + 2–3 spheres/cubes on a grid; click-free selection via `Tab` cycling
which object is "selected" (keep it keyboard-only; picking has its own demos).

**Render loop:**
1. Clear colour/depth/stencil (`glClear` with `GL_STENCIL_BUFFER_BIT`).
2. Draw all objects normally, but for the selected one:
   `glEnable(GL_STENCIL_TEST); glStencilFunc(GL_ALWAYS, 1, 0xFF);
   glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE); glStencilMask(0xFF)`.
3. Outline pass for the selected object: `glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
   glStencilMask(0x00); glDisable(GL_DEPTH_TEST)`; draw it again with
   `outlineScale ≈ 0.03` (vertex: `inVert + inNormal * outlineScale`) and flat orange
   `DefaultShader.COLOUR`-style output; restore state.

**Controls:** `Tab` cycle selection, `O` toggle outline pass (shows the two-pass cost),
`V` visualise the stencil buffer (extra mode: draw a fullscreen tint where stencil==1 —
implement by drawing a screen-size quad with stencil func EQUAL), `+`/`-` outline width.

**Pitfalls:** stencil mask must be re-enabled (0xFF) before the next frame's clear or
the clear won't touch stencil; scaling along normals splits hard-edged cubes (mention in
README — teapot demonstrates clean, cube demonstrates the artefact, both on purpose).

- [ ] Demo with outline + stencil-visualise modes
- [ ] README (stencil func/op state machine table), root README row, smoketest, ruff, commit

---

## Task 4: Billboards (OpenGL)

Camera-facing quads: the technique behind particles, sprites, impostors and HUD markers.

**Files:**
- `Billboards/main.py`, `Billboards/billboard_maths.py`, `Billboards/tests/test_billboard_maths.py`,
  `Billboards/shaders/BillboardVertex.glsl` + frag, `Billboards/README.md`
- Texture: generate a soft radial-gradient RGBA sprite in numpy at startup and upload
  with `glTexImage2D` (no binary asset needed).

**billboard_maths.py (numpy-only, tested):**
```python
def spherical_basis(view: np.ndarray) -> tuple[right, up]   # rows 0..2 of view^T trick
def cylindrical_basis(view: np.ndarray) -> tuple[right, up] # up locked to world +y
```
Right/up extracted from the model-view rotation (transpose of the upper 3x3 in the
row-vector convention — derive carefully and pin with tests: billboard normal must point
at the camera for a set of random view matrices).

**Scene:** grid + teapot (depth reference) + ~30 textured billboards scattered at random
positions (fixed seed). Vertex shader receives per-draw `centre`, `size`, and
`right`/`up` vectors as uniforms; expands a unit quad in the shader OR (simpler, do
this) the CPU rebuilds a small dynamic VBO per frame — N is tiny, and it keeps the
shader trivial for 2nd years. State the choice in the README.

**Modes (`M` cycles, HUD shows):** 1 fixed world-space quads (break when orbiting),
2 cylindrical (trees — tilt breaks them), 3 spherical (always face camera).
`B` toggles alpha blending of the sprite texture (ties back to the Blending demo:
sorted back-to-front using `blend_scene.back_to_front`-style helper from
`billboard_maths`).

**Tests:** spherical basis orthonormal and camera-facing for random azimuth/elevation
views; cylindrical up is exactly world +y; degenerate case (looking straight down)
doesn't produce NaNs for cylindrical mode (document the expected fallback).

- [ ] billboard_maths.py + tests green
- [ ] GL demo, 3 modes + blend toggle, procedural sprite texture
- [ ] README (basis derivation, when to use each mode), root README row, smoketest, ruff, commit
