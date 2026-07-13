# UBO / Storage Buffers

![](UBOStorageBuffers.png)
<!-- TODO(Jon): capture a screenshot of each entry script and save as
     UBOStorageBuffers.png (a shot of main.py's teapot + checker grid with
     the HUD visible is the priority; a second shot of StorageWebGPU.py
     with the light-count HUD line visible would also be useful but only
     one image is wired into this README/the root README row). -->

Explicit, buffer-backed uniforms, and the `std140` padding rule almost
everyone gets wrong at least once. Two entry points sharing one folder:

- **`main.py`** (OpenGL) — one `SceneBlock` UBO (`mat4 VP; vec4 lightPos;
  vec4 lightColour;`) bound once, at binding point 0, and read by **two
  different shader programs** — a diffuse teapot and a flat-coloured
  checker grid — via `glUniformBlockBinding` + `glBindBufferBase`. A
  second UBO, `MaterialBlock { vec3 albedo; vec3 specularColour; float
  shininess; }`, carries a deliberate `std140` padding trap, and the HUD
  shows the driver's own `GL_UNIFORM_OFFSET` answers as ground truth.
- **`StorageWebGPU.py`** (WebGPU) — the same scene, plus a
  `var<storage, read>` **runtime-sized** array of point lights (8..64,
  grown/shrunk with `+`/`-`) accumulated in the fragment shader: the one
  thing the OpenGL half of this demo cannot do, because GL 4.1 has no
  shader storage buffer objects.

## Controls

| Key | main.py (OpenGL) | StorageWebGPU.py (WebGPU) |
| :-- | :-- | :-- |
| `X` | toggle `MaterialBlock` CPU layout: std140-correct / naive | same |
| `+` / `-` | — | grow / shrink the storage-buffer light count (8..64) |
| LMB / RMB / wheel | rotate / pan / zoom | same |
| `Space` | reset camera | same |
| `Esc` | quit | same |

## One UBO, two shader programs: the core lesson

A uniform **location** (`glGetUniformLocation`) is per-variable and
per-program — that is why every demo elsewhere in this repo calls
`ShaderLib.set_uniform("MVP", ...)` again for every shader it uses. A
uniform **block binding** is a level of indirection *above* that:

```
GLSL block "SceneBlock" (program A)  --\
                                         >--  binding point 0  ---  one GL_UNIFORM_BUFFER
GLSL block "SceneBlock" (program B)  --/
```

1. `glGetUniformBlockIndex(program, "SceneBlock")` finds the block's index
   *within that program* (every program numbers its own blocks
   independently).
2. `glUniformBlockBinding(program, blockIndex, bindingPoint)` tells that
   program "read this block from context binding point N" — this is what
   `main.py::_bind_scene_block_to_both_programs` calls once per program,
   at startup, for **both** the teapot and grid programs, pointing them at
   the same binding point (0).
3. `glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, buffer)` fills that
   binding point with an actual buffer object.

After that one-time wiring, **one** `glBufferSubData` call per frame
(`main.py::_update_scene_block`) updates the VP matrix and light for both
programs — no per-shader re-upload, no `ShaderLib.use()` required first.
This demo deliberately does **not** use `ShaderLib.set_uniform_buffer()`:
that convenience method allocates a fresh `GL_UNIFORM_BUFFER` *per shader
program* (see `ShaderProgram._register_uniform_blocks` in `ncca.ngl`),
which would give the teapot and grid programs two different buffers and
defeat the entire point of this demo.

**Pitfall:** `glBufferSubData` needs raw `bytes`, not a numpy structured
array directly — always call `.tobytes()` first (`layouts.py`'s dtypes are
uploaded this way throughout both `main.py` and `StorageWebGPU.py`).

## The `std140` padding trap

`layouts.py` defines the CPU-side numpy dtypes for `SceneBlock` and two
competing layouts for `MaterialBlock`:

| Block | Field | "Correct" std140 offset | "Naive" packed offset |
| :-- | :-- | :-: | :-: |
| `SceneBlock` | `VP` (mat4) | 0 | — |
| `SceneBlock` | `lightPos` (vec4) | 64 | — |
| `SceneBlock` | `lightColour` (vec4) | 80 | — |
| `MaterialBlock` | `albedo` (vec3) | 0 | 0 |
| `MaterialBlock` | `specularColour` (vec3) | **16** | **12** |
| `MaterialBlock` | `shininess` (float) | **28** | **24** |

A `vec3`'s base alignment in `std140` (and in WGSL's default uniform
address-space layout) is **16 bytes**, but it only *consumes* 12 — the
rule has two halves and people usually get one of them wrong:

1. A vec3 does **not** pad itself: a following member whose own alignment
   is ≤ 4 (a lone `float`) packs straight into the leftover slot. That is
   why `shininess` sits at 28, immediately after `specularColour`'s 12
   bytes — not pushed to 32.
2. What a vec3 *does* do is push the **next 16-byte-aligned member** up:
   `specularColour` cannot start at 12, so the compiler moves it to 16 and
   bytes 12..15 become padding.

A programmer who assumes a plain, tightly packed C struct places
`specularColour` at 12 and `shininess` at 24. The compiler that built the
actual shader disagrees, and reads from 16 and 28.

Both entry scripts build the `MaterialBlock` upload buffer from
`layouts.py::MATERIAL_BLOCK_STD140_DTYPE` (the correct one) by default.
Pressing `X` switches to `layouts.py::naive_bytes_padded_to_std140()`,
which writes the tightly packed bytes instead. The shader, compiled once
and never recompiled, keeps reading from offsets 16 and 28 regardless of
what was uploaded, so the visible corruption is deterministic:
`specularColour` reads back as `(specular.g, specular.b, shininess)` — the
CPU-side shininess value (64.0) lands in the blue channel — and
`shininess` reads the zero padding at 28. **The teapot's tight warm
highlight smears into a blue-white glare across the whole lit side**
(exponent clamps to 1, glare tinted by the scrambled colour), while
`albedo` (offset 0 in both layouts) looks completely unaffected — which is
exactly why this bug is so easy to half-miss in real code. Nothing about
the GLSL/WGSL source changes between the two states — only the bytes fed
to an already-compiled shader.

The GL demo does not just assert these offsets in comments: at startup
`main.py::_query_material_offsets` asks the driver where the linked
program actually put each member (`glGetUniformIndices` +
`glGetActiveUniformsiv(..., GL_UNIFORM_OFFSET, ...)`) and prints/renders
the answer on the HUD — `albedo@0 specularColour@16 shininess@28` — so the
demo verifies its own layout claims at runtime. (WebGPU has no equivalent
runtime reflection in `wgpu-py`; the WGSL rules are the spec's "Memory
Layout" table, identical to `std140` for these types.)

`tests/test_layouts.py` checks the dtypes' `itemsize`s and field offsets
against hand-computed `std140` values, and that a naive payload read at
the shader's real offsets comes back scrambled exactly as described.

## Why GL stops here: no SSBOs on GL 4.1

Shader storage buffer objects (`GL_SHADER_STORAGE_BUFFER`,
`buffer` interface blocks) are a **GL 4.3** feature. macOS's OpenGL driver
tops out at **4.1 core** (see `CLAUDE.md` / every other demo in this
repo), so a runtime-sized array of point lights simply cannot be expressed
on the OpenGL side — a UBO's arrays must have a fixed maximum size baked
into the GLSL source at compile time (`vec3 lights[MAX_LIGHTS];`), which
is a completely different, much less flexible, feature. This is where
`main.py`'s story ends and `StorageWebGPU.py`'s begins.

## The WebGPU half: a runtime-sized storage buffer

`StorageWebGPU.py` declares:

```wgsl
@group(0) @binding(2) var<storage, read> lights: array<PointLight>;
```

with no length in the type — the shader reads the buffer's actual element
count back with `arrayLength(&lights)` and loops over exactly that many
lights. Pressing `+`/`-` changes `self.light_count` (clamped 8..64) and
calls `_rebuild_lights()`, which:

1. Builds a new `PointLight` array (ring of lights above the scene) sized
   to the new count.
2. Creates a **new** `GPUBuffer` at that size (`STORAGE | COPY_DST`).
3. Creates a **new** bind group referencing it — a WebGPU bind group binds
   a fixed `(buffer, offset, size)` triple, so resizing a storage buffer
   always means a new bind group, never an in-place resize.

This is the WebGPU-side cost that balances the flexibility: growing the
light array is a (cheap, infrequent) buffer + bind-group rebuild, not a
per-frame operation, and the shader itself is never touched.

`SceneBlock` and `MaterialBlock` are ordinary `var<uniform>` buffers here
(bindings 0 and 1 of the teapot pipeline's bind group), sharing the exact
same `layouts.py` numpy dtypes as the OpenGL demo — WGSL's default
uniform-address-space layout follows the identical vec3-alignment rule as
`std140`, so the same padding trap, and the same fix (`X` key), reproduces
identically. The grid pipeline has its own bind group but reads the very
same `scene_buffer` `GPUBuffer` object as the teapot pipeline — one
`write_buffer()` call updates the VP matrix seen by both pipelines, the
WebGPU-side echo of the GL demo's "one UBO update, two programs" story
(WebGPU pipelines are immutable once built, unlike a GL program which
can be re-pointed at a different binding point at runtime — see the
design spec's WebGPU skeleton notes on bind-group-layout sharing for why
each pipeline gets its own bind group here rather than one shared across
both).

## Files

| File | Purpose |
| :-- | :-- |
| `layouts.py` | numpy structured dtypes for `SceneBlock` / `MaterialBlock` (both layouts) shared by both backends, plus `std140_offsets()` |
| `tests/test_layouts.py` | itemsize / offset checks against hand-computed std140 values |
| `main.py` | OpenGL demo |
| `shaders/Scene{Diffuse,Grid}{Vertex,Fragment}.glsl` | the two GL programs sharing `SceneBlock` |
| `StorageWebGPU.py` | WebGPU demo |
| `SceneShader.wgsl` / `GridShader.wgsl` | the two WebGPU pipelines |
| `WebGPUWidget.py` | local copy of the shared WebGPU Qt widget (per-folder, per repo convention) |
