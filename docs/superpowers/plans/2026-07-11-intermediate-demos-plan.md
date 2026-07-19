# Intermediate Demos Implementation Plan — SkyBox/EnvMap, Shadows, Post-Processing, Geometry/Tessellation, UBO/Storage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Read first:** `docs/superpowers/specs/2026-07-11-new-teaching-demos-design.md`.
Reference implementations: `Blending/`, `OITransparency/` (multi-pass + MRT patterns),
`FBODemos/SimpleFBO` (FBO basics), `WebGPUShadows/` (the WebGPU shadow counterpart),
`ScreenTri/` (fullscreen triangle).

**Goal:** Five 3rd-year demos. Audience: students comfortable with FBOs and multiple
shader programs.

---

## Task 1: SkyBoxEnvMap (OpenGL + WebGPU)

Cubemaps: skybox rendering plus reflection/refraction on a teapot. Prerequisite for the
MSc IBL demo.

**Files:**
- `SkyBoxEnvMap/main.py` (GL), `SkyBoxEnvMap/SkyBoxEnvMapWebGPU.py` + `WebGPUWidget.py`
- `SkyBoxEnvMap/cubemap_gen.py` — numpy-only procedural cubemap: per-face 256² RGBA8
  gradient sky (horizon band, sun disc on +z face, ground colour below) so no image
  assets are needed; must return the 6 faces in GL order (+x,-x,+y,-y,+z,-z)
- `SkyBoxEnvMap/shaders/` SkyboxVertex/Fragment, EnvMapVertex/Fragment;
  `SkyBoxEnvMap/SkyBoxShader.wgsl`, `EnvMapShader.wgsl`
- `SkyBoxEnvMap/tests/test_cubemap_gen.py`, `README.md`

**GL specifics:**
- Upload: `glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ...)` per face, LINEAR
  filtering, CLAMP_TO_EDGE all three axes.
- Skybox: draw a unit cube with `glDepthFunc(GL_LEQUAL)` and depth writes off, vertex
  shader sets `gl_Position = (P * mat4(mat3(V)) * v).xyww` (strip translation, force
  depth to far plane); frag samples `samplerCube` with the local position. Restore
  `GL_LESS` after.
- Teapot: `reflect(I, N)` / `refract(I, N, 1.0/1.52)` in world space (needs `M`,
  `camPos` uniforms), plus a Schlick-Fresnel mix mode.

**WebGPU specifics:**
- One `create_texture` with `size=(256,256,6)`, then
  `texture.create_view(dimension="cube")`; sample with `texture_cube<f32>`.
  Upload the faces with `queue.write_texture` per array layer.
- Skybox pass draws after opaque with `depth_compare="less-equal"` and
  `depth_write_enabled=False`; same mat3(view) trick in WGSL (build the
  rotation-only matrix on the CPU and upload it — simpler than WGSL matrix surgery).

**Controls:** `M` cycle teapot mode (reflect / refract / fresnel mix / plain diffuse),
`+`/`-` IOR in refract mode, HUD shows mode.

**Tests:** cubemap_gen face count/shape/dtype; horizon continuity — right edge of +x
face equals left edge of +z face (catches face-order mistakes CPU-side, the classic
cubemap bug).

**Pitfalls:** cubemap face order and per-face v-flip conventions differ between GL and
WebGPU — the horizon-continuity test plus one manual orbit is the check; the skybox
must be drawn LAST (after opaque) for early-z, not first, despite most tutorials.

- [ ] cubemap_gen + tests green
- [ ] GL demo (skybox + 4 teapot modes)
- [ ] WebGPU demo matching
- [ ] README, root README row (new "Environment & Sky" or into Textures & Materials), smoketests, ruff, commit

---

## Task 2: ShadowMapping (OpenGL)

The classic two-pass depth-map shadow demo — deliberately the GL mirror of the existing
`WebGPUShadows` so students can diff the two APIs.

**Files:**
- `ShadowMapping/main.py`, `ShadowMapping/shaders/` (ShadowDepthVertex/Fragment,
  ShadeVertex/Fragment, DebugQuadVertex/Fragment), `README.md`.
  Maths worth testing: `light_space_matrix(light_pos, target, ortho_extents)` in
  `ShadowMapping/shadow_maths.py` + `tests/` (point projects into [0,1]² for points
  inside the frustum).

**Passes:**
1. Depth pass: 2048² FBO with a `GL_DEPTH_COMPONENT24` **texture** (no colour
   attachment: `glDrawBuffer(GL_NONE)`, `glReadBuffer(GL_NONE)`), scene rendered with
   the light's ortho view-projection (directional light orbiting on a timer, `L` pauses).
2. Shade pass: default framebuffer; fragment shader transforms world pos by the light
   matrix, perspective-divides, maps to [0,1], compares against
   `texture(shadowMap, uv).r` with bias; PCF mode averages a 3×3 kernel of
   `textureSize`-scaled taps.
3. Debug inset: draw the depth texture on a small screen-corner quad (own tiny shader,
   linearised for visibility).

**Controls:** `P` toggle PCF, `B`/`Shift+B` bias up/down (HUD shows value — acne vs
peter-panning live), `C` toggle front-face culling during the depth pass, `V` toggle
depth-map inset, `L` pause light orbit.

**Pitfalls:** depth texture needs `GL_TEXTURE_COMPARE_MODE` NONE for manual `.r` reads
(we compare in-shader; do NOT use sampler2DShadow — keep it explicit for teaching);
CLAMP_TO_BORDER with border 1.0 so geometry outside the light frustum is lit, not
shadowed; bias applied in light space, shown as the artefact toggle.

- [ ] shadow_maths + tests green
- [ ] Demo with all toggles + inset
- [ ] README (compare with WebGPUShadows explicitly, table of artefacts→fixes), root README row under Lighting & Shadows, smoketest, ruff, commit

---

## Task 3: PostProcessChain (OpenGL)

HDR rendering + bloom + tonemapping: a real multi-FBO post chain built on the
OITransparency FBO/composite machinery.

**Files:**
- `PostProcessChain/main.py`, `PostProcessChain/shaders/` (SceneVertex/Fragment emitting
  HDR values, BrightPassFragment, BlurFragment, TonemapFragment — all post passes use
  the CompositeVertex.glsl fullscreen-triangle trick from OITransparency),
  `PostProcessChain/tonemap_maths.py` + `tests/` (Reinhard and ACES curves: monotonic,
  map 0→0, large input →≤1), `README.md`.

**Chain:**
1. Scene → RGBA16F FBO: grid, teapot, and 4–6 emissive spheres with `Colour` values
   deliberately >1 (e.g. (8, 4, 0.5)) — a `DefaultShader`-style frag that just outputs
   the HDR colour for emissives, N·L for the rest.
2. Bright pass → half-res RGBA16F: `max(colour - threshold, 0)` (threshold uniform).
3. Separable Gaussian blur: two half-res FBOs ping-ponged H/V for `n_passes` iterations
   (5-tap weights hardcoded, offsets scaled by texel size).
4. Tonemap composite → screen: `scene + bloomStrength * blur`, then exposure multiply,
   then operator: 1 clamp (none), 2 Reinhard `c/(1+c)`, 3 ACES fitted curve; finally
   gamma 2.2. All fed from `tonemap_maths` constants so the shader and the tested numpy
   agree (copy the coefficients both places, test guards the numpy one).

**Controls:** `B` bloom on/off, `T` cycle tonemap operator, `E`/`Shift+E` exposure,
`+`/`-` blur passes (1..8), `H` split-screen mode: left half raw clamp, right half full
chain (implement with `gl_FragCoord.x` branch in the tonemap shader — cheap and vivid).

**Pitfalls:** resize must rebuild all four FBOs (reuse the OITransparency
`_create_fbos`/`_delete_fbos` structure); blur at half resolution both for speed and
softness — sample the full-res bright pass with LINEAR filtering; without gamma the
whole exercise looks broken.

- [ ] tonemap_maths + tests green
- [ ] Demo: full chain + split screen
- [ ] README (why HDR before tonemap, curve plots described), root README row (new "Post Processing" grouping is fine inside FBO section), smoketest, ruff, commit

---

## Task 4: GeometryTessellation (OpenGL only — these stages don't exist in WebGPU, which is itself a teaching point)

**Files:**
- `GeometryTessellation/normals_main.py` — geometry-shader normal visualiser
- `GeometryTessellation/tess_main.py` — tessellated displaced plane
- `GeometryTessellation/shaders/` per sub-demo, `README.md` (one folder, two entry
  scripts — RunDemos handles multiple executables per folder)

**API check FIRST:** inspect `ShaderLib` in the PyNGL source
(`/Users/jmacey/teaching/Code/PyNGL`) for geometry/tess-stage support in
`load_shader`/its lower-level API. If absent, compile programs with raw PyOpenGL
(`glCreateShader(GL_GEOMETRY_SHADER)` etc.) inside a small local `program.py` helper in
the demo folder — do NOT edit the library; note the outcome in the README. All shaders
`#version 410 core` (GL 4.1 supports both stages; SSBO/compute do NOT exist here).

**normals_main.py:** teapot drawn twice: normal diffuse pass, then a second program
whose geometry shader (`triangles` → `line_strip, max_vertices=6`) emits a line per
vertex along the normal, length uniform via `+`/`-`; `F` switches to one line per *face*
(average position/normal) — the two modes make smooth vs faceted normals visible.

**tess_main.py:** a 16×16 grid of `GL_PATCHES` (4 verts per patch,
`glPatchParameteri(GL_PATCH_VERTICES, 4)`); TCS sets outer/inner levels from
camera-distance (`distance-based LOD`, clamped 1..64, toggle `L` for fixed level via
`+`/`-`); TES (quads, fractional_even) bilinearly interpolates the patch, displaces y
by procedural fbm noise (implement fbm in GLSL, 4 octaves — no texture needed), computes
the normal by finite differences of the same noise; frag shades by height + N·L.
`W` wireframe (`glPolygonMode`) is essential here — the LOD is the visual.

**Pitfalls:** patches draw nothing without the patch-vertices call; TCS must only write
levels on invocation 0; fractional tessellation avoids popping — mention spacing modes
in README; there is no `Primitives` path for patches, build the grid VBO in numpy.

- [ ] ShaderLib stage support determined; helper written if needed
- [ ] Normal visualiser (vertex + face modes)
- [ ] Tessellation demo (distance LOD + manual level + wireframe)
- [ ] README (pipeline diagram of the 5 programmable stages, WebGPU-absence note), root README row (new "Geometry & Tessellation Shaders" section), smoketests, ruff, commit

---

## Task 5: UBOStorageBuffers (OpenGL UBO + WebGPU storage buffer)

Explicit buffer-backed uniforms and the std140 padding rules everyone gets wrong.

**Files:**
- `UBOStorageBuffers/main.py` (GL UBO), `UBOStorageBuffers/StorageWebGPU.py` +
  `WebGPUWidget.py`, `UBOStorageBuffers/layouts.py` (the numpy structured dtypes used by
  BOTH backends + a `std140_offsets()` explainer function), `tests/test_layouts.py`,
  shaders per backend, `README.md`.

**GL demo:** one UBO holding a `SceneBlock { mat4 VP; vec4 lightPos; vec4 lightColour; }`
bound at binding point 0 and shared by TWO different shader programs (diffuse teapot +
colour grid) via `glUniformBlockBinding` + `glBindBufferBase` — the point is one
update feeds many programs. Second UBO `MaterialBlock` with a deliberate trap: a
`vec3 albedo; float shininess;` pair — key `X` switches the CPU dtype between the
correct std140 layout (vec3 padded to 16) and the naive packed one, visibly corrupting
shininess. HUD names the active layout. NOTE: SSBOs are GL 4.3+, unavailable on macOS
GL 4.1 — the README says exactly that and hands over to the WebGPU half.

**WebGPU demo:** same scene; a `var<storage, read>` runtime-sized array of point lights
(8..64, `+`/`-` keys) accumulated in the fragment shader — the thing a UBO cannot do —
plus the same padding trap in a uniform struct (WGSL offsets printed in HUD).

**Tests:** `layouts.py` dtypes: itemsize and field offsets match std140 hand-computed
values for both blocks; the "naive" layout differs exactly where the README says it does.

**Pitfalls:** `glBufferSubData` from a numpy structured array needs `.tobytes()`;
UBO binding points vs uniform locations confusion is the core lesson — diagram in README.

- [ ] layouts + tests green
- [ ] GL UBO demo (shared block across 2 programs + padding trap)
- [ ] WebGPU storage demo (runtime-sized light array)
- [ ] README (std140 table, GL4.1/SSBO caveat), root README row, smoketests, ruff, commit
