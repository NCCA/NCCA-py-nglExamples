# WebGPU PBRTexture demo — design

## Goal

Add a WebGPU version of the existing `PBR/PBRTexture` OpenGL demo. It renders the
same scene using the same texture sets, but ports the `TexturePack` class to
WebGPU idioms rather than translating the OpenGL texture-unit model line for line.
Both demos live in the same folder and share `textures/` and `textures.json`.

## Scope

Match the OpenGL demo's content: a ~12×12 grid of teapots with random per-teapot
materials and rotations, a textured floor, four toggleable point lights drawn as
indicator spheres, and a first-person camera. Same key bindings.

## New files (all in `PBR/PBRTexture/`)

- `PBRTextureWebGPU.py` — the `WebGPUWidget` application: window, camera,
  input handling, per-frame render loop, `--smoketest`/`--debug` flags.
- `texture_pack_webgpu.py` — the WebGPU-idiomatic `TexturePack`.
- `PBRTexture.wgsl` — one shader module (vertex + fragment), a port of the GLSL
  PBR shader.

## The idiomatic TexturePack

OpenGL binds textures to global units with `glActiveTexture(GL_TEXTURE0 + n)` and
`activate_texture_pack` re-binds all five units before each draw. WebGPU has no
global texture-unit state — resources are bound through bind groups. So the port
is:

- Parse the *same* `textures.json`, reusing the existing duplicate-`TexturePack`-key
  preprocessing.
- For each material, load its five PNGs (`albedo`, `normal`, `metallic`,
  `roughness`, `ao`) via `ncca.ngl.Image`, pad RGB→RGBA, upload each to its own
  `GPUTexture` (`rgba8unorm`) with mipmaps.
- Bake **one `GPUBindGroup` per material** holding the five texture views plus a
  shared repeat/mipmap `GPUSampler`.
- `s_textures: dict[str, GPUBindGroup]`. `activate(render_pass, material)` becomes
  `render_pass.set_bind_group(2, bind_group)` — no per-unit rebinding.

The class is constructed with the `device` and the pipeline's material bind-group
layout, so every pack shares one layout that matches the shader.

## Bind group layout

- `@group(0)` per-draw transforms UBO: `MVP`, `M` (world), `normalMatrix`
  (mat4-padded), `textureRotation` (mat2, padded to WGSL alignment). Rewritten per
  teapot.
- `@group(1)` lights + camera: `lightPositions[4]`, `lightColors[4]`, `camPos`.
  Toggling a light rewrites its colour to black, matching OpenGL.
- `@group(2)` material textures + sampler (from `TexturePack`).

## Shader port (`PBRTexture.wgsl`)

Direct port of `shaders/PBRVertex.glsl` + `shaders/PBRFragment.glsl`:
Cook-Torrance BRDF (GGX distribution, Smith geometry, Schlick Fresnel), four-light
reflectance loop, screen-space normal mapping using WGSL `dpdx`/`dpdy` in place of
GLSL `dFdx`/`dFdy`, `pow(albedo, 2.2)` de-gamma, HDR tonemap and gamma correction.
Vertex stride is the standard PyNGL 8-float layout (pos3, normal3, uv2).

## Render loop

1. Clear colour + depth pass.
2. Seeded RNG (`Random.set_seed_value`) picks a material per teapot so the layout
   matches the OpenGL look; write the transforms UBO and set the material bind
   group per teapot, then draw.
3. Textured floor drawn with the `greasy` pack.
4. Light-indicator spheres via a `PipelineFactory` `SINGLE_COLOUR_TRIANGLES`
   pipeline fed sphere `PrimData`, gated on the `L` toggle. No extra WGSL file.

## Controls (match OpenGL `main.py`)

Arrow keys move the first-person camera, LMB rotates, wheel zooms, `1`–`4` toggle
lights, `L` toggles the light spheres, `R` reseeds the material layout, `Space`
resets the camera, `Esc` quits.

## Verification

No unit tests (GPU/GUI). Wire up the repo's `--smoketest MS` convention (render
for MS ms, print `SMOKETEST OK`, exit) and run it as the verification step.
Capture a screenshot for the README.

## Documentation

- Update `PBR/PBRTexture/README.md` to describe the WebGPU variant and how its
  `TexturePack` differs from the OpenGL one, with a **Future work** section noting
  the deferred optimisations below.
- Add a link to the demo from the root `README.md`.
- Prose in Jon's writing style.

## Deferred (future work, noted in README)

- Instanced/batched teapot drawing instead of one bind group + UBO write per
  teapot per frame.
- Mipmap generation on the GPU (blit chain) rather than per-level upload if
  `Image` mip support is limited.
- Sharing a single sampler object across all packs (already planned) and a single
  transforms UBO with dynamic offsets.
```
