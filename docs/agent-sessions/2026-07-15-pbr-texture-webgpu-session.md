# 2026-07-15 — WebGPU PBRTexture demo

## Goal

Add a WebGPU version of the `PBR/PBRTexture` OpenGL demo, reusing the same
textures and `textures.json`, but porting the `TexturePack` class to WebGPU
idioms (a bind group per material instead of OpenGL global texture units).

## Approach

- Brainstormed and wrote a design doc (`docs/superpowers/specs/2026-07-15-pbr-texture-webgpu-design.md`).
- Matched the OpenGL scene: ~170 teapots in a grid with random materials and
  rotations, a textured floor, four toggleable light spheres, first-person camera.
- WebGPU `TexturePack` loads each material's five maps into their own
  `GPUTexture` and bakes one `GPUBindGroup`; `activate` is a single
  `set_bind_group`.
- Three-group bind layout: per-object transforms (`@group(0)`, dynamic offset,
  one padded slot per teapot so the whole grid draws in one pass), lights +
  camera (`@group(1)`), material pack (`@group(2)`).
- `PBRTexture.wgsl` is a direct port of the GLSL PBR shader (`dpdx`/`dpdy` for
  the tangent-space normal mapping). Light spheres use a tiny solid-colour
  `LightSphere.wgsl`.

## Files changed

- `PBR/PBRTexture/PBRTextureWebGPU.py` (new) — the WebGPU widget/app.
- `PBR/PBRTexture/texture_pack_webgpu.py` (new) — WebGPU texture-pack manager.
- `PBR/PBRTexture/PBRTexture.wgsl` (new) — PBR shader.
- `PBR/PBRTexture/LightSphere.wgsl` (new) — unlit light-indicator shader.
- `PBR/PBRTexture/PBRTextureWebGPU.png` (new) — screenshot.
- `PBR/PBRTexture/README.md` — added WebGPU section + Future work.
- `README.md` — noted the WebGPU version on the PBRTexture row.
- `docs/superpowers/specs/2026-07-15-pbr-texture-webgpu-design.md` (new).

## Notable fix

The metallic/roughness/AO maps are single-channel grayscale (2D arrays), which
the initial RGB→RGBA path didn't handle, causing a `write_texture` size overrun.
Added `_to_rgba` to expand grayscale (replicated across RGB) and 4-channel data.

## Commands run

```bash
git worktree add .worktrees/pbr-texture-webgpu -b agent/pbr-texture-webgpu
uv run ruff check --select I --fix <files>
uv run ruff format <files>
uv run ruff check <files>              # all checks passed
uv run PBRTextureWebGPU.py --smoketest 500   # SMOKETEST OK
```

## Deferred (see README Future work)

- Upload grayscale maps as `r8unorm` instead of RGBA.
- Generate mipmaps via a blit chain (WebGPU has no `glGenerateMipmap`).
- Instanced / per-instance-storage-buffer drawing instead of a bind group and
  uniform write per teapot.
