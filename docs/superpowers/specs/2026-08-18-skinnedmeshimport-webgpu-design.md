# SkinnedMeshImport: WebGPU sibling

## Goal

`SkinnedMeshImport` currently has one backend: OpenGL, via `QOpenGLWindow`. Add
a WebGPU version in the same folder, the way `BVHViewer` already has both —
same window, same timeline, same camera and four-view controls, different
renderer underneath.

## Non-goals

- No change to `mesh.py` / `SkinnedMesh`'s public behaviour or its tests.
- No shared code extracted between this demo and `BVHViewer` — each demo
  folder stays self-contained per the repo's convention.
- Not matching the OpenGL shader's bone-count ceiling (see below).

## Precedent

`BVHViewer/main_webgpu.py`, `webgpu_renderer.py`, `bvh_scene_webgpu.py` and
`bvh_webgpu.wgsl` already do this split for a different demo. The design
below mirrors that structure directly rather than inventing a new one.

## File layout

New files in `SkinnedMeshImport/`:

- `main_webgpu.py` — `SkinWebGPUViewport(WebGPUWidget)` and the entry point,
  mirroring `BVHViewer/main_webgpu.py`'s `BvhWebGPUViewport`.
- `webgpu_renderer.py` — owns the wgpu device objects: pipeline, vertex/index
  buffers, the bone storage buffer, and one texture+sampler+bind-group per
  submesh texture. No separate "scene adapter" file is needed — unlike BVH's
  `bvh_scene_webgpu.py`, which exists to turn a joint hierarchy into instance
  data, `mesh.py`'s `SkinnedMesh` already exposes flat numpy arrays
  (`positions`, `normals`, `texcoords`, `bone_ids`, `bone_weights`,
  `indices`) and per-submesh `(index_offset, index_count, texture_path)` —
  the renderer can consume that directly.
- `skin_webgpu.wgsl` — the shader.

Existing files touched:

- `main.py` — two small, targeted changes (see below), both refactors BVHViewer's own `main.py` already made for the same reason.

## Changes to `main.py` (OpenGL file)

1. **Injectable viewport.** `MainWindow.__init__` currently hardcodes
   `self.viewport = SkinViewport(model_path)`. Give it the same
   `viewport: SkinViewport | QWidget | None = None` parameter BVHViewer's
   `MainWindow` has, defaulting to constructing a `SkinViewport` when `None`.
   Every other `MainWindow` method (`_open_file_dialog`,
   `_sync_timeline_to_mesh`, playback transport) already only touches
   generic `self.viewport.{mesh, model_path, current_frame, set_frame,
   load_model}` — no other change needed for `main_webgpu.py` to reuse them
   unchanged, exactly like BVHViewer's WebGPU `MainWindow` reuses
   `open_bvh_dialog`/`load_bvh` untouched.
2. **Extract `_parse_args()` and `main()`.** Currently the CLI/app setup is
   inline under `if __name__ == "__main__":`. Pull it into a top-level
   `_parse_args(argv=None)` and `main(argv=None) -> int`, matching
   `BVHViewer/main.py`'s shape, so `main_webgpu.py` can call `_parse_args()`
   instead of redeclaring the same flags.

`SkinViewport`, `_compute_view_setup`, `_look_at`, `OrthoView`, the view
constants, `DebugApplication`, `MESH_FILE_FILTER`, `DEFAULT_MODEL` and
`MAX_BONES` are untouched.

## `SkinWebGPUViewport`

Duplicates `SkinViewport`'s camera/four-view/pane-input logic (bounding-box
read, MD5 Z-up `rotate_x(-90)` correction, `FirstPersonCamera` aim, ortho
pane layout, click-to-maximize, WASD/mouse handlers) the same way
`BvhWebGPUViewport` duplicates `BvhViewport`'s — the two Qt base classes
(`QOpenGLWindow` vs `WebGPUWidget`) aren't related, so there's no clean base
to share it from without inventing an abstraction the rest of the repo
doesn't use. Differences from `SkinViewport`:

- `FirstPersonCamera(..., PerspMode.WebGPU)` and `ortho(..., PerspMode.WebGPU)`
  for WebGPU's 0..1 depth range, matching `WebGPUOrthoView`/
  `_set_perspective_projection` in `BVHViewer/main_webgpu.py`.
- `load_model(path)` tears down and rebuilds wgpu buffers/textures instead of
  a GL VAO — same contract as `SkinViewport.load_model`: raise on a bad file
  without mutating the currently-loaded mesh, so the inherited
  `MainWindow._open_file_dialog` keeps working unchanged.
- `paintWebGPU` draws each pane (1 or 4, or 1 maximized) via
  `render_pass.set_viewport`/`set_scissor_rect`, same loop shape as
  `BvhWebGPUViewport.paintWebGPU`.

## `webgpu_renderer.py`

- **Vertex data**: five separate vertex buffers (position vec3, normal vec3,
  uv vec2, bone_ids vec4, bone_weights vec4) uploaded straight from
  `mesh.positions` / `.normals` / `.texcoords` / `.bone_ids` / `.bone_weights`
  — no interleaving step, since `mesh.py` already keeps them as separate
  arrays and `TextureWebGPU.py` already establishes the multi-vertex-buffer
  pattern in this repo.
- **Index data**: `mesh.indices` (uint32) as one index buffer;
  `draw_indexed(submesh.index_count, 1, submesh.index_offset)` per submesh,
  same loop as the OpenGL `_draw_mesh`.
- **Bone palette**: a `read_only-storage` buffer sized to the mesh's actual
  bone count (`len(mesh.bone_names)`), rewritten every frame from
  `mesh.bone_transforms(time_seconds)` stacked into one `(N, 4, 4)` float32
  array. No `MAX_BONES` cap — that ceiling exists in the OpenGL version only
  because GLSL needs a fixed-size uniform array; a storage buffer doesn't.
  Resized (destroy + recreate + rebuild bind group) on load, same pattern as
  `BvhWebGPURenderer.update_instances`'s capacity growth.
- **Textures**: one `texture_2d<f32>` + `sampler` + bind group per unique
  submesh texture path, built the same way `TextureWebGPU.py._create_texture`
  builds its single texture (`ncca.ngl.Image`, RGB→RGBA padding,
  `write_texture`), looped over `mesh.submeshes` the way OpenGL's
  `_load_textures` loops. A missing/failed texture gets the same 1x1 flat
  white fallback as the OpenGL path, logged as a warning rather than losing
  the mesh.
- **Camera/light uniform**, one buffer per pane (4, like `BvhWebGPURenderer`,
  so four simultaneously-drawn panes in one command buffer don't stomp each
  other's uniform writes before submission): `view_projection`, `model`
  (the constant Z-up correction matrix), `eye_position`, `light_position`.
  Ambient/diffuse/specular colour and shininess are `const`s in the WGSL
  file, not uniform data — they're hardcoded constants in the OpenGL path
  too (`ShaderLib.set_uniform("material.ambient", 0.2, 0.2, 0.2, 1.0)` etc.,
  every frame, never varying), so baking them into the shader is a direct
  port, not a simplification.

## `skin_webgpu.wgsl`

- Vertex stage: same four-bone linear-blend skin as `SkinVertex.glsl`
  (`bone[ids[i]] * weights[i]`, summed), reading bone ids as `vec4<f32>` and
  casting to `u32` per lookup (mirrors the GLSL comment: PyNGL vertex buffers
  are float32, so ids arrive as floats regardless of backend).
- Lighting: Blinn-Phong ported to **world space** instead of the GLSL's eye
  space — `world_position = model * skinned_position`, light and eye
  direction both computed against `world_position` rather than through a
  view-space `MV`. This matches the world-space convention already
  established across this repo's other WebGPU lit shaders (the PBR family),
  and drops the need to upload a separate `MV` matrix.
- Texture sampling: `textureSample(t_diffuse, s_diffuse, vec2(uv.x, 1.0 -
  uv.y))`. `mesh.py` already flips V once for OpenGL's bottom-left texture
  origin (a backend-specific flip baked into the otherwise backend-agnostic
  loader); WebGPU's texture origin is top-left, so the shader flips it back.
  Documented as a wrinkle in the README rather than fixed at the loader,
  same treatment the OpenGL README already gives the MD5 Z-up wrinkle.

## Testing

- `uv run pytest SkinnedMeshImport/tests` must still pass unchanged (no
  `mesh.py` behaviour change).
- Manual: `uv run SkinnedMeshImport/main_webgpu.py`, exercised via the `run`
  skill — load the default guard model, confirm skinning animates, texture
  orientation is correct (not upside-down), four-view/click-to-maximize,
  File > Open with a different rigged mesh, `--smoketest`.
- A WebGPU screenshot is not required to match a hard repo rule — BVHViewer
  itself has no WebGPU-specific screenshot and no root-README entry. Update
  the existing root README `SkinnedMeshImport` line to mention both backends
  instead of adding a second image.

## README updates (`SkinnedMeshImport/README.md`)

Mirror `BVHViewer/README.md`'s shape: a short "There is also a WebGPU
version" paragraph up top with the run command, the new files added to the
"How it works" list, and a "Differences from the OpenGL version" section
covering: no bone-count ceiling, and the V-flip wrinkle.
