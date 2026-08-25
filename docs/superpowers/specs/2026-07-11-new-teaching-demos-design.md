# New Teaching Demos — Design & Shared Conventions

This spec covers a batch of new PyNGLDemos spanning 2nd Year UG through MSc. Two demos
from the original list (**Blending**, **OITransparency**) are already implemented and act
as the reference implementations for every convention below — read both folders before
starting any task. The remaining demos are broken into three plan files:

- `../plans/2026-07-11-ug2-demos-plan.md` — Instancing, SceneGraph, StencilOutline, Billboards
- `../plans/2026-07-11-intermediate-demos-plan.md` — SkyBoxEnvMap, ShadowMapping, PostProcessChain, GeometryTessellation, UBOStorageBuffers
- `../plans/2026-07-11-msc-demos-plan.md` — SkeletalAnimation, BoidsCompute, MarchingCubes, RayMarchingSDF, GimbalLock, IBL (stretch)

## Suggested implementation order

Within each plan, demos are ordered by priority. Cross-plan dependencies:
SkyBoxEnvMap must precede IBL; PostProcessChain's ping-pong FBO helper is
reused conceptually by IBL's BRDF LUT; Billboards informs BoidsCompute's
rendering; Blending/OITransparency patterns (already done) are prerequisites
for nothing but are the style reference for everything.

## Repo conventions (apply to every demo)

1. **Folder layout** — one self-contained top-level folder per demo:
   entry scripts (`main.py` for OpenGL, `<Name>WebGPU.py` for WebGPU), `shaders/` for
   GLSL, top-level `*.wgsl` for WebGPU, `README.md`, a preview `.png` named after the
   demo, optional `tests/` for numpy-only maths. No imports between demo folders.
2. **Entry scripts** are executable: shebang `#!/usr/bin/env -S uv run --script` and
   `chmod +x`. `RunDemos.py` discovers them automatically — no registration.
3. **Smoketest** — every entry script supports `--smoketest` (checked in `__main__`):
   run one paint pass via `QTimer.singleShot(200, app.quit)`, print `SMOKETEST OK`, exit 0.
   Verify with `QT_QPA_PLATFORM=offscreen uv run <Demo>/main.py --smoketest`.
   (Blending/OITransparency predate this; add the flag to them opportunistically.)
4. **Testable maths** — anything that is pure maths (sorting, weights, graph
   composition, SDFs, skinning, boid rules) lives in a numpy-only module (no GL/Qt/wgpu
   imports) with pytest tests in `tests/`, using the
   `sys.path.insert(0, str(Path(__file__).parent.parent))` pattern from
   `RayPickingSelection/tests/`. Run with `uv run pytest <Demo>/tests`.
5. **Lint** — `ruff check` and `ruff format` must pass (see `.pre-commit-config.yaml`).
6. **Docs** — `README.md` per demo (description, controls table, teaching points,
   image reference `![](<Demo>.png)`); add a row (with preview image) to the root
   `README.md` under the appropriate section, adding the section + Contents entry if new.
7. **Screenshots** cannot be captured by a headless agent — leave the referenced PNG
   missing and list it in the final report for Jon to capture.
8. **No library edits** — `ncca.ngl` lives at `/Users/jmacey/teaching/Code/PyNGL`
   (editable install). If a demo seems to need a library change, stop and report;
   demo-side workarounds are preferred. The agent MAY read that source tree to check
   API signatures (recommended for anything not demonstrated by an existing demo).
9. **Commits** — one demo per commit: `git add <Demo>/ README.md && git commit -m "feat: add <Demo> demo"`.

## OpenGL skeleton

Copy from `Blending/main.py` (which follows `BlankPySide6NGL/using_mixin.py`):

- `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)` with
  `setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01,
  zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` — provides mouse orbit/pan/zoom
  and the attributes `spin_x_face`, `spin_y_face`, `model_position`.
- GL 4.1 core (macOS ceiling): `QSurfaceFormat` major 4 / minor 1 / CoreProfile /
  24-bit depth / 4x samples. Shaders `#version 330 core` (use `410 core` only when a
  feature needs it, e.g. tessellation).
- **Attribute locations are fixed** by `ncca.ngl` primitives: 0 = inVert (vec3),
  1 = inNormal (vec3), 2 = inUV (vec2).
- `ShaderLib.load_shader(name, vert_path, frag_path)` with absolute paths built from
  `Path(__file__).parent / "shaders" / ...`; `ShaderLib.use(name)`;
  `ShaderLib.set_uniform(...)` accepts scalars, unpacked floats, Vec3/Vec4, Mat3/Mat4.
- Built-ins: `DefaultShader.COLOUR` (uniforms `MVP`, `Colour`) and
  `DefaultShader.DIFFUSE` (`MVP`, `Colour`, `lightPos`, `lightDiffuse`, `normalMatrix`).
- `Primitives.load_default_primitives()` then `Primitives.draw("teapot")`;
  `Primitives.create(Prims.SPHERE, "name", radius, precision)`,
  `Prims.LINE_GRID` (w, d, steps), `Prims.TRIANGLE_PLANE` (w, d, wp, dp, normal Vec3),
  `Prims.TORUS` if available (check PyNGL source; otherwise generate in numpy).
- HUD text: `Text.add_font("Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20)`
  in `initializeGL`, `Text.render_text("Arial", x, y, str, Vec3 colour)` in `paintGL`,
  `Text.set_screen_size(w, h)` in `resizeGL`.
- `DebugApplication` (notify-override) block copied verbatim; `--debug` flag support.

## Maths conventions (critical — get these wrong and everything renders garbage)

- numpy side uses the **row-vector convention**: points transform as `row_vec @ M`,
  translation lives in **row 3** (`m[3,0..2]`). See `RayPickingSelection/picking_maths.py`.
- Matrix composition reads like column-major maths: `MVP = project @ view @ model`
  applies `model` first (rightmost first). Shaders then do `MVP * vec4(v, 1.0)` in both
  GLSL and WGSL — `Mat4.to_numpy()` bytes land in WGSL as column-major, which cancels out.
- Normal matrix: `Mat3.from_mat4(MV).inverse().transposed()` (GL) or a full
  `mv.inverse().transposed()` Mat4 in WGSL uniform blocks (mat3 has padding issues).
- **Avoid `Transform.set_rotation` with more than one non-zero axis** — the Euler
  composition order is not obvious from call sites. Compose explicitly instead:
  `model = Mat4.rotate_y(b) @ Mat4.rotate_x(a)` (X applied first), then set row-3
  translation. See `OITransparency/main.py::_panel_matrix`.
- Mouse scene transform (copy verbatim): `rot_y @ rot_x` from `spin_*_face`, then
  row-3 translation from `model_position`.
- View-space depth of a local point: `([x,y,z,1] @ model_view_numpy)[2]` — negative in
  front of the camera; back-to-front = ascending. Reusable helper in
  `Blending/blend_scene.py`.

## WebGPU skeleton

Copy from `Blending/BlendingWebGPU.py` / `OITransparency/OITWebGPU.py`:

- Copy `WebGPUWidget.py` into the demo folder (each folder keeps its own copy; take the
  latest from `FBODemos/WebGPURenderToTexture/`). Scene class subclasses it, sets
  `self.msaa_sample_count`, `self.device = get_default_device()`, builds pipelines and
  scene, then calls `self._create_render_buffer()`.
- **MSAA**: widget default flow expects `msaa_sample_count=4` with a multisample colour
  target resolved into `colour_buffer_texture`. For multi-target/multi-pass demos set
  `msaa_sample_count=1` and OVERRIDE `_create_render_buffer` to build your own textures —
  it must still set `colour_buffer_texture`, `colour_buffer_texture_view` (rgba8unorm,
  RENDER_ATTACHMENT|COPY_SRC) and `readback_buffer`
  (`self._calculate_aligned_buffer_size()`), and it is re-called on every resize, so
  also rebuild any bind groups that reference the texture views. Guard with
  `if not hasattr(self, "device"): return`.
- Geometry: `PrimData.primitive(Prims.TEAPOT.value)` returns interleaved
  x,y,z,nx,ny,nz,u,v float32 (stride 8*4; vertex count = data.size // 8). Procedural
  prims (sphere/grid) are GL-only — generate quads/spheres in numpy (see `quad()` in
  the Blending demo).
- Uniform blocks: numpy structured dtype mirroring WGSL std140-ish layout — mind vec3
  padding (offsets 0/12/16... pattern in `FBODemos/WebGPURenderToTexture/TeapotPipeline.py`).
  Write with `device.queue.write_buffer(buf, 0, arr.tobytes())`.
- **Bind groups across pipeline variants**: pipelines created with `layout="auto"` get
  unique incompatible layouts. When several pipelines must share per-object bind groups,
  create an explicit `create_bind_group_layout` + `create_pipeline_layout`
  (see `Blending/BlendingWebGPU.py::_create_pipelines`). `layout="auto"` is fine for a
  single pipeline whose bind group you rebuild alongside it.
- Full-screen passes: triangle from `@builtin(vertex_index)`, `render_pass.draw(3)`,
  and **flip v** (`uv.y = 1.0 - uv.y`) when sampling a texture you rendered — see
  `OITransparency/CompositeShader.wgsl`.
- Projection: `perspective(fov, aspect, near, far, PerspMode.WebGPU)`.
- HUD: `self.render_text(x, y, text, size, "Arial", QColor(255,255,255))` before/after
  `_update_colour_buffer()` in `paintWebGPU`.
- Mouse/keyboard handlers: no mixin for QWidget — copy the handler block from the
  Blending WebGPU demo.
- Do not wrap `paintWebGPU` bodies in bare `try/except: pass` (some older demos do);
  let `DebugApplication`-style visibility win.

## Backend policy

Jon teaches mainly WebGPU but keeps OpenGL for quick demos. Per-demo backend choices are
stated in each plan. Where both are built, share the scene/maths in a numpy-only module
(the `blend_scene.py` pattern) so the two versions render the identical scene.

## Definition of done (every demo)

- [ ] `ruff check` + `ruff format --check` clean; `uv run pytest <Demo>/tests` passes
- [ ] `QT_QPA_PLATFORM=offscreen uv run <Demo>/<entry>.py --smoketest` prints SMOKETEST OK for every entry script
- [ ] README.md written; root README.md row + (if new) section/Contents entry added
- [ ] Entry scripts executable with the uv shebang
- [ ] Screenshot listed as TODO for Jon (agent cannot capture)
- [ ] Committed as a single `feat: add <Demo> demo` commit
