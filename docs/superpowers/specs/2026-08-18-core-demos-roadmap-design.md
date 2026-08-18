# Design: Core Demos Roadmap (NGL9Demos port, wave 2)

## Goal

Port a further batch of "core" NGL9Demos to PyNGLDemos, each with both an OpenGL and a
WebGPU version except where the underlying GPU technique has no WebGPU equivalent.

## Relationship to prior specs

`2026-07-04-ngl9-demo-port-design.md` scoped 23 demos in 5 phases. Only Phase 1 (10
"straightforward reuse" demos: Camera, ColourObj, CurveDemos, QuatSlerp, KleinBottle,
FrustumCull, PointCloud, AnimatedTextures, Interpolation, ImageHeightMap) was ever turned
into a plan and built. Of the rest:

- **Built separately, under different plans:** MassSpring (`2026-07-17-mass-spring*`),
  BVH → `BVHViewer` (large standalone effort), Instancing, TessellationShader →
  `GeometryTessellation`, RayMarching → `RayMarchingSDF`, SpatialHash → reinterpreted as
  `WebGPUCompute/SpatialHash2D`/`SpatialHash3D` (WebGPU-compute demos, not a port of the
  original OpenGL broad-phase visualization).
- **Never built, still open:** AffineTransforms, GameKeyControl, MorphObj, OctreeAbstract,
  the combined "TextureBuffer+TexelFetch", GeometryShaders, Sponza.

This spec **supersedes** that remaining open scope with updated findings from a fresh
source-code survey (see below), plus adds demos the original spec never covered
(LookAtDemos, ViewToWorldTransform, Spotlight, Collisions, ShadedGrid, ImageMaze,
ResetLine, LoadShaderFromJSon). It also changes the backend policy: the original spec was
OpenGL-only; this wave targets **OpenGL + WebGPU per demo**, per the
`2026-07-11-new-teaching-demos-design.md` convention set, except where noted.

## Scope changes from the source survey

A fresh read of the NGL9Demos source (not just folder names) found:

- **MatrixStack vs MatrixStackTemplated** — identical teaching content; the only
  difference is a C++ template vs a concrete class, a distinction Python doesn't have.
  → one demo, `MatrixStack`.
- **GameKeyControl vs AdvancedGameKeyControl** — the Advanced version is a strict
  superset (adds input record/replay). → one demo, ported from the Advanced source,
  named `GameKeyControl`.
- **TextureBuffer** — despite its name, this is not a Texture Buffer Object demo; it's
  plain OBJ+PNG texturing, redundant with the existing `Textures`/`SimpleTexture` demos.
  → dropped.
- **ShadedGrid vs GeometryShaders** (the original spec's Phase-4 line item) — both use a
  real OpenGL geometry shader to visualize per-face normals; near-identical technique
  (the two C++ folders even share `normalGeo.glsl`). → one demo, `ShadedGrid`, retiring
  the old spec's separate `GeometryShaders` line.
- **GridVis** — turned out to be 8 build variants benchmarking the same particle-grid sim
  across CPU/SIMD/TBB-threading/GPU-compute, i.e. a C++ concurrency lecture, not a
  graphics-API demo. → dropped from this wave.
- **Sponza** — ships with 127MB of OBJ+texture assets and needs a materially different
  WebGPU implementation (bind-group-per-material vs GL texture units). → dropped from
  this wave; revisit separately if wanted.
- **TextureCompressor** (DXT/S3TC) and **ShadedGrid**/the geometry-shader half of
  **AffineTransforms** have no WebGPU equivalent stage/feature at all — ship OpenGL-only
  rather than build a "reinterpreted" substitute that would teach a different mechanism.

## Final scope (15 demos)

| # | Demo | NGL9Demos source | Backend | Size | Notes |
|---|---|---|---|---|---|
| 1 | MatrixStack | MatrixStack (+Templated, merged) | GL+WebGPU | S | push/pop CPU-side Mat4 stack, trivial both ways |
| 2 | LookAtDemos | LookAtDemos (SimpleLookAt, MultipleViews) | GL+WebGPU | S | combine both sub-demos, camera-select like `Camera` |
| 3 | ViewToWorldTransform | ViewToWorldTransform | GL+WebGPU | M | screen→world unprojection; pure-math core → pytest, à la `RayPickingSelection` |
| 4 | AffineTransforms | AffineTransforms | GL+WebGPU (normal-viz sub-feature GL-only) | M | PySide slider UI driving a `Transform`, à la `ShadingModels`; per old spec's mapping |
| 5 | Spotlight | Spotlight | GL+WebGPU | M | `ngl::SpotLight`-equivalent cone attenuation |
| 6 | ShadedGrid | ShadedGrid | **GL-only** | S | geometry-shader normal visualization; retires old spec's `GeometryShaders` line |
| 7 | Collisions | Collisions (Ray-Sphere, Ray-Triangle, Sphere-Sphere, Sphere-Plane) | GL+WebGPU | M | pure analytic-geometry maths → pytest |
| 8 | TexelFetch | TexelFetch | GL+WebGPU (WebGPU side: storage-buffer read, reinterpreted) | S | TBO doesn't exist in WebGPU |
| 9 | LoadShaderFromJSon | LoadShaderFromJSon | GL+WebGPU (WebGPU side reinterpreted) | S | JSON-driven shader loading; WGSL isn't loaded via ShaderLib's JSON path |
| 10 | TextureCompressor | TextureCompressor | **GL-only** | M | DXT/S3TC desktop-GL feature |
| 11 | GameKeyControl | AdvancedGameKeyControl (supersedes GameKeyControl) | GL+WebGPU | S–M | WASD ship control + record/replay; needs `SpaceShip.obj` |
| 12 | ImageMaze | ImageMaze | GL+WebGPU | S–M | pixel-grid maze render; needs 3 small PNGs |
| 13 | ResetLine | ResetLine | GL+WebGPU (WebGPU reinterpreted — no primitive-restart equivalent) | S | multiple `drawIndexed` calls or degenerate strips on WebGPU |
| 14 | MorphObj | MorphObj | GL+WebGPU | M | vertex-shader blend between 3 obj poses (`BrucePose1-3.obj`) |
| 15 | OctreeAbstract | OctreeAbstract | GL+WebGPU | M | C++ template → plain Python octree class |

**Out of scope for this wave:** MatrixStackTemplated, GameKeyControl (plain),
TextureBuffer, GeometryShaders, GridVis, Sponza (all explained above); anything requiring
a non-portable engine/binding (VulkanSDL, MayaNGL, GLFWNGL, SDLJoyPad, Box2D,
BulletNGL/BulletTower, RVO2NGL, Fluid2D, FacialAnimation, PointBake).

## Conventions

Follow `2026-07-11-new-teaching-demos-design.md` in full: folder layout, executable entry
scripts with the `uv run --script` shebang, `--smoketest` flag verified with
`QT_QPA_PLATFORM=offscreen`, pure-maths modules with pytest coverage (`ViewToWorldTransform`,
`Collisions`), ruff clean, README + root README row per demo, one commit per demo, no edits
to the `ncca.ngl` library.

## Verification

Same as prior waves: `QT_QPA_PLATFORM=offscreen uv run <Demo>/<entry>.py --smoketest` for
every entry script (OpenGL and WebGPU), `uv run pytest <Demo>/tests` where present, `ruff
check`/`ruff format --check` clean. Screenshots cannot be captured headlessly — listed as a
TODO for Jon per demo, same as prior waves.

## Implementation order

Grouped by theme, roughly ascending complexity/port-novelty, so the trickiest WebGPU
reinterpretations (TexelFetch, LoadShaderFromJSon, ResetLine, OctreeAbstract) come after
the per-backend demo pattern is well-established for this wave:

1. **Transform/camera fundamentals:** MatrixStack, LookAtDemos, ViewToWorldTransform, AffineTransforms
2. **Lighting:** Spotlight, ShadedGrid
3. **Collision maths:** Collisions
4. **Texture/shader infrastructure:** TexelFetch, LoadShaderFromJSon, TextureCompressor
5. **Input handling:** GameKeyControl
6. **Applied demo:** ImageMaze
7. **Higher-novelty ports:** ResetLine, MorphObj, OctreeAbstract

Each demo is implemented one at a time in its own worktree/branch
(`agent/<demo-name>`), with a brief check-in before starting each, per Jon's preference
for this wave.
