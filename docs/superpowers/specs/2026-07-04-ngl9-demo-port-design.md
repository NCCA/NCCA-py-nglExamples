# Design: Port NGL9Demos to PyNGLDemos (batch 1)

## Goal

Fill the gap between the C++ NGL9Demos collection (`/Volumes/teaching/NGL9Demos`) and this
repo by porting 23 demos that are currently missing, following this repo's existing
PySide6/QOpenGLWindow conventions and the `ncca.ngl` Python library.

## Scope

The 11 demos already listed in `TODO.md` that aren't done yet, plus 12 additional demos
selected from NGL9Demos:

TODO.md carryover: Camera, ColourObj, CurveDemos, Instancing, KleinBottle, MassSpring,
MorphObj, OctreeAbstract, QuatSlerp, Sponza, AffineTransforms (Transforms with GUI).

Additional: FrustumCull, TextureBuffer+TexelFetch (combined), GeometryShaders,
TessellationShader, SpatialHash, BVH, Interpolation, ImageHeightMap, RayMarching,
GameKeyControl, PointCloud, AnimatedTextures.

Out of scope: anything requiring a non-portable engine/binding (VulkanSDL, MayaNGL, GLFWNGL,
SDLJoyPad, Box2D, BulletNGL/BulletTower, RVO2NGL) and WebGPU ports (this batch targets OpenGL
only, matching the C++ source).

## Conventions

Each demo is a self-contained top-level folder, following `VAOPrimitives`/`NormalMapping`:

- `<DemoName>/main.py` — executable (`chmod +x`), shebang
  `#!/usr/bin/env -S uv run --script`, a `QOpenGLWindow` subclass implementing
  `initializeGL`/`paintGL`/`resizeGL`, with the standard mouse orbit/pan/zoom + keyboard
  handling copied from the VAOPrimitives skeleton and adapted per demo.
- `<DemoName>/README.md` — adapted from the corresponding NGL9Demos README (same
  description/image-link style).
- `<DemoName>/shaders/*.glsl` — GLSL sources, ported from the C++ demo's `shaders/`.
- `<DemoName>/models/`, `<DemoName>/textures/` — only where the demo needs them, copied
  from the source demo's assets.
- No preview `.png` is required up front (can be added later by running the demo and
  screenshotting).
- RunDemos.py needs no changes — it auto-discovers any executable `.py` under the repo root.

## Library boundary

No changes to the `ncca.ngl` package (a sibling project). `ShaderLib.load_shader()` only
wraps vertex/fragment/geometry. Demos needing tessellation, compute shaders, or texture
buffer objects — none of which `ShaderLib` wraps — build their `Shader`/`ShaderProgram`
objects directly using the already-exported low-level classes (`Shader`, `ShaderProgram`,
`ShaderType`), and issue raw `OpenGL.GL` calls for TBO setup. This keeps every demo
self-contained within PyNGLDemos.

## Per-demo mapping

| Demo | NGL9Demos source | Notes |
|---|---|---|
| Camera | Camera | reuse `FirstPersonCamera`, `look_at` |
| ColourObj | ColourObj | `Obj` loading + per-face colour shader |
| CurveDemos | CurveDemos/CurveDemo | reuse `BezierCurve` |
| Instancing | Instancing/DivisorInstancing + InstanceMeshes | one demo: `glVertexAttribDivisor` per-instance transforms |
| KleinBottle | KleinBottle | parametric mesh generation |
| MassSpring | MassSpring | particle/spring simulation; UI ported as inline PySide widgets, not a `.ui` file |
| MorphObj | MorphObj | vertex-blend shader between mesh targets |
| OctreeAbstract | OctreeAbstract | octree build + wireframe visualization |
| QuatSlerp | QuatSlerp | reuse `Quaternion.slerp` |
| Sponza | Sponza | assets (127MB) copied in full into `Sponza/models`, `Sponza/textures` |
| AffineTransforms | AffineTransforms | PySide `.ui` sliders driving a `Transform`, à la ShadingModels |
| FrustumCull | FrustumCull | bounding-volume vs frustum-plane test, visualize culled vs visible |
| TextureBuffer + TexelFetch | TextureBuffer, TexelFetch | combined demo; manual TBO setup, `texelFetch` sampling in shader |
| GeometryShaders | GeometryShaders/Normals | normal-visualization geometry shader |
| TessellationShader | TessellationShader | manual tesc/tese `ShaderProgram` assembly |
| SpatialHash | SpatialHash | broad-phase neighbor query visualization |
| BVH | BVH | build + wireframe-draw a BVH over scene objects |
| Interpolation | Interpolation | easing/interpolation technique comparison |
| ImageHeightMap | ImageHeightMap | heightmap-from-image terrain mesh |
| RayMarching | RayMarching | fullscreen-tri SDF fragment shader, à la ScreenTri |
| GameKeyControl | GameKeyControl | WASD-style object control |
| PointCloud | PointCloud | `GL_POINTS` render of a point dataset |
| AnimatedTextures | AnimatedTextures | flipbook/UV-scroll texture animation |

## Verification

Each demo is smoke-tested non-interactively with `QT_QPA_PLATFORM=offscreen` to confirm it
initializes, renders at least one frame, and exits cleanly — catching import/shader-compile
errors. This is not a substitute for visually reviewing the demo; the user should run each
interactively afterward.

## Implementation order

Given the batch size (23 demos), implementation proceeds in phases grouped by shared
technique/risk, so problems in one phase (e.g. shader-stage plumbing) don't block unrelated
demos:

1. **Straightforward reuse** (low risk, existing library features only): Camera, ColourObj,
   CurveDemos, QuatSlerp, KleinBottle, FrustumCull, PointCloud, AnimatedTextures,
   Interpolation, ImageHeightMap.
2. **GUI-driven**: AffineTransforms, MassSpring, GameKeyControl.
3. **Structural/algorithmic**: OctreeAbstract, BVH, SpatialHash, Instancing, MorphObj.
4. **Advanced shader stages** (needs manual `Shader`/`ShaderProgram` assembly):
   GeometryShaders, TessellationShader, TextureBuffer+TexelFetch, RayMarching.
5. **Large-asset**: Sponza.
