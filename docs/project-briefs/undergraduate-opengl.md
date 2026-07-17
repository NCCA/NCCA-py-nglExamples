# Undergraduate Project Briefs — OpenGL

All projects use PyNGL (`ncca.ngl`) with the `BlankPySide6NGL` template unless
stated otherwise. See [README.md](README.md) for common requirements and marking
guidance.

---

## UG-GL1: Particle System Toolkit

**Starting demos**: `Particles/ParticleQuads`, `AnimatedTextures`, `GUIDemos`

Extend the particle quad demo into a designer-facing particle system suitable
for motion-graphics or game-effects work.

**Core deliverables**
- Emitter types: point, disc, and mesh-surface emission.
- Force fields: gravity, wind, vortex, and point attractors, combinable per system.
- Textured sprites with additive and alpha blending, colour/size over lifetime.
- A Qt control panel for live parameter tweaking, with save/load of presets.

**Stretch goals**
- Flipbook (sprite-sheet) animation using the `AnimatedTextures` approach.
- Soft particles (depth-fade against scene geometry).
- Sub-emitters (e.g. sparks spawned on particle death).

**Key concepts assessed**: instanced/batched rendering, blending and depth
interaction, Euler integration, UI-driven tool design.

---

## UG-GL2: Curve-Based Animation Tool

**Starting demos**: `CurveDemos`, `Interpolation`, `ColourSelectionOpenGL`

Build an interactive path-animation tool of the kind found in every DCC package.

**Core deliverables**
- Bezier and Catmull-Rom curve editor with mouse-draggable control points
  (colour-ID picking as in `ColourSelectionOpenGL`).
- Arc-length reparameterisation so objects travel the path at constant speed.
- An object oriented along the curve using a Frenet (or parallel-transport) frame.
- Playback controls: play, pause, scrub, speed.

**Stretch goals**
- Ease-in/ease-out timing curves layered on top of arc length.
- Banking on curves (roll proportional to curvature).
- Export of sampled positions/orientations to a simple keyframe file.

**Key concepts assessed**: parametric curves, arc-length integration, moving
frames, interactive picking.

---

## UG-GL3: Turntable Asset Viewer

**Starting demos**: `ObjViewer`, `ShadingModels`, `PBR`, `SimpleTexture`

Grow the OBJ viewer into a lookdev turntable tool for reviewing assets.

**Core deliverables**
- Load an OBJ with diffuse/normal textures; sensible camera framing of any model.
- Switchable shading: Lambert, Phong/Blinn, and the PBR model from the `PBR` demo.
- Display modes: shaded, wireframe overlay, normals visualisation, UV check grid.
- Turntable rotation with adjustable speed; screenshot export to PNG.

**Stretch goals**
- Environment/light rotation independent of the model.
- Side-by-side A/B comparison of two shading models.
- Basic material editor (roughness/metallic sliders) with live update.

**Key concepts assessed**: mesh loading, shading model theory, texture mapping,
camera mathematics.

---

## UG-GL4: Height-Map Terrain Flyover

**Starting demos**: `ImageHeightMap`, `Camera`, `QuatSlerp`, `FrustumCull`

Create an explorable terrain with a smooth fly-through camera.

**Core deliverables**
- Terrain generated from a height image or Perlin/simplex noise, with correct normals.
- Texture splatting by height and slope (e.g. grass/rock/snow).
- Fly camera with smooth quaternion-based rotation (no gimbal lock).
- Terrain divided into chunks culled with the `FrustumCull` approach; show
  culled-chunk counts on screen.

**Stretch goals**
- Simple distance fog and a skybox.
- Level of detail: lower-resolution meshes for distant chunks.
- A recorded camera path using UG-GL2-style curves.

**Key concepts assessed**: procedural generation, normal computation,
quaternion camera control, spatial culling.

---

## UG-GL5: Keyframe Animation Editor

**Starting demos**: `GUIDemos`, `Interpolation`, `QuatSlerp`, `VAOPrimitives`

Build a miniature keyframe animator: the core of every animation package.

**Core deliverables**
- A timeline widget (Qt) with a scrubbable playhead and keyframe markers.
- Translate/rotate/scale keyframes on multiple scene objects.
- Selectable interpolation per channel: step, linear, ease in/out; slerp for rotation.
- Playback at a fixed frame rate independent of refresh rate.

**Stretch goals**
- A curve editor view showing animation channels as editable 2D curves.
- Onion-skinning (ghosted poses at neighbouring keys).
- Save/load animations to JSON.

**Key concepts assessed**: interpolation theory, quaternion rotation, time
management, model/view UI design.

---

## UG-GL6: Voxel Sculpting Sandbox

**Starting demos**: `Voxels`, `ColourSelectionOpenGL`, `Obj2Numpy`

Extend the voxel demo into a small MagicaVoxel-style editor.

**Core deliverables**
- Add and remove voxels with mouse picking (ray-cast or colour-ID).
- Per-voxel colour painting from a palette.
- Only render voxel faces not hidden by neighbours (face culling within the grid).
- Save/load the voxel grid via NumPy arrays (see `Obj2Numpy` for the plumbing).

**Stretch goals**
- Cheap ambient-occlusion approximation from neighbour occupancy.
- Mirror-mode editing and box-fill tools.
- Export the visible surface as an OBJ.

**Key concepts assessed**: 3D grids and indexing, picking, mesh generation from
volumes, file I/O with NumPy.

---

## UG-GL7: 2D Game or Motion-Graphics Scene

**Starting demos**: `2DDrawingOpenGL`, `FontRendering`, `AnimatedTextures`

For students stronger on gameplay or design than rendering theory: a complete,
polished 2D interactive piece.

**Core deliverables**
- Sprite batching with texture atlases; at least three parallax background layers.
- Animated sprites (flipbook) and a text HUD using `FontRendering`.
- Simple physics: gravity/velocity plus AABB or circle collision.
- A complete interaction loop (win/lose state, or a looping motion-graphics
  sequence with timed events).

**Stretch goals**
- Screen-space effects: shake, flash, simple 2D particle bursts.
- Gamepad input; sound via Qt multimedia.

**Key concepts assessed**: 2D transforms, batching, game loop and timing,
collision basics.

---

## UG-GL8: Post-Processing Stack

**Starting demos**: `FBODemos/SimpleFBO`, `FBODemos/DOF`, `FBODemos/Blit`, `ScreenTri`

Build a chainable post-processing system of the kind found in every engine.

**Core deliverables**
- Render the scene to an FBO and pass it through a chain of full-screen effects.
- At least four effects: bloom (bright-pass + blur), vignette, colour grading
  via 3D LUT or curves, and one of edge detection / pixelation / chromatic
  aberration.
- A Qt panel to toggle, reorder, and parameterise each pass.
- Correct handling of window resize.

**Stretch goals**
- Depth-based effects using the depth attachment (fog, DOF refinement).
- Half-resolution intermediate targets for the blur chain, with a measured
  performance comparison.

**Key concepts assessed**: framebuffers and render targets, full-screen
shader passes, ping-pong rendering, colour theory.
