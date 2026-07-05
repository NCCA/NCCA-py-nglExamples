# Postgraduate Project Briefs — OpenGL

These briefs assume undergraduate-level graphics fundamentals and are sized for
a masters project with a substantial evaluation component. All use PyNGL
(`ncca.ngl`) with the `BlankPySide6NGL` template. See [README.md](README.md)
for common requirements and marking guidance — note the heavier evaluation
weighting at PG level.

---

## PG-GL1: Deferred PBR Pipeline

**Starting demos**: `DefferedLighting`, `PBR`, `NormalMapping`, `FBODemos`

Merge the deferred and PBR demos into a modern many-light pipeline.

**Core deliverables**
- G-buffer (albedo, normal, roughness/metallic, depth) with tangent-space
  normal mapping.
- Cook-Torrance PBR lighting resolved in a deferred pass.
- Many-light support (hundreds) with tiled or clustered light culling.
- HDR intermediate targets with a tone-mapping pass.
- Evaluation: light-count scaling of deferred vs forward rendering, and the
  effect of tile size on cull efficiency.

**Stretch goals**
- SSAO from the G-buffer; bloom on the HDR target.
- Image-based ambient lighting from an environment map.

**Key concepts assessed**: render-pass architecture, G-buffer design and
bandwidth trade-offs, physically based shading, performance methodology.

---

## PG-GL2: Skeletal Animation System

**Starting demos**: `SimplePyNGL/WithQuat.py`, `QuatSlerp`, `ObjViewer`

Build a skinned-character playback system from the maths up.

**Core deliverables**
- Load a rigged, skinned mesh (a documented custom JSON/NumPy format exported
  from a DCC is acceptable and recommended).
- GPU skinning in the vertex shader; joint hierarchy evaluated per frame.
- Both linear blend skinning and dual-quaternion skinning, switchable.
- Animation clip playback with looping and time scaling; blending between
  two clips.
- Evaluation: quantitative comparison of candy-wrapper/volume-loss artefacts
  between the two skinning methods (e.g. volume measurement on a twisting limb).

**Stretch goals**
- A simple animation state machine (idle/walk/run) with transition blending.
- Additive animation layers.

**Key concepts assessed**: transform hierarchies, quaternion mathematics,
skinning theory, data pipeline design.

---

## PG-GL3: Position-Based Dynamics Cloth

**Starting demos**: `VAOPrimitives`, `PBR`, `GUIDemos`

Implement Müller-style position-based dynamics for cloth and simple soft bodies.

**Core deliverables**
- PBD solver with distance and bending constraints on a cloth grid.
- Collision with spheres and planes; pinned-vertex attachments.
- Per-frame normal recomputation and PBR-shaded rendering.
- Interactive dragging of cloth points.
- Evaluation: stability and stiffness behaviour vs iteration count and time
  step, with graphs; comparison against a basic mass-spring integrator.

**Stretch goals**
- Self-collision via spatial hashing.
- Tearing when constraints exceed a strain threshold.
- A soft-body ball using volume constraints.

**Key concepts assessed**: constraint-based simulation, numerical stability,
collision response, evaluation design.

---

## PG-GL4: Procedural City or Landscape Generator

**Starting demos**: `ImageHeightMap`, `FrustumCull`, `PointCloud`, `VAOPrimitives`

A procedural environment generator with the rendering machinery to display it
at scale.

**Core deliverables**
- Procedural layout: L-system road network, wave-function-collapse tiles, or
  noise-based landscape zoning — student's choice, justified in the report.
- Instanced rendering of building/vegetation geometry with frustum culling.
- At least two LOD levels selected by distance.
- Deterministic generation from a seed.
- Evaluation: draw-call and memory strategies (naive vs instanced vs merged
  meshes) measured at increasing world sizes.

**Stretch goals**
- `PointCloud`-style scattered detail (grass, debris) with density maps.
- Day/night lighting cycle.

**Key concepts assessed**: procedural generation algorithms, instancing, LOD,
scalability analysis.

---

## PG-GL5: Real-Time Fluid Simulation

**Starting demos**: `FBODemos`, `ScreenTri`, `Particles/ParticleQuads`

Implement one of the two classic real-time fluid approaches.

**Option A — 2D stable fluids (Stam)**: advection, diffusion, and pressure
projection implemented as fragment-shader passes over ping-ponged FBO textures;
dye injection and forces from mouse interaction.

**Option B — 3D SPH**: CPU SPH with spatial-hash neighbour search; density,
pressure, and viscosity forces; rendered as particles or screen-space
metaballs via an FBO pass.

**Core deliverables (either option)**
- A stable, interactive simulation with runtime-adjustable parameters.
- A clear write-up of the governing equations and their discretisation.
- Evaluation: resolution/particle-count vs frame time; visual comparison of
  parameter regimes; where applicable, solver iterations vs divergence.

**Stretch goals**
- Option A: vorticity confinement; obstacles in the domain.
- Option B: surface tension; NumPy vectorisation study of the neighbour search.

**Key concepts assessed**: PDE discretisation or particle hydrodynamics,
multi-pass GPU techniques, numerical analysis.

---

## PG-GL6: Inverse Kinematics and Motion Editing

**Starting demos**: `QuatSlerp`, `CurveDemos`, `Interpolation`, `VAOPrimitives`

Build an interactive IK rig and study solver behaviour.

**Core deliverables**
- An articulated joint chain (arm/leg/tail) with interactive target dragging.
- Two solvers: CCD and FABRIK, switchable at runtime.
- Joint rotation limits; pole-vector control for chain orientation.
- Locomotion along a `CurveDemos` path with IK-planted feet on uneven ground.
- Evaluation: convergence rate, iteration counts, and failure cases of the two
  solvers across reachable and unreachable targets.

**Stretch goals**
- A Jacobian-transpose solver as a third comparison point.
- Full-body: two legs plus pelvis height solving.

**Key concepts assessed**: kinematic chains, iterative solvers, constraint
handling, character-technical-direction thinking.

---

## PG-GL7: Large Point-Cloud Viewer

**Starting demos**: `PointCloud`, `Camera`, `FrustumCull`

A viewer for large scanned point clouds, aimed at students interested in
scanning and photogrammetry pipelines.

**Core deliverables**
- Load large clouds (PLY/XYZ, millions of points) into NumPy-backed buffers.
- Octree spatial index with frustum culling and level-of-detail point budgets.
- Normal estimation (PCA over k-nearest neighbours) for lit rendering.
- Eye-dome lighting or splat-based shading for depth perception.
- Evaluation: points-per-frame budget vs frame time and visual quality; octree
  build times vs depth.

**Stretch goals**
- Out-of-core streaming of octree nodes.
- Measurement tools (point picking, distances).

**Key concepts assessed**: spatial data structures, LOD strategies, geometry
processing, handling real-world data at scale.
