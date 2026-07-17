# Postgraduate Project Briefs — WebGPU

These briefs target the PyNGL WebGPU stack (`ncca.ngl.webgpu`, wgpu-py, WGSL)
and are sized for a masters project with a substantial evaluation component.
See [README.md](README.md) for common requirements and marking guidance.

---

## PG-WG1: GPU Path Tracer in Compute

**Starting demos**: `WebGPUCompute`, `ScreenTri`, `Obj2Numpy`

A progressive path tracer written in WGSL compute — strong dissertation
material with clear, quantifiable evaluation.

**Core deliverables**
- Compute-shader path tracing of triangle meshes with a BVH built on the CPU
  (mesh data via the `Obj2Numpy` pipeline) and traversed in WGSL.
- Progressive accumulation into a storage texture, reset on camera move;
  tone-mapped display via a `ScreenTri` pass.
- Lambertian and metallic/specular materials; emissive lights.
- Next-event estimation (direct light sampling) with a demonstrated
  convergence improvement.
- Evaluation: samples-per-second vs scene complexity; convergence (RMSE vs a
  long-run reference image) with and without NEE; BVH quality metrics.

**Stretch goals**
- Russian roulette termination; a microfacet BRDF; depth of field.
- Wavefront (multi-kernel) architecture compared against a megakernel.

**Key concepts assessed**: Monte Carlo rendering theory, acceleration
structures, GPU memory layout, rigorous quantitative evaluation.

---

## PG-WG2: GPU-Driven Rendering

**Starting demos**: `WebGPUMultiGeo`, `WebGPUCompute`, `FrustumCull`

Move scene traversal onto the GPU: a very current, industry-relevant topic.

**Core deliverables**
- A scene of 10k–100k objects with per-object transforms in storage buffers.
- Compute-shader frustum culling writing survivors to a compact draw list.
- Indirect draws (`draw_indirect`) consuming the GPU-built list — no per-object
  CPU work per frame.
- Evaluation: CPU frame cost and total frame time, CPU-driven vs GPU-driven,
  across object counts; cull efficiency statistics read back for display.

**Stretch goals**
- Two-level LOD selection in the culling shader.
- Occlusion culling with a hierarchical depth pyramid from the previous frame.

**Key concepts assessed**: indirect drawing, GPU scene representation, compute
compaction patterns, CPU/GPU workload analysis.

---

## PG-WG3: SPH Fluid on the GPU

**Starting demos**: `WebGPUCompute`, `SimpleComputeWebGPU`

A complete smoothed-particle-hydrodynamics fluid living entirely in GPU memory.

**Core deliverables**
- Uniform-grid neighbour search built each frame in compute (counting sort or
  atomics-based binning).
- Density, pressure (equation of state), and viscosity kernels; boundary
  handling against a container.
- 50k+ particles interactive; parameters adjustable at runtime.
- Evaluation: particle count vs frame time with a breakdown per stage
  (binning / density / forces / integration); comparison against the CPU SPH
  option in the OpenGL brief if attempted by a cohort.

**Stretch goals**
- Screen-space fluid surface rendering (depth smoothing + normals) instead of
  raw particles.
- PCISPH or DFSPH for larger stable time steps, with a stability study.

**Key concepts assessed**: GPU sorting/binning, atomics, particle
hydrodynamics, multi-kernel pipeline design.

---

## PG-WG4: GPU Crowd Simulation

**Starting demos**: `WebGPUCompute`, `WebGPUMultiGeo`

Large-scale agent simulation with visual analysis — appeals strongly to
animation students.

**Core deliverables**
- Thousands of agents updated in compute: social-force or simplified RVO
  local avoidance.
- Global navigation via a flow field (gradient of a grid distance transform)
  toward one or more goals.
- Instanced rendering of agents with orientation and simple animation
  (e.g. bobbing/phase-offset walk imposters).
- A density heatmap visualisation mode for congestion analysis.
- Evaluation: agent count vs frame time; emergent behaviour analysis (lane
  formation, arching at exits) against published crowd literature.

**Stretch goals**
- Multiple agent types/goals; doorway and bottleneck scenarios.
- Trajectory recording and playback for analysis.

**Key concepts assessed**: agent simulation at scale, pathfinding, instancing,
connecting results to research literature.

---

## PG-WG5: Grass and Strand Rendering

**Starting demos**: `WebGPUCompute`, `WebGPUMultiGeo`, `ImageHeightMap`

Real-time strand rendering (grass, fur, stylised hair) bridging DCC-tool
thinking and real-time constraints.

**Core deliverables**
- Compute-generated strands (hundreds of thousands of blades) scattered over a
  ground mesh with density/length maps.
- Strands expanded to camera-facing or view-dependent ribbons; per-strand
  variation in height, tilt, and colour.
- Procedural wind (layered noise) and local interaction (a sphere pushing
  strands aside).
- Evaluation: strand count vs frame time; vertex- vs compute-expansion
  comparison; LOD (density fade with distance) quality/performance trade-off.

**Stretch goals**
- Bezier-curved blades with correct normals for lighting.
- Shadowing/self-occlusion approximation for grounded lighting.

**Key concepts assessed**: massive instancing/generation on GPU, procedural
animation, LOD design, art-direction of a technical system.

---

## PG-WG6: Clipmap or Virtually-Textured Terrain

**Starting demos**: `ImageHeightMap`, `WebGPUCompute`, `FBODemos/WebGPURenderToTexture`

A systems-heavy large-terrain renderer for engine-oriented students.

**Core deliverables**
- A terrain far larger than GPU memory would allow naively, streamed as tiles.
- Either geometry clipmaps (nested rings following the camera) or a virtual
  texture with a compute-managed page table and per-frame feedback.
- Asynchronous tile loading without frame hitches (measure and show it).
- Debug visualisations: tile/mip boundaries, residency map.
- Evaluation: memory residency vs view distance; frame-time stability during
  fast camera movement; page-fault/upload statistics.

**Stretch goals**
- Compressed on-disk tiles; height and albedo streamed independently.
- Crack-free stitching between clipmap levels, demonstrated.

**Key concepts assessed**: streaming architecture, texture memory management,
compute-driven bookkeeping, systems evaluation.

---

## PG-WG7: Cross-API Renderer Abstraction

**Starting demos**: the paired demos in both APIs — `Particles/ParticleQuads` /
`WebGPUCompute`, `WebGPUShadows`, `FBODemos/SimpleFBO` /
`FBODemos/WebGPURenderToTexture`, plus the `ncca.ngl.opengl` / `ncca.ngl.webgpu`
split itself

Design a small scene-graph renderer with interchangeable OpenGL and WebGPU
backends — uniquely well supported by this codebase, since both stacks already
share the maths and mesh layer. Also the formalised "port and compare" option
for students who want depth over novelty.

**Core deliverables**
- A backend-agnostic API (scene graph, mesh, material, camera, light) with no
  GL or wgpu types leaking through the interface.
- Both backends rendering the same demo scene: textured meshes, at least one
  light, render-to-texture.
- GLSL and WGSL shader pairs with a documented strategy for keeping them in
  sync (translation, generation, or disciplined pairing).
- Evaluation: an honest API-design write-up — where the abstraction fit
  naturally, where it leaked (render passes, bind groups vs uniforms, coordinate
  conventions) — plus performance parity measurements on identical scenes.

**Stretch goals**
- A third tiny backend target (e.g. a headless image writer) to stress the
  abstraction.
- Hot-swapping backends at runtime.

**Key concepts assessed**: software architecture, API design trade-offs, deep
understanding of both graphics APIs, comparative evaluation.
