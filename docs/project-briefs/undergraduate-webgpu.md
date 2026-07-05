# Undergraduate Project Briefs — WebGPU

All projects use the PyNGL WebGPU stack (`ncca.ngl.webgpu`, wgpu-py) with the
`BlankWebGPU` template unless stated otherwise. See [README.md](README.md) for
common requirements and marking guidance.

---

## UG-WG1: WGSL Shader Playground

**Starting demos**: `BlankWebGPU`, `ScreenTri`, `GUIDemos`

Build a Shadertoy-style live-coding tool for full-screen fragment shaders.

**Core deliverables**
- Full-screen triangle rendering a user-supplied WGSL fragment shader.
- Live reload: recompile on file save or from an in-app editor, without
  restarting; keep the last working shader on compile failure.
- Standard uniforms: time, resolution, mouse position, frame number.
- WGSL error messages surfaced clearly in the UI.

**Stretch goals**
- User-defined uniform sliders generated from comments in the shader source.
- Texture channel inputs (image files bound as sampled textures).
- Multi-pass support (previous frame as an input texture).

**Key concepts assessed**: WebGPU pipeline creation, WGSL, uniform buffers and
bind groups, robust error handling.

---

## UG-WG2: Compute-Shader Particle System

**Starting demos**: `SimpleComputeWebGPU`, `WebGPUCompute`, `Particles/ParticleQuads`

Simulate a large particle system entirely on the GPU.

**Core deliverables**
- Particle state (position, velocity, age, colour) in storage buffers, updated
  by a compute shader; target 100k+ particles.
- Rendering as instanced quads or points directly from the storage buffer —
  no CPU round trip.
- Mouse-driven attractor/repulsor forces.
- Frame-time comparison against the CPU-updated OpenGL `ParticleQuads` demo at
  matched particle counts, presented as a graph.

**Stretch goals**
- Particle respawn/emission controlled on the GPU.
- Colour mapped from speed or age; additive blending.
- Simple collision with a ground plane or sphere.

**Key concepts assessed**: compute pipelines, storage buffers, workgroup sizing,
GPU/CPU performance analysis.

---

## UG-WG3: GPU Boids Flocking

**Starting demos**: `WebGPUCompute`, `WebGPUMultiGeo`

Implement Reynolds' boids on the GPU and render an oriented flock.

**Core deliverables**
- Separation, alignment, and cohesion computed in a compute shader
  (brute-force neighbour search is acceptable at this level).
- Boids rendered as instanced oriented cones/darts using the instancing
  pattern from `WebGPUMultiGeo`, aligned to velocity.
- Runtime-adjustable weights and radii via the UI.
- Boundary handling: wrap-around or steering back into a bounding volume.

**Stretch goals**
- Predator/prey behaviour or obstacle avoidance.
- A study of flock size vs frame time, identifying the O(n²) wall.

**Key concepts assessed**: agent-based simulation, compute shaders, instancing,
orientation from direction vectors.

---

## UG-WG4: Game of Life / Reaction-Diffusion

**Starting demos**: `WebGPUCompute`, `FBODemos/WebGPURenderToTexture`, `ScreenTri`

A cellular-automata or reaction-diffusion simulator: small in scope,
visually striking, and an excellent introduction to GPGPU texture techniques.

**Core deliverables**
- Simulation state in ping-ponged storage textures updated by a compute shader.
- Either Conway's Game of Life (with at least one rule variant) or Gray-Scott
  reaction-diffusion with adjustable feed/kill rates.
- Mouse painting to seed the simulation interactively.
- A colour-mapping display pass (not raw state values).

**Stretch goals**
- Multiple presets illustrating distinct Gray-Scott regimes.
- Simulation resolution independent of window resolution.
- Speed control: multiple simulation steps per displayed frame.

**Key concepts assessed**: storage textures, ping-pong technique, texture
sampling vs loading, emergent systems.

---

## UG-WG5: Shadowed Diorama Scene

**Starting demos**: `WebGPUShadows` (including `DepthDisplayShader.wgsl`),
`WebGPUMultiGeo`

Compose and light a small scene with real-time shadow maps.

**Core deliverables**
- A composed diorama of at least five distinct objects with varied materials.
- Shadow mapping from a directional or spot light, with PCF filtering and
  tuned bias (demonstrate acne and peter-panning, then fix them).
- An animated light with correctly moving shadows.
- The depth-map debug view (from `DepthDisplayShader`) toggleable as a
  diagnostic overlay.

**Stretch goals**
- A second shadow-casting light.
- Shadow-map resolution comparison with screenshots and frame times.

**Key concepts assessed**: depth-only render passes, light-space transforms,
shadow artefacts and mitigation, scene composition.
