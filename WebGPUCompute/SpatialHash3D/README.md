# WebGPU 3D Spatial Hashing Demo

![](SpatialHash3D.png)

The 3D version of [`../SpatialHash2D`](../SpatialHash2D). `CollisionCompute3D.wgsl` runs the same multi-pass compute pipeline each frame — clear grid, count particles per cell, build prefix-sum offsets, fill the grid, then resolve collisions and integrate — but over a 3D uniform grid, so each particle now tests the 27 neighbouring cells (3x3x3) rather than 9. Positions and velocities are `vec3` and the boundaries wrap in all three axes.

Each particle is drawn as an instanced diffuse-lit sphere (one shared `PrimData.sphere()` mesh, a single draw call) with a standard rotate / zoom camera, and the spatial hash grid can be drawn over the simulation.

## Running

```bash
# basic version, -p sets the particle count, --random / --equispaced the layout
uv run WebGPUCompute/SpatialHash3D/WebGPU3D.py -p 5000

# version with a Qt control panel for the simulation parameters
uv run WebGPUCompute/SpatialHash3D/WebGPU3DGui.py
```

## Controls

- Left-drag : rotate, Wheel : zoom
- Arrow keys : add wind in x / y, Page Up/Down : wind in z
- `a` : toggle animation
- `g` : toggle grid display
- `space` : reset camera and wind
- `Esc` : quit

The GUI version puts the particle count, distribution, simulation size, cell size, particle radius, wind and camera on a control panel instead.

## References

- S. Green, "Particle Simulation using CUDA", NVIDIA 2010 — [PDF](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf) — the canonical uniform-grid GPU particle collision pipeline, in 3D as here.
- M. Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV 2003 — [PDF](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) — the 3D spatial-hash function family.
- [GPU Gems 3, Ch. 39 — Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) — the prefix-sum building block used for cell offsets.
- [Ten Minute Physics (Matthias Müller)](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) — spatial hashing explained from first principles.
- [WebGPU Fundamentals — Compute Shader Basics](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) — workgroups and multi-pass compute pipelines.
