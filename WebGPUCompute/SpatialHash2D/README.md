# WebGPU 2D Spatial Hashing Demo

Particle collision detection on the GPU using a uniform grid spatial hash.
`CollisionCompute.wgsl` runs a multi-pass compute pipeline each frame — clear
grid, count particles per cell, build prefix-sum offsets, fill the grid, then
resolve collisions and integrate — so only particles in neighbouring cells
(3x3) are tested instead of all pairs, giving O(n) rather than O(n²)
collision detection. The grid and cell occupancy counts can be drawn over the
simulation. A 3D version of this demo is in `../SpatialHash3D`.

## Running

```bash
# basic version, -p sets the particle count, --random / --equispaced the layout
./WebGPU2D.py -p 20000

# version with a Qt control panel for the simulation parameters
./WebGPU2DGui.py
```

## Controls

- Left/Right-drag : pan, Wheel : zoom (around cursor)
- Arrow keys : add wind in x / y
- `a` : toggle animation
- `g` : toggle grid display
- `n` : toggle per-cell particle counts
- `space` : reset wind
- `Esc` : quit

## References

- S. Green, "Particle Simulation using CUDA", NVIDIA 2010 — [PDF](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf) — the canonical uniform-grid GPU particle collision pipeline (count / prefix-sum / fill / resolve) this demo follows.
- M. Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV 2003 — [PDF](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf) — spatial hashing for neighbourhood queries.
- [GPU Gems 3, Ch. 39 — Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) — the prefix-sum building block used for cell offsets.
- [Ten Minute Physics (Matthias Müller)](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) — see "Finding collisions among thousands of objects" for a very clear spatial-hash walkthrough.
- [WebGPU Fundamentals — Compute Shader Basics](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) — workgroups and multi-pass compute pipelines.
