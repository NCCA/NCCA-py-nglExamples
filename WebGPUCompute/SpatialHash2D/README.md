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
