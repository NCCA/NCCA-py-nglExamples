# RaySphere

![](RaySphere.png)

50 randomly-placed spheres (yellow, configurable via `--spheres N`), tested
every tick against 2 animated rays sweeping in opposite `x` directions using
`collision_maths.ray_sphere_intersect`. A sphere hit by either ray is drawn
wireframe instead of filled, with red/green markers dropped at the ray's
near and far intersection points.

## Controls
`space` : pause/resume ray animation, `f` : fullscreen, `n` : windowed
Left-drag : orbit, Right-drag : pan, Wheel : zoom

## WebGPU version

`main_webgpu.py` reproduces the same 50-sphere/2-ray setup independently.
Spheres use the baked `octahedron` mesh; a hit sphere is tinted red
instead of drawn wireframe (wgpu has no practical per-draw polygon-mode
toggle against a pooled pipeline).
