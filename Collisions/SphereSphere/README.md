# SphereSphere

![](SphereSphere.png)

Two large, static spheres (yellow) and two small spheres (red, blue) that
move toward each other and bounce apart on collision -- with each other
and with the static spheres -- using an analytic sphere/sphere overlap
test (`collision_maths.sphere_sphere_collide`).

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset

## WebGPU version

`main_webgpu.py` reproduces the same 4-sphere setup and collision rules
independently. Spheres are drawn as the baked `octahedron` mesh (WebGPU
has no runtime sphere primitive here).
