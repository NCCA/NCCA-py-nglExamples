# ShadedGrid

![](ShadedGrid.png)

An animated wave-height grid, Phong-shaded with 3-point lighting, with a
geometry-shader pass drawn on top that visualizes each triangle's face
normal (red) and each vertex's normal (yellow) as line segments — watch
them rotate and stretch as the surface undulates. Normals use the standard
heightfield central-difference formula, correct at every edge (the
NGL9Demos C++ original's per-vertex neighbour method left most of the grid
boundary with degenerate normals).

## Controls
- `1` : toggle face-normal lines
- `2` : toggle vertex-normal lines
- `+` / `-` : normal line length
- `u` : toggle wave animation
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset

## WebGPU version

`main_webgpu.py` is a separate entry point built on `wgpu-py` rather than
PyOpenGL. It reuses the same `_wave_heights`/`build_wave_grid` maths as
`main.py` — same heightfield, same central-difference normals — and rebuilds
the grid on the CPU every animation tick, pushing it to the GPU with
`queue.write_buffer()` rather than recreating the vertex buffer. Lighting is
a WGSL port of `shaders/PhongVertex.glsl`/`PhongFragment.glsl`: three point
lights, each with ambient/diffuse/specular, done in world space so the
normal matrix is the inverse-transpose of the model matrix alone.

It deliberately leaves out the normal-visualization pass — WebGPU has no
geometry-shader stage, and a compute-shader rewrite of that one feature
wasn't worth the scope for what is otherwise a straight port. So `1`, `2`
and `+`/`-` above don't apply here; `u` to toggle the animation and the
orbit/pan/zoom/reset controls all work the same.
