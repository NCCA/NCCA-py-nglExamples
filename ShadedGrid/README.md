# ShadedGrid

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
