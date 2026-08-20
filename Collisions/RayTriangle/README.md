# RayTriangle

![](RayTriangle.png)

50 randomly-placed triangles (yellow, configurable via `--triangles N`),
re-tested every frame against one ray you move with the keyboard using
`collision_maths.ray_triangle_intersect` (Moller-Trumbore). A hit triangle
is drawn wireframe instead of filled, with a small marker sphere dropped
at the exact hit point. A cube marks each triangle's `v0`. There's no
animation timer here -- the scene only changes when you move the ray.

## Controls
`up`/`down`/`left`/`right` : move the ray's end point
`w`/`z` : move the ray's start point up/down, `a`/`s` : left/right
Left-drag : orbit, Right-drag : pan, Wheel : zoom
