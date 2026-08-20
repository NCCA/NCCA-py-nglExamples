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
