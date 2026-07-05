# FrustumCull

Culls a 3D grid of spheres against a test camera's view frustum (6-plane
extraction + sphere/plane distance test) and only draws the spheres that are
inside or intersecting. View from the observer camera (top-down) to see the
effect; the title bar shows drawn/total sphere counts.

## Controls
`1` : view from test camera, `2` : view from observer camera
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
