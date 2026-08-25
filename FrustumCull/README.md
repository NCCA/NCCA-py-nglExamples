# FrustumCull

![FrustumCull](FrustumCull.png)

Culls a 3D grid of spheres against a test camera's view frustum (6-plane
extraction + sphere/plane distance test) and only draws the spheres that are
inside or intersecting. The test camera's frustum is drawn as a yellow
wireframe box, and its eye position is marked with a small cube when viewing
from the observer camera (top-down), so the culling effect is visible from
outside the test camera itself. The title bar shows drawn/total sphere
counts.

## Controls

- `1` : view from test camera.
- `2` : view from observer camera.
- `e`/`l`/`b`/`/` : select what Left/Right/Up/Down/`i`/`o` move on the test
  camera (eye / look / both / slide).
- `Left`/`Right`/`Up`/`Down` : move the test camera on its local x/y axes.
- `i`/`o` : move the test camera in/out along its local z axis.
- `r`/`p`/`y` : roll / pitch / yaw the test camera.
- `+`/`-` : widen / narrow the test camera's field of view.
- `w`/`s` : wireframe / solid fill.
- Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset.

## References

- G. Gribb & K. Hartmann, "Fast Extraction of Viewing Frustum Planes from the World-View-Projection Matrix", 2001 — [PDF](https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf) — the 6-plane extraction method used here.
- [Lighthouse3D — View Frustum Culling](https://www.lighthouse3d.com/tutorials/view-frustum-culling/) — geometric vs radar approaches, and the sphere/plane test.
- [The ryg blog — Frustum planes from the projection matrix](https://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/) — why the row-combination trick works.
