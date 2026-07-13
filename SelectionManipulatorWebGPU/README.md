# Selection and Manipulation (WebGPU)

![Screenshot](SelectionManipulatorWebGPU.png)

A WebGPU port of the [`SelectionManipulator`](../SelectionManipulator) OpenGL
demo: DCC style object selection and manipulation in the spirit of Maya or
Houdini. A small scene of objects can be picked with the mouse and transformed
with visual Translate / Rotate / Scale gizmos.

```bash
uv run main.py          # or ./main.py
```

## Controls (Maya-style)

| Input                | Action                                                     |
| -------------------- | ---------------------------------------------------------- |
| `Q`                  | Select mode (gizmo hidden)                                 |
| `W`                  | Translate mode (arrows)                                    |
| `E`                  | Rotate mode (rings)                                        |
| `R`                  | Scale mode (boxes)                                         |
| Left click           | Select the object under the cursor (replaces selection)    |
| `Ctrl` + click       | Toggle an object in / out of the selection (multi-select)  |
| Drag an axis handle  | Transform **all** selected objects along that axis         |
| Drag the centre cube | Free screen-plane move (translate) / uniform scale (scale) |
| `Alt` + LMB drag     | Tumble the camera                                          |
| `Alt` + RMB drag     | Pan the camera                                             |
| Mouse wheel          | Dolly in / out                                             |
| `Space`              | Reset the camera                                           |
| `Escape`             | Quit                                                       |

## How it uses the WebGPU stack

The demo leans on the stock `ncca.ngl.webgpu` factory pipelines wherever they
fit, and adds one small custom pipeline only where the factory can't help:

| Part of the scene             | Pipeline                                                |
| ----------------------------- | ------------------------------------------------------- |
| Ground grid                   | factory `SINGLE_COLOUR_LINES`                           |
| Gizmo handles (display)       | factory `SINGLE_COLOUR_TRIANGLES`, one pass per part    |
| Picking – gizmo handles       | factory `SINGLE_COLOUR_TRIANGLES` (reserved ID colours) |
| Picking – objects             | the object pipeline in flat "pick" mode                 |
| Objects (diffuse + wireframe) | custom `ObjectPipeline` (`ObjectShader.wgsl`)           |

## What changed from the OpenGL version

The scene, the `SelectionObject` hierarchy, the Maya controls, and all of the
manipulator drag maths are unchanged. Three things that OpenGL does with fixed
function state have no direct WebGPU equivalent and are done differently:

### Wireframe overdraw :- single-pass barycentric

OpenGL redraws selected objects with `glPolygonMode(GL_LINE)` and a negative
polygon offset. WebGPU has no polygon line mode, so `ObjectShader.wgsl` draws
the fill and the wireframe in **one pass**: the geometry is a non-indexed
triangle soup, so `vertex_index % 3` identifies each triangle corner and emits
a barycentric coordinate `(1,0,0)/(0,1,0)/(0,0,1)`. These interpolate across the
triangle, and the **smallest** of the three components is the (barycentric)
distance to the nearest edge — zero on an edge, largest at the centroid. No
extra vertex buffer is needed.

### Adaptive wireframe (thin lines for dense meshes)

A naive `edge < constant` test draws lines that are thick where a triangle is
big on screen and thin where it is small — and for a dense mesh (the teapot has
~14k triangles) every triangle is smaller than a pixel, so _every_ fragment is
within one line-width of an edge and the object washes out to solid white. The
shader fixes both problems with two screen-space quantities derived from the
edge distance:

1. **Constant-thickness lines.** `fwidth(edge)` is how fast the edge distance
   changes per pixel, so

   ```wgsl
   let dist_px = edge / fwidth(edge);   // distance to the edge, in pixels
   var wire = 1.0 - smoothstep(thickness, thickness + 1.0, dist_px);
   ```

   makes the line a fixed `thickness` pixels wide (plus one pixel of
   anti-aliasing) regardless of how far away or foreshortened the triangle is —
   a near face and a far face of the same object get the same line weight.

2. **Density fade.** `1.0 / fwidth(edge)` approximates how many pixels the
   triangle spans on screen. When that extent drops to only a few pixels the
   triangle can no longer show a readable line, so the wire is faded out:

   ```wgsl
   let extent_px = 1.0 / fwidth(edge);
   wire = wire * smoothstep(4.0, 14.0, extent_px);   // 0 below ~4px, full by ~14px
   ```

The means that coarse meshes (cube, dodecahedron, sphere) keep a crisp thin
wireframe, while dense meshes (teapot, troll) keep their shaded surface and only
show wireframe on the larger triangles near their silhouette — so a selection is
still visible without the object turning into a white blob. The `4/14`px
thresholds and the `0.6`px `thickness` are the two parameters to tune.

### `glReadPixels` picking :- offscreen render + readback

On click the scene is rendered flat (each object in its unique colour ID, the
gizmo handles in reserved ID colours on top) into an **offscreen** MSAA texture
that resolves to a copyable target. That target is copied to a mapped buffer
with `copy_texture_to_buffer`, and a 9×9 pixel block under the cursor is read
back on the CPU. Handle colours are matched first (so handles stay grabbable in
front of geometry), then object IDs. The object pass reuses the object pipeline
with a `render_mode` uniform that switches the fragment shader to flat pick
colour — no lighting, no wireframe.

### `glClear(GL_DEPTH_BUFFER_BIT)` gizmo overdraw :- a second pass

To keep the gizmo on top of the scene, each handle part is drawn in its own
render pass that **loads** the colour attachment but **clears** the depth
buffer on the first part, so the handles always sit over the geometry — the
WebGPU equivalent of clearing only the depth buffer before drawing the gizmo.

### Objects render in one pass via a storage buffer

Every object is one instance in a storage buffer (model / normal matrix /
colour / pick colour / selected flag) indexed by `instance_index`, so all
objects draw in a single pass — the same pattern as the `WebGPUMultiGeo` demo.

## Possible extensions

- Marquee (rubber-band) selection.
- Fold the gizmo parts into the instanced pipeline to avoid the per-part passes.

## References

- [opengl-tutorial — Picking with an OpenGL hack](http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/picking-with-an-opengl-hack/) — the colour-ID picking idea, here rendered into an offscreen WebGPU target.
- [WebGPU Fundamentals](https://webgpufundamentals.org/) — render targets, buffer mapping and readback in WebGPU.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) — reference implementation of Maya-style transform gizmos.
- See [`WebGPUComputePicking`](../WebGPUComputePicking) for the compute-reduction variant that avoids whole-image readback.
