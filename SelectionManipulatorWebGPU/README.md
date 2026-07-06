# Selection and Manipulation (WebGPU)

A WebGPU port of the [`SelectionManipulator`](../SelectionManipulator) OpenGL
demo: DCC-style object selection and manipulation in the spirit of Maya or
Houdini. A small scene of objects can be picked with the mouse and transformed
with visual Translate / Rotate / Scale gizmos.

```bash
uv run main.py          # or ./main.py
```

![](SelectionManipulatorWebGPU.png)

## Controls (Maya-style)

| Input | Action |
|---|---|
| `Q` | Select mode (gizmo hidden) |
| `W` | Translate mode (arrows) |
| `E` | Rotate mode (rings) |
| `R` | Scale mode (boxes) |
| Left click | Select the object under the cursor (replaces selection) |
| `Ctrl` + click | Toggle an object in / out of the selection (multi-select) |
| Drag a handle | Transform **all** selected objects along that axis |
| `Alt` + LMB drag | Tumble the camera |
| `Alt` + RMB drag | Pan the camera |
| Mouse wheel | Dolly in / out |
| `Space` | Reset the camera |
| `Escape` | Quit |

## How it uses the WebGPU stack

The demo leans on the stock `ncca.ngl.webgpu` factory pipelines wherever they
fit, and adds one small custom pipeline only where the factory can't help:

| Part of the scene | Pipeline |
|---|---|
| Ground grid | factory `SINGLE_COLOUR_LINES` |
| Gizmo handles (display) | factory `SINGLE_COLOUR_TRIANGLES`, one pass per part |
| Picking – gizmo handles | factory `SINGLE_COLOUR_TRIANGLES` (reserved ID colours) |
| Picking – objects | the object pipeline in flat "pick" mode |
| Objects (diffuse + wireframe) | custom `ObjectPipeline` (`ObjectShader.wgsl`) |

## What changed from the OpenGL version

The scene, the `SelectionObject` hierarchy, the Maya controls, and all of the
manipulator drag maths are unchanged. Three things that OpenGL does with fixed
-function state have no direct WebGPU equivalent and are done differently:

### Wireframe overdraw → single-pass barycentric

OpenGL redraws selected objects with `glPolygonMode(GL_LINE)` and a negative
polygon offset. WebGPU has no polygon-line mode, so `ObjectShader.wgsl` draws
the fill and the wireframe in **one pass**: the geometry is a non-indexed
triangle soup, so `vertex_index % 3` identifies each triangle corner and emits
a barycentric coordinate `(1,0,0)/(0,1,0)/(0,0,1)`. In the fragment shader the
smallest barycentric component is the distance to the nearest edge; `fwidth`
keeps the line ~1px wide and anti-aliased. No extra vertex buffer is needed.

### `glReadPixels` picking → offscreen render + readback

On click the scene is rendered flat (each object in its unique colour ID, the
gizmo handles in reserved ID colours on top) into an **offscreen** MSAA texture
that resolves to a copyable target. That target is copied to a mapped buffer
with `copy_texture_to_buffer`, and a 9×9 pixel block under the cursor is read
back on the CPU. Handle colours are matched first (so handles stay grabbable in
front of geometry), then object IDs. The object pass reuses the object pipeline
with a `render_mode` uniform that switches the fragment shader to flat pick
colour — no lighting, no wireframe.

### `glClear(GL_DEPTH_BUFFER_BIT)` gizmo overdraw → a second pass

To keep the gizmo on top of the scene, each handle part is drawn in its own
render pass that **loads** the colour attachment but **clears** the depth
buffer on the first part, so the handles always sit over the geometry — the
WebGPU equivalent of clearing only the depth buffer before drawing the gizmo.

### Objects render in one pass via a storage buffer

Every object is one instance in a storage buffer (model / normal matrix /
colour / pick colour / selected flag) indexed by `instance_index`, so all
objects draw in a single pass — the same pattern as the `WebGPUMultiGeo` demo.

## Possible extensions

* A centre handle on the scale gizmo for uniform scaling.
* Screen-aligned free-move handle for translate.
* Marquee (rubber-band) selection.
* Fold the gizmo parts into the instanced pipeline to avoid the per-part passes.
