# LookAtDemos

Combines NGL9Demos' SimpleLookAt and MultipleViews demos. `Tab` switches
between a single interactive perspective camera (`ngl.look_at` +
`ngl.perspective`) and a 2x2 grid comparing that same perspective view
against three fixed orthographic reference views (top, front, side) of the
identical troll-and-grid scene, built with `ngl.ortho`.

## Controls
- `Tab` : toggle simple / multi-view mode
- Left-drag : orbit, Right-drag : pan, Wheel : zoom (perspective view only)
- `Space` : reset, `Esc` : quit

![LookAtDemos](LookAtDemos.png)

## WebGPU version

`main_webgpu.py` draws all four quadrants in a single render pass using
`render_pass.set_viewport()` / `set_scissor_rect()` per pane (the same
technique `BVHViewer`'s four-view mode uses). It omits the reference grid,
since WebGPU has no baked line-grid primitive data — the camera comparison
itself is unaffected.
