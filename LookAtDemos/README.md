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
