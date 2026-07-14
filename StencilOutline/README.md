# Stencil Outline

![](StencilOutline.png)

A Maya-style selection outline done the classic way: two passes and the stencil buffer, no post-process, no second render target. A teapot, two cubes and a sphere sit on a grid; `Tab` cycles which one is "selected" and gets the orange fringe.

There's no picking here — that's [RayPickingSelection](../RayPickingSelection) and [SelectionManipulator](../SelectionManipulator). This demo is only about the stencil state machine.

## Controls

| Key               | Action                                                                      |
| :---------------- | :-------------------------------------------------------------------------- |
| `Tab`             | cycle which object is selected                                              |
| `O`               | toggle the outline pass — watch it disappear along with the extra draw call |
| `V`               | visualise the stencil buffer — tints every pixel currently marked 1         |
| `+`/`-`           | grow / shrink the outline width (`outlineScale`)                            |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits                            |

## The recipe

Every frame:

1. Clear colour, depth **and** stencil (`glClear(... | GL_STENCIL_BUFFER_BIT)`).
2. Draw every object normally. While drawing the selected one, also stamp 1 into the stencil buffer wherever it passes the depth test.
3. Redraw the selected object again, fattened along its normals, with depth testing off and the stencil test set to reject anywhere already stamped in step 2. What's left is only the fringe beyond the original silhouette.

## Stencil state machine

| Pass                          | `glStencilFunc`        | `glStencilOp` (fail, zfail, zpass) | `glStencilMask`    | Depth test | Draws                                                                 |
| :---------------------------- | :--------------------- | :--------------------------------- | :----------------- | :--------- | :-------------------------------------------------------------------- |
| Non-selected objects          | `GL_ALWAYS, 1, 0xFF`   | `KEEP, KEEP, KEEP`                 | `0x00` (no write)  | on         | normal shading                                                        |
| Selected object, normal pass  | `GL_ALWAYS, 1, 0xFF`   | `KEEP, KEEP, REPLACE`              | `0xFF`             | on         | normal shading, stamps stencil = 1                                    |
| Selected object, outline pass | `GL_NOTEQUAL, 1, 0xFF` | `KEEP, KEEP, KEEP`                 | `0x00` (read-only) | **off**    | fattened copy, flat orange, rejected wherever stencil already reads 1 |
| Stencil-visualise (`V`)       | `GL_EQUAL, 1, 0xFF`    | `KEEP, KEEP, KEEP`                 | `0x00` (read-only) | off        | full-screen tint triangle, alpha-blended                              |
| End of frame (restore)        | `GL_ALWAYS, 1, 0xFF`   | `KEEP, KEEP, KEEP`                 | `0xFF`             | on         | — resting state for the next `glClear`                                |

## Things to know

1. **`format.setStencilBufferSize(8)` is not optional.** It's one line in the `QSurfaceFormat` block in `main.py`, easy to forget, and if you do the whole demo still runs — no error, no crash — it just never shows an outline, because there is no stencil buffer to write into. If you copy this skeleton for your own stencil work, that line is the entire "setup".
2. **The write mask has to go back to `0xFF` before the next frame's `glClear`.** `glStencilMask` gates what a clear is allowed to touch as well as what a draw call writes. Leave it at `0x00` from the outline pass's read-only state and the next frame's clear silently stops touching the stencil buffer — the old selection outline never disappears.
3. **Fattening along the normal only looks clean on smooth meshes.** On the teapot, where neighbouring triangles share close-to-continuous normals, the pushed-out silhouette is a clean, continuous fringe. On the cubes, each face has its own flat normal, so the fattened copy pulls the six faces apart at every edge and leaves a visible seam/gap in the outline at the corners. Both are shown in this demo on purpose — it's the standard limitation of "extrude along normal" outlines, not a bug. Fixing it properly means either duplicating vertices per-face with averaged corner normals, or switching to a screen-space (Sobel-on-depth/stencil) outline instead.

## References

- [LearnOpenGL — Stencil Testing](https://learnopengl.com/Advanced-OpenGL/Stencil-testing) — the object-outlining recipe this demo follows.
- [OpenGL Wiki — Stencil Test](https://www.khronos.org/opengl/wiki/Stencil_Test) — `glStencilFunc`/`glStencilOp`/`glStencilMask` reference.
