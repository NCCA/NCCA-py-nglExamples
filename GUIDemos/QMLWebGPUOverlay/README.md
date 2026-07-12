## QML WebGPU Overlay

Floating, draggable QML control panels (`ncca.ngl.qml`) layered over a live 3D
teapot viewport — the **WebGPU** version of `GUIDemos/QMLOverlayApp`, and the
working resolution of the limitation documented in `GUIDemos/QMLFloatingWidgets`.

![QMLWebGPUOverlay](QMLWebGPUOverlay.png)

### What this demo shows

A `QMainWindow` stacks two widgets:

- **The 3D viewport** (`PyNGLScene`) — a `ncca.ngl.webgpu.WebGPUWidget` that
  renders a diffuse-lit teapot **offscreen into a numpy buffer** and blits it
  via `QPainter`. It exposes `set_model_matrix(Mat4)`, `set_view_matrix(Mat4)`
  and `set_colour(r, g, b)` slots.
- **A transparent overlay** (`OverlayQuickWidget`, a `QQuickWidget`) carrying
  three floating panels built from `ncca.ngl.qml`'s `TransformWidget`,
  `RGBColourWidget` and `LookAtWidget`, plus a theme picker. Each panel drives
  the teapot directly through the slots above: model matrix, colour, and the
  camera's view matrix.

Because the overlay is transparent, clicks that miss every panel are
hit-tested (`PanelRegistry`) and forwarded down to the WebGPU scene. Panel
positions and the selected theme persist between runs via QML's `Settings`.

### Why WebGPU instead of OpenGL

`QMLFloatingWidgets` tried to render the teapot with a `QQuickFramebufferObject`
inside a pure Qt Quick scene graph. Under Qt 6's RHI/Metal backend that class
never obtains a valid OpenGL context (its renderer is never invoked, and forcing
the OpenGL backend segfaults on `QOpenGLFramebufferObject` construction).

`QMLOverlayApp` worked around this by rendering with a `QOpenGLWidget`, but that
forces the whole top-level surface to composite via OpenGL, so the overlay
`QQuickWidget` then *also* has to be forced onto the OpenGL scene graph backend
(`QQuickWindow.setGraphicsApi(OpenGL)` + a `QSurfaceFormat`) or it renders
nothing under Metal.

This demo removes that constraint entirely. `WebGPUWidget` is a **plain
`QWidget`** — it renders with `wgpu` offscreen and never creates a Qt OpenGL
surface. With no OpenGL top-level surface in play, the overlay `QQuickWidget`
uses Qt's default RHI/Metal backend directly: no `setGraphicsApi`, no
`QSurfaceFormat`, no backend forcing at all.

### Files

- `main.py` — builds the window, stacks the WebGPU scene and the transparent
  QML overlay, wires up mouse-forwarding hit testing
- `PyNGLScene.py` — the `WebGPUWidget` teapot scene and its matrix/colour slots
- `TeapotPipeline.py` — the diffuse render pipeline and uniform buffers
- `DiffuseShader.wgsl` — a minimal single-light diffuse shader (the WebGPU
  equivalent of PyNGL's `DefaultShader.DIFFUSE`)
- `panel_registry.py` — screen-rect registry used for click-through hit testing
- `main.qml` — the panels, themes, and layout persistence
- `DraggablePanel.qml` — drag-by-handler chrome with theme-able body

### Running

```bash
uv run GUIDemos/QMLWebGPUOverlay/main.py
```

### See also

- `GUIDemos/QMLOverlayApp` — the OpenGL version of this same overlay app
- `GUIDemos/QMLFloatingWidgets` — the pure-QML attempt kept as a documented
  reference for the RHI limitation this demo avoids

## References

- [QQuickWidget (Qt for Python)](https://doc.qt.io/qtforpython-6/PySide6/QtQuickWidgets/QQuickWidget.html) — the transparent QML layer over the viewport.
- [Qt QML Applications](https://doc.qt.io/qt-6/qmlapplications.html) — QML language and application structure.
- [wgpu-py documentation](https://wgpu-py.readthedocs.io/) — offscreen rendering with the Python WebGPU binding.
