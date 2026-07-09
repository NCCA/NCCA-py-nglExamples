## QML Floating Widgets

**Status: known platform limitation — this demo does not currently render.**
It's kept in the repo as a documented reference rather than deleted, since
the code and the failure mode are instructive.

### What this demo attempts

A pure Qt Quick / QML application: the entire window, including the 3D
viewport, is meant to be one Qt Quick scene graph.

- `TeapotView`, a custom `QQuickFramebufferObject`, is meant to draw a
  teapot with PyNGL's `ShaderLib`/`Primitives` into an offscreen FBO each
  frame.
- Three floating, draggable panels — built from `ncca.ngl.qml`'s
  `TransformWidget`, `RGBColourWidget` and `LookAtWidget` — sit on top of the
  viewport and drive it directly: `TransformWidget` sets the teapot's model
  matrix, `RGBColourWidget` sets its colour, `LookAtWidget` sets the
  camera's view matrix.
- Because the viewport and the panels would share one scene graph, panel
  dragging and z-order would be native QML behaviour (contrast with
  `GUIDemos/QMLOverlayApp`, which layers a separate `QOpenGLWidget` and
  `QQuickWidget` and has to do this by hand).

### Why it doesn't work

`QQuickFramebufferObject.Renderer` never obtains a valid, current OpenGL
context on this stack (PySide6 6.10.1, Qt 6, macOS 15.7.7):

- Under Qt 6's default RHI/Metal scene graph backend, `createRenderer()` /
  `synchronize()` / `render()` are silently never called at all.
- Forcing `QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)`
  before the application is constructed does make the renderer get invoked
  — but `QOpenGLFramebufferObject` construction then segfaults natively.
  This was confirmed with a minimal, PyNGL-free reproduction (a bare FBO,
  no format, no GL calls) that crashes at the exact same line, ruling out
  any bug in this demo's own code or in PyNGL.

This matches a known class of Qt 6 issue: `QQuickFramebufferObject` is a
Qt5-era API that doesn't reliably get an OpenGL context under RHI on all
platforms, which is why Qt 6.7+ introduced `QQuickRhiItem` as its
RHI-native replacement. A real fix would mean rewriting the renderer
against `QQuickRhiItem`'s RHI command-list model, which PyNGL's
`OpenGL.GL`-based `ShaderLib`/`Primitives` do not support — that would be
new library-level work in PyNGL itself, not a demo-level fix.

### Files

- `main.py` - registers `ncca.ngl.qml` and this demo's `teapot_view`
  module, forces the OpenGL scene graph backend (needed to get the
  renderer invoked at all, per above), loads `main.qml`
- `teapot_view.py` - `TeapotView` (`QQuickFramebufferObject`) + its
  renderer
- `main.qml` - the application window, viewport, and the three panels
- `DraggablePanel.qml` - drag-by-title-bar + raise-to-front chrome

### See instead

`GUIDemos/QMLOverlayApp` demonstrates the same floating-panel idea using a
`QOpenGLWidget` for the 3D rendering (not `QQuickFramebufferObject`), which
is unaffected by this issue and actually runs.
