## QML Floating Widgets

A pure Qt Quick / QML application: the entire window, including the 3D
viewport, is one Qt Quick scene graph.

- The teapot is rendered by `TeapotView`, a custom `QQuickFramebufferObject`
  that draws with PyNGL's `ShaderLib`/`Primitives` into an offscreen FBO
  each frame.
- Three floating, draggable panels — built from `ncca.ngl.qml`'s
  `TransformWidget`, `RGBColourWidget` and `LookAtWidget` — sit on top of the
  viewport and drive it directly: `TransformWidget` sets the teapot's model
  matrix, `RGBColourWidget` sets its colour, and `LookAtWidget` sets the
  camera's view matrix. There is no mouse-drag camera; `LookAtWidget` is the
  only camera control.
- Because the viewport and the panels share one scene graph, panel dragging,
  z-order ("always on top of whatever's behind"), and click ownership are
  all native QML behaviour — no manual hit-testing is needed (contrast with
  `GUIDemos/QMLOverlayApp`, which layers a separate `QOpenGLWidget` and
  `QQuickWidget` and has to do this by hand).

### Files

- `main.py` - registers `ncca.ngl.qml` and this demo's `teapot_view` module,
  loads `main.qml`
- `teapot_view.py` - `TeapotView` (`QQuickFramebufferObject`) + its renderer
- `main.qml` - the application window, viewport, and the three panels
- `DraggablePanel.qml` - drag-by-title-bar + raise-to-front chrome

### Controls

- Drag a panel's title bar to move it; click it to bring it to front.
- Transform panel: teapot position/rotation/scale.
- Colour panel: teapot colour.
- Camera panel: camera eye/look/world-up (drives the view matrix directly).
