## QML Overlay App

A classic `QWidget`-based PyNGL app (matching the style of
`GUIDemos/PySideGUIOpenGL`), with a transparent `QQuickWidget` layered on top
of a `QOpenGLWidget` to host floating, draggable panels — the QWidget/QML
interop counterpart to `GUIDemos/QMLFloatingWidgets`.

- `PyNGLScene` (a `QOpenGLWidget`) renders the teapot, driven by pre-composed
  matrices (`set_model_matrix`, `set_view_matrix`) and a colour
  (`set_colour`), rather than raw position/rotation/scale floats.
- Three floating panels — `TransformWidget`, `RGBColourWidget`,
  `LookAtWidget` from `ncca.ngl.qml` — sit in a transparent `QQuickWidget`
  stacked on top of the GL widget, and push matrices/colour into
  `PyNGLScene` via context-property calls in `main.qml`.
- Because the GL widget and the Qt Quick surface are two separate widgets
  (not one scene graph), a `PanelRegistry` tracks each panel's current
  screen rect; `OverlayQuickWidget` hit-tests every mouse event against it
  and either lets Qt Quick handle it (inside a panel) or forwards it through
  to `PyNGLScene` beneath (empty overlay space) — contrast with
  `GUIDemos/QMLFloatingWidgets`, where this is automatic because everything
  lives in one Qt Quick scene graph.
- There is no mouse-drag camera; `LookAtWidget` is the only camera control.

### Files

- `main.py` - `QMainWindow`, `OverlayQuickWidget` (click-through logic)
- `PyNGLScene.py` - the `QOpenGLWidget` teapot scene
- `panel_registry.py` - `PanelRegistry`, tracks panel rects for hit testing
- `main.qml` / `DraggablePanel.qml` - the three floating panels

### Controls

- Drag a panel's title bar to move it; click it to bring it to front.
- Transform panel: teapot position/rotation/scale.
- Colour panel: teapot colour.
- Camera panel: camera eye/look/world-up (drives the view matrix directly).
