# QML floating-widget demos (ImGui-style panels over a PyNGL 3D viewport)

## Goal

Add two new sibling demos under `GUIDemos/` that show how to combine PyNGL OpenGL
rendering with `ncca.ngl.qml`'s Qt Quick widgets (`TransformWidget`,
`RGBColourWidget`, `LookAtWidget`, ...) as movable, floating panels layered over
a 3D viewport — similar in spirit to how Dear ImGui overlays debug panels on top
of a render. Two different integration strategies are demonstrated side by side:

- `GUIDemos/QMLFloatingWidgets` — a pure Qt Quick / QML application, with the
  OpenGL scene itself rendered as a QML item (`QQuickFramebufferObject`).
- `GUIDemos/QMLOverlayApp` — a classic `QWidget`-based app (matching the style
  of the existing `PySideGUIOpenGL` / `WebGPUGUI` demos), with a transparent
  `QQuickWidget` stacked on top of a `QOpenGLWidget` to host the floating panels.

Both demos control the *same* scene through the *same* three existing
`ncca.ngl.qml` widgets, so the two folders are directly comparable — the only
difference between them is the integration mechanism, which is the point of
the demo.

## Scene & controls (shared by both demos)

- Geometry: a single teapot, drawn with `DefaultShader.DIFFUSE` (same shader
  path as `GUIDemos/PySideGUIOpenGL`).
- Three floating panels, each wrapping an existing `ncca.ngl.qml` widget:
  - `TransformWidget` → `TransformModel.matrix` is used directly as the model
    matrix `M` for the teapot.
  - `RGBColourWidget` → `RGBColourModel.r/g/b` is used as the `Colour` shader
    uniform.
  - `LookAtWidget` → `LookAtModel.matrix` is used directly as the view matrix.
- There is **no mouse-drag arcball camera** in either demo. The camera is
  entirely panel-driven via `LookAtWidget`. This is a deliberate scope
  reduction: it removes any need to reconcile "drag a panel" vs. "drag the
  viewport to rotate the camera" mouse-ownership conflicts, keeping both demos
  focused on the QML/OpenGL integration technique rather than camera controls
  (already covered by other demos in this repo).
- Per frame: `MVP = project @ view(LookAtModel) @ M(TransformModel)`, plus the
  standard normal-matrix computation from `PyNGLScene.load_matrices_to_shader`.
- `project` is a fixed `perspective(45.0, aspect, 0.1, 100.0)` recomputed on
  resize, as in the existing demos.

## Draggable panel chrome

Each demo has its own copy of a small `DraggablePanel.qml` (NOT added to the
shared `ncca.ngl.qml` package — it's demo-only chrome, duplicated between the
two folders to keep each demo folder self-contained, per this repo's
"standalone demo" convention):

- A `Frame` with a title `ToolButton` acting as a drag handle
  (`MouseArea { drag.target: root }` on the title bar only, so the widget's
  own interactive controls — sliders, spin boxes — aren't hijacked by dragging).
- Raise-to-front on press: the containing `main.qml` keeps a
  `property int nextZ: 1` counter; each panel's title-bar `MouseArea.onPressed`
  does `root.z = mainWindow.nextZ++`.
- Semi-transparent panel background so the viewport is visible underneath,
  reinforcing the "overlay" look.

## Demo 1: `GUIDemos/QMLFloatingWidgets` (pure QML)

Files:
- `main.py` — `QGuiApplication` + `QQmlApplicationEngine`. Imports
  `ncca.ngl.qml` (registers the shared widgets) and this demo's own
  `teapot_view` module (registers `TeapotView`), adds this demo's directory to
  the QML import path, loads `main.qml`. Modeled directly on
  `ncca/ngl/qml/__main__.py`.
- `teapot_view.py` — `TeapotView(QQuickFramebufferObject)`, decorated
  `@QmlElement` (`QML_IMPORT_NAME = "qmlfloatingwidgets"`, version 1.0).
  QML-facing properties: `transformMatrix: Mat4`, `viewMatrix: Mat4`,
  `colour: Vec3` (bound in QML straight from the panels' models: e.g.
  `transformMatrix: transformWidget.model.matrix`). An inner
  `Renderer(QQuickFramebufferObject.Renderer)` owns the actual GL calls:
  `initializeGL`-equivalent one-time setup (`ShaderLib.use`,
  `Primitives.load_default_primitives()`), and per-`render()` uses the item's
  current property values to compute `MVP`/`normalMatrix`/`Colour` and issues
  `Primitives.draw("teapot")`. `createFramebufferObject`/`synchronize` follow
  the standard PySide6 `QQuickFramebufferObject` pattern (synchronize copies
  the QML-thread property values into renderer-owned fields under the
  renderer's `synchronize` call, matching Qt's threaded-render-loop contract).
- `main.qml` — `ApplicationWindow` with `TeapotView { anchors.fill: parent }`
  as the base layer, then three `DraggablePanel` instances on top at staggered
  initial positions, each with its own `TransformModel`/`RGBColourModel`/
  `LookAtModel` instance created by the respective widget.
- `DraggablePanel.qml` — as described above.

Because the viewport and the panels live in the same Qt Quick scene graph,
z-ordering and mouse-event ownership (a panel on top "steals" the click,
empty space passes through to whatever's beneath — here, nothing, since the
viewport doesn't need mouse input) are both native QML behaviour. No manual
hit-testing or event forwarding is needed in this demo.

## Demo 2: `GUIDemos/QMLOverlayApp` (QWidget + QQuickWidget overlay)

Files:
- `PyNGLScene.py` — a `QOpenGLWidget`, closely modeled on
  `GUIDemos/PySideGUIOpenGL/PyNGLScene.py`, but takes pre-composed matrices
  instead of raw position/rotation/scale floats:
  - `set_model_matrix(Mat4)`, `set_view_matrix(Mat4)`, `set_colour(r, g, b)`
    slots, called whenever the corresponding QML model changes.
  - `initializeGL`/`paintGL`/`resizeGL` otherwise match the existing demo.
- `panel_registry.py` — `PanelRegistry(QObject)`, `@QmlElement`, registered as
  a QML singleton. Holds `dict[str, QRect]` of currently-known panel screen
  rects. `Slot(str, float, float, float, float)` `update_rect(panel_id, x, y,
  w, h)` called from each `DraggablePanel.qml` on geometry change
  (`onXChanged`/`onYChanged`/`onWidthChanged`/`onHeightChanged`). Plain method
  `hit_test(pos: QPoint) -> bool` = `any(rect.contains(pos) for rect in
  self._rects.values())` — pure Python, no Qt event-loop dependency, so it's
  unit-testable headlessly.
- `main.py` — `QMainWindow`; `PyNGLScene` fills the central widget/window.
  A `QQuickWidget` subclass (`OverlayQuickWidget`) is created as a sibling,
  resized to match and `raise()`-d above it on every resize, with
  `setClearColor(Qt.transparent)`, `Qt.WA_TranslucentBackground`, and
  `Qt.WA_AlwaysStackOnTop` set so the GL widget remains visible through the
  gaps between panels.
  `OverlayQuickWidget` overrides `mousePressEvent` / `mouseMoveEvent` /
  `mouseReleaseEvent`: if `PanelRegistry.hit_test(event.position())` is
  `True`, defer to `super()` (normal QtQuick panel drag/click handling);
  otherwise call `event.ignore()` and `QApplication.sendEvent(self.scene,
  translated_event)` to forward the click through to `PyNGLScene` beneath.
  There is no camera-drag behaviour hooked up on `PyNGLScene` yet (no mouse
  camera per the scope decision above), so today this forwarding is inert —
  it exists so the "click-through when not over a panel" behaviour is
  demonstrably correct and extendable, which is the actual teaching point of
  this demo.
- `main.qml` — `ApplicationWindow`-free root `Item` (loaded via
  `QQuickWidget.setSource`, not a full `ApplicationWindow`, since the window
  itself is the `QMainWindow`) containing the three `DraggablePanel`
  instances only (no viewport item — that's the separate `PyNGLScene` widget
  beneath).
- `DraggablePanel.qml` — same chrome as demo 1's copy, plus the
  `PanelRegistry.update_rect(...)` calls on geometry change.

## Testing

- `GUIDemos/QMLOverlayApp/tests/test_panel_registry.py` — headless unit test
  for `PanelRegistry.hit_test`/`update_rect` (pure Python/`QRect` logic, no GL
  or QML runtime needed), following this repo's existing pattern of testing
  only the pure-Python logic pieces of GUI demos.
- No test for `QMLFloatingWidgets` (no pure-Python logic to isolate — the
  `TeapotView` renderer is inherently GL/QML-runtime-coupled). Manual
  smoke-test only, as with the repo's other rendering demos.

## Documentation

- Each demo gets a `README.md` (what it demonstrates, the two integration
  approaches contrasted, controls) and a `.png` screenshot, per repo
  convention.
- Root `README.md` gets a link to both new demo folders alongside the
  existing `GUIDemos` entries.

## Out of scope

- Mouse-drag arcball camera (explicitly deferred; `LookAtWidget` is the only
  camera control).
- Any change to the shared `ncca.ngl.qml` package in the separate PyNGL repo —
  both demos consume its existing widgets/models unmodified.
- Object/primitive selection, wireframe toggle, or other controls present in
  `PySideGUIOpenGL` but not in the `ncca.ngl.qml` widget set.
