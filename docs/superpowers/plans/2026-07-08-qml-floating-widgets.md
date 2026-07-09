# QML Floating Widgets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status update (during execution):** Task 3 uncovered a platform-level
> blocker: `QQuickFramebufferObject` (Tasks 1-4's `QMLFloatingWidgets` demo)
> cannot obtain a valid, current OpenGL context under Qt 6's RHI scene graph
> backend on this machine (PySide6 6.10.1, macOS 15.7.7) — confirmed with a
> minimal, PyNGL-free reproduction that segfaults at bare
> `QOpenGLFramebufferObject` construction. Fixing it properly would require
> rewriting the renderer against `QQuickRhiItem` (Qt 6.7+'s RHI-native
> replacement) plus new library-level support in PyNGL, since `ShaderLib`/
> `Primitives` are hardcoded to raw `OpenGL.GL` calls. Per user decision,
> Tasks 1-4 are considered done-as-documented-limitation (see
> `GUIDemos/QMLFloatingWidgets/README.md`), not pursued further. **Tasks
> 5-10 (`QMLOverlayApp`) proceed as originally planned** — that demo uses a
> `QOpenGLWidget`, not `QQuickFramebufferObject`, and is unaffected.

**Goal:** Add two new sibling demos under `GUIDemos/` — `QMLFloatingWidgets` (pure QML, OpenGL-in-`QQuickFramebufferObject`) and `QMLOverlayApp` (`QOpenGLWidget` + transparent `QQuickWidget` overlay with click-through) — both showing a teapot controlled entirely by draggable, ImGui-style floating panels built from the existing `ncca.ngl.qml` widgets (`TransformWidget`, `RGBColourWidget`, `LookAtWidget`).

**Architecture:** Both demos share one control model: `TransformModel.get_matrix()` drives the teapot's `M`, `RGBColourModel.get_value()` drives the `Colour` uniform, `LookAtModel.get_matrix()` drives the view matrix. Demo 1 renders inside the Qt Quick scene graph itself (one scene graph → native z-order/mouse-ownership, no pass-through code needed). Demo 2 keeps the classic `QOpenGLWidget` approach and layers a transparent `QQuickWidget` on top, using a `PanelRegistry` singleton (populated by each panel's QML geometry) to decide whether a click should be handled by Qt Quick or forwarded through to the GL widget.

**Tech Stack:** Python 3.13, PySide6 (`QtQuick`, `QtQml`, `QtOpenGL`, `QtQuickWidgets`, `QtOpenGLWidgets`), PyOpenGL, `ncca.ngl` (`Mat4`, `Vec3`, `Transform`, `perspective`, `ShaderLib`, `Primitives`, `DefaultShader`), `uv`, `ruff`, `pytest`.

## Global Constraints

- Use **uv** exclusively: `uv run <script.py>`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`.
- Each demo script is directly executable: shebang `#!/usr/bin/env -S uv run --script` as the first line (see `GUIDemos/WebGPUGUI/main.py:1`), `chmod +x`.
- Follow existing GUIDemos conventions: `README.md` + a `.png` screenshot per demo folder, and a new row added to the root `README.md` demo table (same table that already lists `GUIDemos/PySideGUIOpenGL`, `GUIDemos/NGLWidgetsOpenGL`, `GUIDemos/WebGPUGUI` at `README.md:149-151`).
- No changes to the separate PyNGL repo (`/Volumes/teaching/Code/PyNGL`) — only consume its existing public API (`ncca.ngl`, `ncca.ngl.qml`, `ncca.ngl.opengl`). **Amended during Task 3:** a genuine library-level packaging gap (missing `qmldir`, see Task 3's note) required a one-line additive `qmldir` file in PyNGL, added with explicit user approval and merged to PyNGL's `Version1.0`. This is the only sanctioned exception; it does not reopen this constraint for anything else.
- `ruff check --select I --fix` and `ruff format` must pass on every new `.py` file before it's committed.
- Per this repo's global git workflow: work happens in a worktree/branch (`agent/qml-floating-widgets`), never commit directly to `main`/`master`/`Version1.0`, run tests before each commit.

---

## File Structure

```
GUIDemos/
  QMLFloatingWidgets/
    main.py                 # QGuiApplication + QQmlApplicationEngine entry point
    teapot_view.py           # TeapotView(QQuickFramebufferObject), @QmlElement
    main.qml                 # ApplicationWindow: TeapotView + 3 DraggablePanels
    DraggablePanel.qml       # title-bar-drag + raise-to-front chrome
    README.md
    QMLFloatingWidgets.png   # screenshot (added last, manual step)

  QMLOverlayApp/
    main.py                  # QMainWindow: PyNGLScene + OverlayQuickWidget
    PyNGLScene.py             # QOpenGLWidget, set_model_matrix/set_view_matrix/set_colour slots
    panel_registry.py         # PanelRegistry(QObject), @QmlElement singleton, hit_test()
    main.qml                  # root Item: 3 DraggablePanels only (no viewport item)
    DraggablePanel.qml        # same chrome as demo 1, plus PanelRegistry.update_rect() calls
    tests/
      test_panel_registry.py  # headless unit test for PanelRegistry.hit_test/update_rect
    README.md
    QMLOverlayApp.png         # screenshot (added last, manual step)
```

**Interfaces at a glance** (exact names used across tasks):

- `ncca.ngl.qml.TransformModel` (existing): `get_matrix() -> Mat4`, signal `valueChanged`, QML property `model` on `TransformWidget`.
- `ncca.ngl.qml.LookAtModel` (existing): `get_matrix() -> Mat4`, signal `valueChanged`, QML property `model` on `LookAtWidget`.
- `ncca.ngl.qml.RGBColourModel` (existing): `get_value() -> Vec3`, signal `colourChanged`, QML property `model` on `RGBColourWidget`.
- `TeapotView` (new, Demo 1): QML properties `transformModel: var`, `lookAtModel: var`, `colourModel: var` (each assigned a model instance, not a raw matrix).
- `PanelRegistry` (new, Demo 2): `Slot(str, float, float, float, float) update_rect(panel_id, x, y, w, h)`, plain method `hit_test(pos: QPointF) -> bool`.
- `PyNGLScene` (new, Demo 2): `Slot(Mat4) set_model_matrix(m)`, `Slot(Mat4) set_view_matrix(m)`, `Slot(float, float, float) set_colour(r, g, b)`.

---

## Task 1: `DraggablePanel.qml` chrome, written once and verified in isolation

**Files:**
- Create: `GUIDemos/QMLFloatingWidgets/DraggablePanel.qml`
- Create: `GUIDemos/QMLFloatingWidgets/main.qml` (minimal smoke-test version, replaced/extended in Task 3)
- Create: `GUIDemos/QMLFloatingWidgets/main.py` (minimal pure-QML launcher, no `TeapotView` yet)

**Interfaces:**
- Produces: `DraggablePanel { id: panel; title: "..."; default property contentItem }` — a `Frame`-based wrapper any child widget can be placed inside via QML default-property children, draggable by its title bar, raises its own `z` above siblings on press via `ApplicationWindow.nextZ`.

- [ ] **Step 1: Write `DraggablePanel.qml`**

```qml
// GUIDemos/QMLFloatingWidgets/DraggablePanel.qml
import QtQuick
import QtQuick.Controls

Frame {
    id: root

    property string title: ""
    default property alias content: contentArea.children

    width: contentArea.implicitWidth + 16
    height: contentArea.implicitHeight + titleBar.height + 24
    opacity: 0.92

    background: Rectangle {
        color: "#2b2b2b"
        border.color: "#555555"
        radius: 6
    }

    function raiseToFront() {
        var win = root.Window.window
        if (win && win.nextZ !== undefined) {
            win.nextZ += 1
            root.z = win.nextZ
        }
    }

    Column {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 6

        Rectangle {
            id: titleBar
            width: parent.width
            height: 24
            color: "#3c3c3c"
            radius: 4

            Text {
                anchors.centerIn: parent
                text: root.title
                color: "white"
                font.bold: true
            }

            MouseArea {
                anchors.fill: parent
                drag.target: root
                onPressed: root.raiseToFront()
            }
        }

        Item {
            id: contentArea
            width: parent.width
            height: childrenRect.height
        }
    }
}
```

- [ ] **Step 2: Write a minimal `main.qml` to smoke-test the panel**

```qml
// GUIDemos/QMLFloatingWidgets/main.qml
import QtQuick
import QtQuick.Controls
import QtQuick.Window

ApplicationWindow {
    id: window
    property int nextZ: 1

    title: "QML Floating Widgets"
    visible: true
    width: 1024
    height: 720
    color: "#606060"

    DraggablePanel {
        title: "Test Panel"
        x: 40
        y: 40
        content: [
            Text { text: "Drag me by the title bar"; color: "white" }
        ]
    }
}
```

- [ ] **Step 3: Write the launcher**

```python
#!/usr/bin/env -S uv run --script
# GUIDemos/QMLFloatingWidgets/main.py
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


def main() -> int:
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(Path(__file__).parent / "main.qml")))
    if not engine.rootObjects():
        return -1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run it and verify manually**

Run: `chmod +x GUIDemos/QMLFloatingWidgets/main.py && uv run GUIDemos/QMLFloatingWidgets/main.py`
Expected: a window with a grey background and one dark floating panel titled "Test Panel" containing white text "Drag me by the title bar". Dragging the title bar moves the panel; clicking it brings it in front of anything else (only one panel exists yet, so this is just a smoke test that `raiseToFront()` doesn't error).

- [ ] **Step 5: Commit**

```bash
git add GUIDemos/QMLFloatingWidgets/DraggablePanel.qml GUIDemos/QMLFloatingWidgets/main.qml GUIDemos/QMLFloatingWidgets/main.py
git commit -m "feat: add draggable QML panel chrome for QMLFloatingWidgets demo"
```

---

## Task 2: `TeapotView` — OpenGL teapot rendered inside a `QQuickFramebufferObject`

**Files:**
- Create: `GUIDemos/QMLFloatingWidgets/teapot_view.py`

**Interfaces:**
- Consumes: `ncca.ngl.Mat4`, `ncca.ngl.Vec3`, `ncca.ngl.Transform` (unused here; matrices come pre-composed from models), `ncca.ngl.perspective`, `ncca.ngl.opengl.ShaderLib`, `ncca.ngl.opengl.Primitives`, `ncca.ngl.opengl.DefaultShader`.
- Produces: `TeapotView(QQuickFramebufferObject)`, `@QmlElement`, `QML_IMPORT_NAME = "qmlfloatingwidgets"`, `QML_IMPORT_MAJOR_VERSION = 1`. QML-settable properties `transformModel`, `lookAtModel`, `colourModel` (each holds a model object — e.g. `TransformModel` — not a raw matrix). Internally connects each model's change signal to `self.update()` so edits repaint the view.

- [ ] **Step 1: Write `teapot_view.py`**

```python
"""QQuickFramebufferObject that renders a PyNGL teapot driven by ncca.ngl.qml models."""

from PySide6.QtCore import Property, QObject, QSize
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickFramebufferObject

from ncca.ngl import Mat3, Vec3, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib

QML_IMPORT_NAME = "qmlfloatingwidgets"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class TeapotView(QQuickFramebufferObject):
    """Renders a teapot using matrices/colour pulled from ncca.ngl.qml models."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._transform_model: QObject | None = None
        self._look_at_model: QObject | None = None
        self._colour_model: QObject | None = None

    def get_transform_model(self) -> QObject | None:
        return self._transform_model

    def set_transform_model(self, model: QObject) -> None:
        if self._transform_model is not None:
            self._transform_model.valueChanged.disconnect(self.update)
        self._transform_model = model
        model.valueChanged.connect(self.update)
        self.update()

    transformModel = Property(QObject, get_transform_model, set_transform_model)

    def get_look_at_model(self) -> QObject | None:
        return self._look_at_model

    def set_look_at_model(self, model: QObject) -> None:
        if self._look_at_model is not None:
            self._look_at_model.valueChanged.disconnect(self.update)
        self._look_at_model = model
        model.valueChanged.connect(self.update)
        self.update()

    lookAtModel = Property(QObject, get_look_at_model, set_look_at_model)

    def get_colour_model(self) -> QObject | None:
        return self._colour_model

    def set_colour_model(self, model: QObject) -> None:
        if self._colour_model is not None:
            self._colour_model.colourChanged.disconnect(self.update)
        self._colour_model = model
        model.colourChanged.connect(self.update)
        self.update()

    colourModel = Property(QObject, get_colour_model, set_colour_model)

    def createRenderer(self) -> "TeapotRenderer":
        return TeapotRenderer()


class TeapotRenderer(QQuickFramebufferObject.Renderer):
    """Owns the GL state and issues the teapot draw call each frame."""

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._mvp = None
        self._mv = None
        self._normal_matrix = None
        self._colour = Vec3(1.0, 1.0, 0.0)
        self._aspect = 1.0

    def createFramebufferObject(self, size: QSize) -> QOpenGLFramebufferObject:
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObjectFormat.Attachment.CombinedDepthStencil)
        fmt.setSamples(4)
        self._aspect = size.width() / max(size.height(), 1)
        return QOpenGLFramebufferObject(size, fmt)

    def synchronize(self, item: TeapotView) -> None:
        if item.transformModel is None or item.lookAtModel is None or item.colourModel is None:
            return
        model_matrix = item.transformModel.get_matrix()
        view_matrix = item.lookAtModel.get_matrix()
        project = perspective(45.0, self._aspect, 0.1, 100.0)
        mv = view_matrix @ model_matrix
        self._mvp = project @ mv
        self._mv = mv
        normal_matrix = Mat3.from_mat4(mv)
        self._normal_matrix = normal_matrix.inverse().transposed()
        self._colour = item.colourModel.get_value()

    def render(self) -> None:
        import OpenGL.GL as gl

        if not self._initialized:
            ShaderLib.use(DefaultShader.DIFFUSE)
            ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
            Primitives.load_default_primitives()
            self._initialized = True

        if self._mvp is None:
            return

        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("MVP", self._mvp)
        ShaderLib.set_uniform("MV", self._mv)
        ShaderLib.set_uniform("normalMatrix", self._normal_matrix)
        ShaderLib.set_uniform(
            "Colour", self._colour.x, self._colour.y, self._colour.z, 1.0
        )
        Primitives.draw("teapot")
```

- [ ] **Step 2: No automated test** — `TeapotRenderer` is inherently GL/QML-runtime-coupled (per the design doc's testing section); it is verified visually in Task 3 once wired into `main.qml`. Skip to commit.

- [ ] **Step 3: Commit**

```bash
git add GUIDemos/QMLFloatingWidgets/teapot_view.py
git commit -m "feat: add TeapotView QQuickFramebufferObject renderer"
```

---

## Task 3: Wire `TeapotView` + the three floating widget panels into `main.qml`/`main.py`

**Files:**
- Modify: `GUIDemos/QMLFloatingWidgets/main.qml` (replace Task 1's smoke-test contents)
- Modify: `GUIDemos/QMLFloatingWidgets/main.py` (register `ncca.ngl.qml` and `teapot_view`)

**Interfaces:**
- Consumes: `TeapotView` (Task 2), `DraggablePanel` (Task 1), `ncca.ngl.qml.TransformWidget`/`RGBColourWidget`/`LookAtWidget` (existing library QML types, each exposing `property alias model`).

- [ ] **Step 1: Update `main.py` to register both QML modules**

```python
#!/usr/bin/env -S uv run --script
# GUIDemos/QMLFloatingWidgets/main.py
import sys
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml widget types)
import teapot_view  # noqa: F401  (import registers TeapotView)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


def main() -> int:
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    # ncca.ngl.qml's qmldir declares module "ncca.ngl.qml", so the import path
    # must be the directory that CONTAINS the ncca/ package root (four levels
    # up from ncca/ngl/qml/__init__.py), not the qml/ leaf directory itself.
    engine.addImportPath(str(Path(ncca.ngl.qml.__file__).parent.parent.parent.parent))
    engine.load(QUrl.fromLocalFile(str(Path(__file__).parent / "main.qml")))
    if not engine.rootObjects():
        return -1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

> **Note (discovered during implementation):** this requires a `qmldir` file at
> `ncca/ngl/qml/qmldir` in the separate PyNGL repo declaring the composite
> widget types (`TransformWidget`, `LookAtWidget`, `RGBColourWidget`, etc.) as
> module `ncca.ngl.qml` version 1.0. Without it, those types only resolve via
> Qt's implicit same-directory import, which works for PyNGL's own bundled
> demo (`main.qml` lives in that same directory) but not for a `main.qml` in a
> different directory, like this one. This was added to PyNGL directly (with
> explicit user approval, merged to `Version1.0` there) since it's a
> library-level packaging gap, not a demo-level issue.

- [ ] **Step 2: Replace `main.qml` with the full scene**

```qml
// GUIDemos/QMLFloatingWidgets/main.qml
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import ncca.ngl.qml 1.0
import qmlfloatingwidgets 1.0

ApplicationWindow {
    id: window
    property int nextZ: 100

    title: "QML Floating Widgets"
    visible: true
    width: 1200
    height: 800

    TeapotView {
        id: teapotView
        anchors.fill: parent
        transformModel: transformWidget.model
        lookAtModel: lookAtWidget.model
        colourModel: rgbWidget.model
    }

    TransformWidget {
        id: transformWidget
        name: "Transform"
    }
    DraggablePanel {
        title: "Transform"
        x: 30
        y: 30
        content: [transformWidget]
    }

    RGBColourWidget {
        id: rgbWidget
        name: "Colour"
    }
    DraggablePanel {
        title: "Colour"
        x: 30
        y: 260
        content: [rgbWidget]
    }

    LookAtWidget {
        id: lookAtWidget
        name: "Camera"
    }
    DraggablePanel {
        title: "Camera"
        x: 30
        y: 360
        content: [lookAtWidget]
    }
}
```

> Note: `transformWidget`/`rgbWidget`/`lookAtWidget` are declared as top-level children of `ApplicationWindow` (not inside `DraggablePanel`) purely so `TeapotView`'s property bindings (`transformModel: transformWidget.model`) can reference them by `id` before the panel reparents their visual representation via `content:`. Setting a `default property alias content: contentArea.children` on `DraggablePanel` reparents the *Item* into the panel's layout while the `id` reference remains valid for bindings elsewhere in the file — this is normal QML behavior (reparenting doesn't invalidate `id` lookups).

- [ ] **Step 3: Run it and verify manually**

Run: `uv run GUIDemos/QMLFloatingWidgets/main.py`
Expected: a window showing a yellow teapot rendered via the `TeapotView` background, with three draggable panels on top ("Transform", "Colour", "Camera"). Moving the Transform panel's position/rotation/scale spin boxes should move/rotate/scale the teapot live. Moving the Colour panel's sliders should recolour it. Moving the Camera (LookAtWidget) eye/look values should change the view. Dragging any panel's title bar should move it and bring it to front.

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLFloatingWidgets/main.qml GUIDemos/QMLFloatingWidgets/main.py
git commit -m "feat: wire TeapotView and floating control panels into QMLFloatingWidgets demo"
```

---

## Task 4: `QMLFloatingWidgets` README + screenshot + root README link

**Files:**
- Create: `GUIDemos/QMLFloatingWidgets/README.md`
- Create: `GUIDemos/QMLFloatingWidgets/QMLFloatingWidgets.png` (manual screenshot — see Step 2)
- Modify: `README.md` (repo root, demo table around `README.md:149-151`)

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Take a screenshot**

Run `uv run GUIDemos/QMLFloatingWidgets/main.py`, arrange the three panels so all are visible without excessive overlap, take a screenshot, save as `GUIDemos/QMLFloatingWidgets/QMLFloatingWidgets.png` (same convention as `GUIDemos/PySideGUIOpenGL/PySideGUI.png`).

- [ ] **Step 3: Add a row to the root README's demo table**

Find the row for `GUIDemos/WebGPUGUI` at `README.md:151` and add a new row immediately after it:

```markdown
| <a href="GUIDemos/QMLFloatingWidgets"><img src="GUIDemos/QMLFloatingWidgets/QMLFloatingWidgets.png" width="220"></a> | [GUIDemos/QMLFloatingWidgets](GUIDemos/QMLFloatingWidgets) | Pure QML app: OpenGL teapot in a QQuickFramebufferObject with floating ncca.ngl.qml control panels |
```

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLFloatingWidgets/README.md GUIDemos/QMLFloatingWidgets/QMLFloatingWidgets.png README.md
git commit -m "docs: add README, screenshot and root link for QMLFloatingWidgets demo"
```

---

## Task 5: `PanelRegistry` with TDD unit test (Demo 2, no GUI needed)

**Files:**
- Create: `GUIDemos/QMLOverlayApp/panel_registry.py`
- Create: `GUIDemos/QMLOverlayApp/tests/test_panel_registry.py`

**Interfaces:**
- Produces: `PanelRegistry(QObject)`, `@QmlElement`, `QML_IMPORT_NAME = "qmloverlayapp"`, `QML_IMPORT_MAJOR_VERSION = 1`. `Slot(str, float, float, float, float) update_rect(panel_id: str, x: float, y: float, w: float, h: float) -> None`. Plain method `hit_test(pos: QPointF) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# GUIDemos/QMLOverlayApp/tests/test_panel_registry.py
import sys
from pathlib import Path

from PySide6.QtCore import QPointF

sys.path.insert(0, str(Path(__file__).parent.parent))

from panel_registry import PanelRegistry  # noqa: E402


def test_hit_test_false_when_no_panels_registered():
    registry = PanelRegistry()
    assert registry.hit_test(QPointF(10, 10)) is False


def test_hit_test_true_inside_a_registered_panel():
    registry = PanelRegistry()
    registry.update_rect("transform", 100.0, 100.0, 200.0, 150.0)
    assert registry.hit_test(QPointF(150.0, 150.0)) is True


def test_hit_test_false_outside_all_registered_panels():
    registry = PanelRegistry()
    registry.update_rect("transform", 100.0, 100.0, 200.0, 150.0)
    assert registry.hit_test(QPointF(0.0, 0.0)) is False


def test_update_rect_replaces_previous_rect_for_same_id():
    registry = PanelRegistry()
    registry.update_rect("transform", 0.0, 0.0, 10.0, 10.0)
    registry.update_rect("transform", 500.0, 500.0, 10.0, 10.0)
    assert registry.hit_test(QPointF(5.0, 5.0)) is False
    assert registry.hit_test(QPointF(505.0, 505.0)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest GUIDemos/QMLOverlayApp/tests/test_panel_registry.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'panel_registry'` (file doesn't exist yet).

- [ ] **Step 3: Write `panel_registry.py`**

```python
"""QML-exposed registry of floating panel screen rects, used for click-through hit testing."""

from PySide6.QtCore import QObject, QPointF, QRectF, Slot
from PySide6.QtQml import QmlElement

QML_IMPORT_NAME = "qmloverlayapp"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class PanelRegistry(QObject):
    """Tracks each floating panel's current screen rect for hit testing."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rects: dict[str, QRectF] = {}

    @Slot(str, float, float, float, float)
    def update_rect(self, panel_id: str, x: float, y: float, w: float, h: float) -> None:
        """Record (or replace) the current screen rect of a panel.

        Args:
            panel_id: A stable identifier for the panel (e.g. its QML `objectName`).
            x: Left edge, in overlay-widget-local pixels.
            y: Top edge, in overlay-widget-local pixels.
            w: Width in pixels.
            h: Height in pixels.
        """
        self._rects[panel_id] = QRectF(x, y, w, h)

    def hit_test(self, pos: QPointF) -> bool:
        """Return True if pos falls inside any currently-registered panel rect.

        Args:
            pos: A position in overlay-widget-local pixels.

        Returns:
            True if any registered panel rect contains pos.
        """
        return any(rect.contains(pos) for rect in self._rects.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest GUIDemos/QMLOverlayApp/tests/test_panel_registry.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add GUIDemos/QMLOverlayApp/panel_registry.py GUIDemos/QMLOverlayApp/tests/test_panel_registry.py
git commit -m "feat: add PanelRegistry with hit-test unit tests for QMLOverlayApp"
```

---

## Task 6: `PyNGLScene` — matrix-driven `QOpenGLWidget` for Demo 2

**Files:**
- Create: `GUIDemos/QMLOverlayApp/PyNGLScene.py`

**Interfaces:**
- Consumes: `ncca.ngl.Mat3`, `ncca.ngl.Mat4`, `ncca.ngl.Vec3`, `ncca.ngl.perspective`, `ncca.ngl.opengl.{DefaultShader, Primitives, ShaderLib}` (same imports as `GUIDemos/PySideGUIOpenGL/PyNGLScene.py`).
- Produces: `PyNGLScene(QOpenGLWidget)` with `Slot(Mat4) set_model_matrix(m)`, `Slot(Mat4) set_view_matrix(m)`, `Slot(float, float, float) set_colour(r, g, b)`.

- [ ] **Step 1: Write `PyNGLScene.py`**

```python
"""QOpenGLWidget teapot scene driven by pre-composed matrices from ncca.ngl.qml models."""

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class PyNGLScene(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_width: int = 1024
        self.window_height: int = 720
        self._model_matrix = Mat4()
        self._view_matrix = Mat4()
        self._colour = Vec3(1.0, 1.0, 0.0)

    @Slot(Mat4)
    def set_model_matrix(self, matrix: Mat4) -> None:
        self._model_matrix = matrix
        self.update()

    @Slot(Mat4)
    def set_view_matrix(self, matrix: Mat4) -> None:
        self._view_matrix = matrix
        self.update()

    @Slot(float, float, float)
    def set_colour(self, r: float, g: float, b: float) -> None:
        self._colour = Vec3(r, g, b)
        self.update()

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.project = perspective(45.0, self.width() / self.height(), 0.1, 100.0)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.DIFFUSE)
        mv = self._view_matrix @ self._model_matrix
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform(
            "Colour", self._colour.x, self._colour.y, self._colour.z, 1.0
        )
        Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

- [ ] **Step 2: No automated test** — identical rationale to Task 2 (GL-runtime-coupled); verified visually once wired into `main.py` in Task 8.

- [ ] **Step 3: Commit**

```bash
git add GUIDemos/QMLOverlayApp/PyNGLScene.py
git commit -m "feat: add matrix-driven PyNGLScene for QMLOverlayApp demo"
```

---

## Task 7: `DraggablePanel.qml` + `main.qml` for Demo 2 (panels only, no viewport item)

**Files:**
- Create: `GUIDemos/QMLOverlayApp/DraggablePanel.qml`
- Create: `GUIDemos/QMLOverlayApp/main.qml`

**Interfaces:**
- Consumes: `PanelRegistry` (Task 5) exposed as a QML **singleton context property** named `panelRegistry` (wired up in Task 8's `main.py` via `engine.rootContext().setContextProperty("panelRegistry", registry_instance)` — a plain context property is simpler and equally correct here vs. a `pragma Singleton`, since there is exactly one instance and it's created in Python anyway).
- Produces: `DraggablePanel { id; title; content }`, same shape as Demo 1's, plus registering its geometry with `panelRegistry` on every geometry change.

- [ ] **Step 1: Write `DraggablePanel.qml`**

```qml
// GUIDemos/QMLOverlayApp/DraggablePanel.qml
import QtQuick
import QtQuick.Controls

Frame {
    id: root

    property string panelId: ""
    property string title: ""
    default property alias content: contentArea.children

    width: contentArea.implicitWidth + 16
    height: contentArea.implicitHeight + titleBar.height + 24
    opacity: 0.92

    background: Rectangle {
        color: "#2b2b2b"
        border.color: "#555555"
        radius: 6
    }

    function reportRect() {
        panelRegistry.update_rect(root.panelId, root.x, root.y, root.width, root.height)
    }

    onXChanged: reportRect()
    onYChanged: reportRect()
    onWidthChanged: reportRect()
    onHeightChanged: reportRect()
    Component.onCompleted: reportRect()

    function raiseToFront() {
        var win = root.Window.window
        if (win && win.nextZ !== undefined) {
            win.nextZ += 1
            root.z = win.nextZ
        }
    }

    Column {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 6

        Rectangle {
            id: titleBar
            width: parent.width
            height: 24
            color: "#3c3c3c"
            radius: 4

            Text {
                anchors.centerIn: parent
                text: root.title
                color: "white"
                font.bold: true
            }

            MouseArea {
                anchors.fill: parent
                drag.target: root
                onPressed: root.raiseToFront()
            }
        }

        Item {
            id: contentArea
            width: parent.width
            height: childrenRect.height
        }
    }
}
```

- [ ] **Step 2: Write `main.qml`**

```qml
// GUIDemos/QMLOverlayApp/main.qml
import QtQuick
import QtQuick.Window
import ncca.ngl.qml 1.0

Item {
    id: overlayRoot
    property int nextZ: 100

    TransformWidget {
        id: transformWidget
        name: "Transform"
        onValueChanged: pyNGLScene.set_model_matrix(model.get_matrix())
    }
    DraggablePanel {
        panelId: "transform"
        title: "Transform"
        x: 30
        y: 30
        content: [transformWidget]
    }

    RGBColourWidget {
        id: rgbWidget
        name: "Colour"
        onColourChanged: {
            var c = model.get_value()
            pyNGLScene.set_colour(c.x, c.y, c.z)
        }
    }
    DraggablePanel {
        panelId: "colour"
        title: "Colour"
        x: 30
        y: 260
        content: [rgbWidget]
    }

    LookAtWidget {
        id: lookAtWidget
        name: "Camera"
        onValueChanged: pyNGLScene.set_view_matrix(model.get_matrix())
    }
    DraggablePanel {
        panelId: "camera"
        title: "Camera"
        x: 30
        y: 360
        content: [lookAtWidget]
    }

    Component.onCompleted: {
        pyNGLScene.set_model_matrix(transformWidget.model.get_matrix())
        pyNGLScene.set_view_matrix(lookAtWidget.model.get_matrix())
        var c = rgbWidget.model.get_value()
        pyNGLScene.set_colour(c.x, c.y, c.z)
    }
}
```

> Note: `main.qml` here is a root `Item`, not an `ApplicationWindow` — the actual top-level window is the Python `QMainWindow` created in Task 8; this QML file is loaded into a `QQuickWidget` embedded inside it. `nextZ` therefore lives on `overlayRoot` (the root `Item`), and `DraggablePanel.raiseToFront()` (Task copied verbatim from Demo 1 but referencing `root.Window.window`) needs its window lookup to still resolve `nextZ` — `Window.window` on an `Item` inside a `QQuickWidget` resolves to the `QQuickWindow` the widget owns, which does **not** have a `nextZ` property. Fix: change `raiseToFront()` in this copy of `DraggablePanel.qml` to walk to `overlayRoot` instead of `Window.window`:
> ```qml
> function raiseToFront() {
>     var p = root.parent
>     while (p && p.nextZ === undefined) p = p.parent
>     if (p) {
>         p.nextZ += 1
>         root.z = p.nextZ
>     }
> }
> ```
> Apply this version of `raiseToFront()` in Step 1 above (already reflected in the code block).

- [ ] **Step 3: Commit**

```bash
git add GUIDemos/QMLOverlayApp/DraggablePanel.qml GUIDemos/QMLOverlayApp/main.qml
git commit -m "feat: add panel-registry-aware DraggablePanel and main.qml for QMLOverlayApp"
```

---

## Task 8: `main.py` — `QMainWindow` + click-through `QQuickWidget` overlay

**Files:**
- Create: `GUIDemos/QMLOverlayApp/main.py`

**Interfaces:**
- Consumes: `PyNGLScene` (Task 6), `PanelRegistry` (Task 5), `DraggablePanel.qml`/`main.qml` (Task 7).

- [ ] **Step 1: Write `main.py`**

```python
#!/usr/bin/env -S uv run --script
# GUIDemos/QMLOverlayApp/main.py
import sys
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml widget types)
from panel_registry import PanelRegistry
from PySide6.QtCore import QEvent, QUrl, Qt
from PySide6.QtGui import QMouseEvent, QSurfaceFormat
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QMainWindow
from PyNGLScene import PyNGLScene


class OverlayQuickWidget(QQuickWidget):
    """Transparent QQuickWidget that forwards clicks outside any panel to the GL widget beneath."""

    def __init__(self, scene: PyNGLScene, registry: PanelRegistry, parent=None) -> None:
        super().__init__(parent)
        self._scene = scene
        self._registry = registry
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setClearColor(Qt.GlobalColor.transparent)
        self.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)

    def _forward_to_scene(self, event: QMouseEvent) -> None:
        forwarded = QMouseEvent(
            event.type(),
            self._scene.mapFromGlobal(event.globalPosition()),
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        QApplication.sendEvent(self._scene, forwarded)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mousePressEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mouseMoveEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mouseReleaseEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self.scene = PyNGLScene()
        self.setCentralWidget(self.scene)

        self.registry = PanelRegistry()
        self.overlay = OverlayQuickWidget(self.scene, self.registry, self.scene)
        self.overlay.rootContext().setContextProperty("panelRegistry", self.registry)
        self.overlay.rootContext().setContextProperty("pyNGLScene", self.scene)
        self.overlay.setSource(QUrl.fromLocalFile(str(Path(__file__).parent / "main.qml")))
        self.overlay.setGeometry(self.scene.rect())

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self.overlay.setGeometry(self.scene.rect())


def main() -> int:
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setMajorVersion(4)
    fmt.setMinorVersion(1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and verify manually**

Run: `chmod +x GUIDemos/QMLOverlayApp/main.py && uv run GUIDemos/QMLOverlayApp/main.py`
Expected: a window with a grey `PyNGLScene` teapot viewport filling the whole window, and three draggable panels ("Transform", "Colour", "Camera") floating on top with visible teapot showing through the gaps between them (translucent panel backgrounds, fully transparent empty overlay area). Moving panel controls updates the teapot live, same as Demo 1. Click-dragging a panel's title bar moves only that panel. Clicking on the teapot itself (outside any panel rect) should not be swallowed by the (invisible, empty) Qt Quick surface — verify by resizing the window and confirming the teapot viewport still resizes/repaints correctly and no stray click-blocking occurs over empty overlay regions.

- [ ] **Step 3: Run the existing unit test suite once more to confirm nothing broke**

Run: `uv run pytest GUIDemos/QMLOverlayApp/tests/test_panel_registry.py -v`
Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLOverlayApp/main.py
git commit -m "feat: add QMainWindow with click-through QQuickWidget overlay for QMLOverlayApp"
```

---

## Task 9: `QMLOverlayApp` README + screenshot + root README link

**Files:**
- Create: `GUIDemos/QMLOverlayApp/README.md`
- Create: `GUIDemos/QMLOverlayApp/QMLOverlayApp.png` (manual screenshot)
- Modify: `README.md` (repo root)

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Take a screenshot**

Run `uv run GUIDemos/QMLOverlayApp/main.py`, arrange the panels, screenshot, save as `GUIDemos/QMLOverlayApp/QMLOverlayApp.png`.

- [ ] **Step 3: Add a row to the root README's demo table**

Add immediately after the `QMLFloatingWidgets` row added in Task 4:

```markdown
| <a href="GUIDemos/QMLOverlayApp"><img src="GUIDemos/QMLOverlayApp/QMLOverlayApp.png" width="220"></a> | [GUIDemos/QMLOverlayApp](GUIDemos/QMLOverlayApp) | QWidget OpenGL viewport with a transparent QQuickWidget overlay of floating ncca.ngl.qml panels |
```

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLOverlayApp/README.md GUIDemos/QMLOverlayApp/QMLOverlayApp.png README.md
git commit -m "docs: add README, screenshot and root link for QMLOverlayApp demo"
```

---

## Task 10: Repo-wide lint pass and final verification

**Files:** all files created above.

- [ ] **Step 1: Run ruff import sort + format across both new folders**

Run: `uv run ruff check --select I --fix GUIDemos/QMLFloatingWidgets GUIDemos/QMLOverlayApp && uv run ruff format GUIDemos/QMLFloatingWidgets GUIDemos/QMLOverlayApp`
Expected: no errors; any reformatting is applied in place.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass, including the 4 new `test_panel_registry.py` tests, with no regressions in other demos' existing tests.

- [ ] **Step 3: Commit any lint fixes (only if ruff changed something)**

```bash
git add -u GUIDemos/QMLFloatingWidgets GUIDemos/QMLOverlayApp
git commit -m "style: ruff format/import-sort QML floating-widget demos"
```
