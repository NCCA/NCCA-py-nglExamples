# QML Panel Resize Handles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the floating QML panels in `GUIDemos/QMLOverlayApp` and `GUIDemos/QMLWebGPUOverlay` be resized by dragging their edges/corners, not just repositioned.

**Architecture:** All logic lives in the shared `DraggablePanel.qml` (currently byte-identical in both demos): 8 transparent edge/corner `MouseArea`-based handles clamp width/height against a content-derived minimum, and `contentArea` switches from center-anchored to top-left-anchored so growth adds empty space rather than stretching controls. Each demo's `main.qml` gets `width`/`height` `Settings` aliases per panel, mirroring the existing `x`/`y` aliases, so resized sizes persist like dragged positions already do.

**Tech Stack:** PySide6 (QtQuick/QML), `ncca.ngl.qml`, `uv run`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-qml-panel-resize-design.md`.
- No Python changes — `panel_registry.py`'s `reportRect()`/`hit_test` already operate on whatever `width`/`height` a panel reports.
- `DraggablePanel.qml` is byte-identical in both demos today; keep it byte-identical after this change (task 3 syncs by copy, not independent edits).
- A panel must never be draggable smaller than `contentArea.width/height + 16` (its natural content size plus the existing 16px chrome margin).
- Enlarging a panel must add empty space, not stretch child controls (`contentArea` keeps its natural size).
- No new automated tests are introduced — this is QML interaction logic; the existing Python test suite (`GUIDemos/QMLOverlayApp/tests/test_panel_registry.py`) must still pass unmodified. Verification is by launching each demo and confirming no QML errors, plus a manual interactive checklist (final task).
- Work happens in the worktree `.worktrees/qml-panel-resize` on branch `agent/qml-panel-resize`. All file paths below are relative to that worktree root (i.e. `/Volumes/teaching/Code/PyNGLDemos/.worktrees/qml-panel-resize/`).

---

### Task 1: Content-driven minimum size + top-left content anchoring

**Files:**
- Modify: `GUIDemos/QMLOverlayApp/DraggablePanel.qml`

**Interfaces:**
- Produces: `root.minimumWidth`, `root.minimumHeight` (readonly `real` properties on `DraggablePanel`'s root `Frame`) — later tasks (2, 4) clamp against these.
- Produces: `contentArea` now anchored top-left (was `anchors.centerIn: parent`) — task 2's handles rely on the panel having empty space to the right/below when enlarged, not stretched content.

This task only adds the minimum-size properties and changes content anchoring; it introduces no new interaction yet, so panel appearance and behavior are unchanged (natural size == minimum size on first run, so top-left anchoring with 8px margins renders identically to the previous centered anchoring).

- [ ] **Step 1: Read the current file to confirm line numbers before editing**

Run: `sed -n '1,90p' GUIDemos/QMLOverlayApp/DraggablePanel.qml`

Confirm the `dragEnabled` property (around line 25), the `reportRect`/`Component.onCompleted` block (around lines 43-51), and the `contentArea` `Item` (around lines 84-89) match what's shown below. If line numbers have drifted, locate the same blocks by content instead.

- [ ] **Step 2: Add `minimumWidth`/`minimumHeight` properties**

Find:

```qml
    property bool dragEnabled: true
```

Replace with:

```qml
    property bool dragEnabled: true

    // The smallest this panel can be dragged to: content's own natural
    // size (contentArea never itself resizes) plus the 16px chrome margin
    // baked into the width/height bindings below. Resize handles (added in
    // a later change) clamp against these so content never clips.
    readonly property real minimumWidth: contentArea.width + 16
    readonly property real minimumHeight: contentArea.height + 16
```

- [ ] **Step 3: Clamp restored/initial size in `Component.onCompleted`**

Find:

```qml
    onXChanged: reportRect()
    onYChanged: reportRect()
    onWidthChanged: reportRect()
    onHeightChanged: reportRect()
    Component.onCompleted: reportRect()
```

Replace with:

```qml
    onXChanged: reportRect()
    onYChanged: reportRect()
    onWidthChanged: reportRect()
    onHeightChanged: reportRect()
    Component.onCompleted: {
        // Only assign (which would break the width/height content-size
        // bindings) if a persisted value from a previous run undershoots
        // what the content now needs - e.g. a panel gained a new control
        // since the size was last saved. On a fresh panel this is a no-op:
        // width/height already equal minimumWidth/minimumHeight, so the
        // strict "<" leaves the live binding intact.
        if (width < minimumWidth) width = minimumWidth
        if (height < minimumHeight) height = minimumHeight
        reportRect()
    }
```

- [ ] **Step 4: Switch `contentArea` to top-left anchoring**

Find:

```qml
    Item {
        id: contentArea
        anchors.centerIn: parent
        width: childrenRect.width
        height: childrenRect.height
    }
```

Replace with:

```qml
    Item {
        id: contentArea
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 8
        width: childrenRect.width
        height: childrenRect.height
    }
```

- [ ] **Step 5: Sanity-check the demo still launches without QML errors**

Run: `cd GUIDemos/QMLOverlayApp && timeout 5 uv run main.py; echo "exit=$?"` (run from the worktree root, adjust `cd` back afterwards, e.g. `cd -` or use an absolute path)

Expected: the process starts the GUI (no `qrc:`/`file:` QML error output on stderr about `DraggablePanel.qml`), and is killed by `timeout` after 5s (`exit=124`). A QML parse/binding error would instead print `QQmlApplicationEngine failed to load component` or similar to stderr immediately.

- [ ] **Step 6: Commit**

```bash
git add GUIDemos/QMLOverlayApp/DraggablePanel.qml
git commit -m "feat(qml demos): add content-driven minimum panel size"
```

---

### Task 2: Add the 8 resize handles to `DraggablePanel.qml`

**Files:**
- Modify: `GUIDemos/QMLOverlayApp/DraggablePanel.qml`

**Interfaces:**
- Consumes: `root.minimumWidth`, `root.minimumHeight` from Task 1.
- Produces: drag-to-resize behavior on all 4 edges + 4 corners of every `DraggablePanel` instance. No new properties or signals exposed to callers — `main.qml` needs no changes for this task.

- [ ] **Step 1: Add the `handleSize` property and `ResizeHandle` component**

Find:

```qml
    TapHandler {
        onTapped: root.raiseToFront()
    }
```

Replace with:

```qml
    TapHandler {
        onTapped: root.raiseToFront()
    }

    // Thickness of the invisible edge/corner grab strips, in px.
    property int handleSize: 8

    // One reusable resize-handle definition, configured per instance below
    // by which edge(s) it drags. Tracks the press position and the panel's
    // x/y/width/height *at press time* in root.parent's (overlayRoot's)
    // coordinate space, so each move event recomputes the new geometry
    // from a fixed reference rather than accumulating per-event deltas -
    // this stays correct even though left/top drags move `root` itself
    // out from under the handle while dragging.
    component ResizeHandle: MouseArea {
        id: handle
        property bool resizeLeft: false
        property bool resizeRight: false
        property bool resizeTop: false
        property bool resizeBottom: false
        property point pressGlobal
        property real pressPanelX
        property real pressPanelY
        property real pressWidth
        property real pressHeight

        hoverEnabled: true
        cursorShape: {
            if ((resizeLeft && resizeTop) || (resizeRight && resizeBottom))
                return Qt.SizeFDiagCursor
            if ((resizeRight && resizeTop) || (resizeLeft && resizeBottom))
                return Qt.SizeBDiagCursor
            if (resizeLeft || resizeRight)
                return Qt.SizeHorCursor
            return Qt.SizeVerCursor
        }

        onPressed: (mouse) => {
            pressGlobal = mapToItem(root.parent, mouse.x, mouse.y)
            pressPanelX = root.x
            pressPanelY = root.y
            pressWidth = root.width
            pressHeight = root.height
            root.raiseToFront()
        }
        onPositionChanged: (mouse) => {
            if (!pressed)
                return
            var current = mapToItem(root.parent, mouse.x, mouse.y)
            var dx = current.x - pressGlobal.x
            var dy = current.y - pressGlobal.y

            if (resizeRight)
                root.width = Math.max(root.minimumWidth, pressWidth + dx)
            if (resizeBottom)
                root.height = Math.max(root.minimumHeight, pressHeight + dy)
            if (resizeLeft) {
                var newWidth = Math.max(root.minimumWidth, pressWidth - dx)
                root.x = pressPanelX + (pressWidth - newWidth)
                root.width = newWidth
            }
            if (resizeTop) {
                var newHeight = Math.max(root.minimumHeight, pressHeight - dy)
                root.y = pressPanelY + (pressHeight - newHeight)
                root.height = newHeight
            }
        }
    }
```

- [ ] **Step 2: Add the 8 handle instances after `contentArea`**

Find:

```qml
    Item {
        id: contentArea
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 8
        width: childrenRect.width
        height: childrenRect.height
    }
}
```

Replace with:

```qml
    Item {
        id: contentArea
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 8
        width: childrenRect.width
        height: childrenRect.height
    }

    // Edges first, corners last: corners are declared later in the same
    // Item so they win Qt Quick's topmost-hit-wins arbitration over the
    // edge strips in the small squares where both would otherwise overlap.
    ResizeHandle {
        resizeTop: true
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: root.handleSize
        anchors.rightMargin: root.handleSize
        height: root.handleSize
    }
    ResizeHandle {
        resizeBottom: true
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: root.handleSize
        anchors.rightMargin: root.handleSize
        height: root.handleSize
    }
    ResizeHandle {
        resizeLeft: true
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.topMargin: root.handleSize
        anchors.bottomMargin: root.handleSize
        width: root.handleSize
    }
    ResizeHandle {
        resizeRight: true
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.topMargin: root.handleSize
        anchors.bottomMargin: root.handleSize
        width: root.handleSize
    }
    ResizeHandle {
        resizeLeft: true
        resizeTop: true
        anchors.left: parent.left
        anchors.top: parent.top
        width: root.handleSize
        height: root.handleSize
    }
    ResizeHandle {
        resizeRight: true
        resizeTop: true
        anchors.right: parent.right
        anchors.top: parent.top
        width: root.handleSize
        height: root.handleSize
    }
    ResizeHandle {
        resizeLeft: true
        resizeBottom: true
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        width: root.handleSize
        height: root.handleSize
    }
    ResizeHandle {
        resizeRight: true
        resizeBottom: true
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        width: root.handleSize
        height: root.handleSize
    }
}
```

- [ ] **Step 3: Sanity-check the demo still launches without QML errors**

Run: `cd GUIDemos/QMLOverlayApp && timeout 5 uv run main.py; echo "exit=$?"`

Expected: same as Task 1 Step 5 — no QML error output, `exit=124`.

- [ ] **Step 4: Manually verify resize behavior interactively**

Run: `cd GUIDemos/QMLOverlayApp && uv run main.py` (foreground, interactive)

Check, then close the window:
- Hovering each of the 4 edges and 4 corners of the Transform panel shows the correct OS resize cursor (horizontal/vertical/diagonal).
- Dragging each edge/corner resizes the panel in the expected direction(s); dragging a corner resizes both dimensions together.
- Dragging a panel's body (not an edge/corner) still moves it, not resizes it.
- The panel cannot be shrunk smaller than its content (dragging past the minimum stops at the content's natural size, content never clips).
- Enlarging a panel adds empty space around the content rather than stretching the spin boxes/labels inside it.
- Repeat briefly on the Camera and Style panels; confirm the Style panel's dropdown still opens and dragging is still suspended while it's open (`dragEnabled: !styleCombo.popup.visible` still governs the move-drag; resize handles are independent of it by design — note if this feels wrong, see Task 6's checklist for the final call).

If any check fails, fix `DraggablePanel.qml` before proceeding — do not move to Step 5 with known-broken resize behavior.

- [ ] **Step 5: Commit**

```bash
git add GUIDemos/QMLOverlayApp/DraggablePanel.qml
git commit -m "feat(qml demos): add edge/corner resize handles to DraggablePanel"
```

---

### Task 3: Sync `DraggablePanel.qml` to `QMLWebGPUOverlay`

**Files:**
- Modify: `GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml` (overwritten by copy)

**Interfaces:**
- Consumes: the finished `GUIDemos/QMLOverlayApp/DraggablePanel.qml` from Task 2.
- Produces: nothing new — this keeps the two demos' copies byte-identical, as they were before this plan (confirmed via `diff` during design).

- [ ] **Step 1: Confirm the two files were identical before this plan's changes**

Run: `git diff --stat HEAD~2 -- GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml`

Expected: no output (the WebGPU copy hasn't been touched yet by this plan, so it should still match the pre-Task-1 `QMLOverlayApp` version).

- [ ] **Step 2: Copy the updated file over**

Run: `cp GUIDemos/QMLOverlayApp/DraggablePanel.qml GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml`

- [ ] **Step 3: Confirm the two copies are now identical again**

Run: `diff GUIDemos/QMLOverlayApp/DraggablePanel.qml GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml; echo "exit=$?"`

Expected: no diff output, `exit=0`.

- [ ] **Step 4: Sanity-check the WebGPU demo still launches without QML errors**

Run: `cd GUIDemos/QMLWebGPUOverlay && timeout 5 uv run main.py; echo "exit=$?"`

Expected: no QML error output, `exit=124`. (A wgpu-capable GPU/driver is assumed available, same as for any other manual run of this demo.)

- [ ] **Step 5: Manually verify resize behavior interactively (spot-check, not the full Task 2 Step 4 checklist)**

Run: `cd GUIDemos/QMLWebGPUOverlay && uv run main.py` (foreground, interactive)

Check, then close the window:
- Drag one corner and one edge of the Transform panel; confirm resize works the same as it did in `QMLOverlayApp`.

- [ ] **Step 6: Commit**

```bash
git add GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml
git commit -m "feat(qml demos): sync resizable DraggablePanel into QMLWebGPUOverlay"
```

---

### Task 4: Persist panel size in `QMLOverlayApp/main.qml`

**Files:**
- Modify: `GUIDemos/QMLOverlayApp/main.qml:192-203`

**Interfaces:**
- Consumes: Task 1's `Component.onCompleted` clamp (`if (width < minimumWidth) width = minimumWidth`), which handles the case where a persisted size undershoots current content needs.
- Produces: nothing new consumed elsewhere — this is a leaf change (the `Settings` block).

- [ ] **Step 1: Add `width`/`height` aliases to the `Settings` block**

Find:

```qml
    Settings {
        category: "layout"
        property alias styleIndex: overlayRoot.styleIndex
        property alias transformX: transformPanel.x
        property alias transformY: transformPanel.y
        property alias colourX: colourPanel.x
        property alias colourY: colourPanel.y
        property alias cameraX: cameraPanel.x
        property alias cameraY: cameraPanel.y
        property alias styleX: stylePanel.x
        property alias styleY: stylePanel.y
    }
```

Replace with:

```qml
    Settings {
        category: "layout"
        property alias styleIndex: overlayRoot.styleIndex
        property alias transformX: transformPanel.x
        property alias transformY: transformPanel.y
        property alias transformWidth: transformPanel.width
        property alias transformHeight: transformPanel.height
        property alias colourX: colourPanel.x
        property alias colourY: colourPanel.y
        property alias colourWidth: colourPanel.width
        property alias colourHeight: colourPanel.height
        property alias cameraX: cameraPanel.x
        property alias cameraY: cameraPanel.y
        property alias cameraWidth: cameraPanel.width
        property alias cameraHeight: cameraPanel.height
        property alias styleX: stylePanel.x
        property alias styleY: stylePanel.y
        property alias styleWidth: stylePanel.width
        property alias styleHeight: stylePanel.height
    }
```

- [ ] **Step 2: Sanity-check the demo still launches without QML errors**

Run: `cd GUIDemos/QMLOverlayApp && timeout 5 uv run main.py; echo "exit=$?"`

Expected: no QML error output, `exit=124`.

- [ ] **Step 3: Manually verify persistence**

Run: `cd GUIDemos/QMLOverlayApp && uv run main.py` (foreground, interactive)

1. Resize the Transform panel noticeably larger, then close the window (quitting cleanly, not killing the process, so `Settings` writes out).
2. Relaunch: `uv run main.py`.
3. Confirm the Transform panel opens at the resized size, not its default content-fit size.
4. Close the window.

If persistence doesn't hold, check `app.setOrganizationName("NCCA")` / `app.setApplicationName("QMLOverlayApp")` in `main.py` are still present (they set the `QSettings` storage path) before debugging further.

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLOverlayApp/main.qml
git commit -m "feat(qml demos): persist resized panel sizes in QMLOverlayApp"
```

---

### Task 5: Persist panel size in `QMLWebGPUOverlay/main.qml`

**Files:**
- Modify: `GUIDemos/QMLWebGPUOverlay/main.qml:192-203`

**Interfaces:**
- Consumes: same as Task 4, applied to the WebGPU demo's copy of `main.qml`.
- Produces: nothing new — mirrors Task 4.

- [ ] **Step 1: Add `width`/`height` aliases to the `Settings` block**

Find:

```qml
    Settings {
        category: "layout"
        property alias styleIndex: overlayRoot.styleIndex
        property alias transformX: transformPanel.x
        property alias transformY: transformPanel.y
        property alias colourX: colourPanel.x
        property alias colourY: colourPanel.y
        property alias cameraX: cameraPanel.x
        property alias cameraY: cameraPanel.y
        property alias styleX: stylePanel.x
        property alias styleY: stylePanel.y
    }
```

Replace with:

```qml
    Settings {
        category: "layout"
        property alias styleIndex: overlayRoot.styleIndex
        property alias transformX: transformPanel.x
        property alias transformY: transformPanel.y
        property alias transformWidth: transformPanel.width
        property alias transformHeight: transformPanel.height
        property alias colourX: colourPanel.x
        property alias colourY: colourPanel.y
        property alias colourWidth: colourPanel.width
        property alias colourHeight: colourPanel.height
        property alias cameraX: cameraPanel.x
        property alias cameraY: cameraPanel.y
        property alias cameraWidth: cameraPanel.width
        property alias cameraHeight: cameraPanel.height
        property alias styleX: stylePanel.x
        property alias styleY: stylePanel.y
        property alias styleWidth: stylePanel.width
        property alias styleHeight: stylePanel.height
    }
```

- [ ] **Step 2: Sanity-check the demo still launches without QML errors**

Run: `cd GUIDemos/QMLWebGPUOverlay && timeout 5 uv run main.py; echo "exit=$?"`

Expected: no QML error output, `exit=124`.

- [ ] **Step 3: Manually verify persistence**

Same procedure as Task 4 Step 3, run from `GUIDemos/QMLWebGPUOverlay`.

- [ ] **Step 4: Commit**

```bash
git add GUIDemos/QMLWebGPUOverlay/main.qml
git commit -m "feat(qml demos): persist resized panel sizes in QMLWebGPUOverlay"
```

---

### Task 6: Full verification pass + existing test suite

**Files:**
- None modified — verification only.

**Interfaces:**
- Consumes: the finished state of all prior tasks.
- Produces: nothing — this is the plan's final gate before handoff.

- [ ] **Step 1: Run the existing Python test suite to confirm nothing broke**

Run: `uv run pytest GUIDemos/QMLOverlayApp/tests/test_panel_registry.py -v`

Expected: all existing tests still `PASS` (this plan makes no Python changes, so this is a regression check, not new coverage).

- [ ] **Step 2: Run the repo-wide test suite**

Run: `uv run pytest`

Expected: no new failures relative to the pre-change baseline (any pre-existing unrelated failures are out of scope for this plan).

- [ ] **Step 3: Full manual QA checklist — `QMLOverlayApp`**

Run: `cd GUIDemos/QMLOverlayApp && uv run main.py`

Walk through every item from the spec's Testing section:
- [ ] Drag each of the 8 handles on each of the 4 panels (Transform, Colour, Camera, Style); resize happens in the expected direction(s).
- [ ] No panel can be shrunk below its content's natural size.
- [ ] Resizing a panel larger leaves empty space rather than stretching its controls.
- [ ] Dragging a panel's body still moves it; only edges/corners resize.
- [ ] The Style panel's `dragEnabled: !styleCombo.popup.visible` suppression still prevents accidental moves while its dropdown is open (open the dropdown, try dragging the panel body — it should not move).
- [ ] Resize a panel, close the app, relaunch, confirm the size persisted (already covered in Task 4 Step 3, but re-confirm as part of this end-to-end pass).
- [ ] Click inside and outside a resized panel; confirm clicks route to the QML overlay vs. the GL scene correctly (`panel_registry` hit-testing still tracks the new size).

- [ ] **Step 4: Full manual QA checklist — `QMLWebGPUOverlay`**

Run: `cd GUIDemos/QMLWebGPUOverlay && uv run main.py`

Repeat the same checklist as Step 3.

- [ ] **Step 5: Take a screenshot of each demo showing a resized panel**

Per this repo's `CLAUDE.md` convention ("Ensure a screen shot of the demo running is included in folder along with the README"), capture one screenshot per demo with at least one panel visibly resized (larger or smaller than its default), and update:
- `GUIDemos/QMLOverlayApp/QMLOverlayApp.png`
- `GUIDemos/QMLWebGPUOverlay/QMLWebGPUOverlay.png`

- [ ] **Step 6: Final commit if Step 5 changed any files**

```bash
git add GUIDemos/QMLOverlayApp/QMLOverlayApp.png GUIDemos/QMLWebGPUOverlay/QMLWebGPUOverlay.png
git commit -m "docs(qml demos): update screenshots to show resizable panels"
```

If Step 5's screenshots are unchanged from before (panel resizing doesn't materially change the default screenshot), skip this commit.
