# QML Panel Resize Handles

## Problem

`DraggablePanel.qml` (shared, byte-identical, by `GUIDemos/QMLOverlayApp` and
`GUIDemos/QMLWebGPUOverlay`) can be dragged to reposition, but its size is
fixed to `contentArea.width/height + 16` — there is no way to grab and
resize a panel.

## Goal

Add drag-to-resize to `DraggablePanel.qml` so all four panels (Transform,
Colour, Camera, Style) in both demos can be resized from any edge or
corner, without breaking existing move/drag, popup-suppression, or
hit-testing behaviour.

## Scope

- `GUIDemos/QMLOverlayApp/DraggablePanel.qml`
- `GUIDemos/QMLOverlayApp/main.qml` (Settings block only)
- `GUIDemos/QMLWebGPUOverlay/DraggablePanel.qml`
- `GUIDemos/QMLWebGPUOverlay/main.qml` (Settings block only)

No Python changes. `panel_registry.py`'s `reportRect()` / `hit_test`
already operate on whatever `width`/`height` the panel reports, so resizing
works with the existing hit-testing unmodified.

## Design

### Resize handles

Add 8 small transparent `Item`s overlaid on `DraggablePanel`'s edges and
corners (top, bottom, left, right, top-left, top-right, bottom-left,
bottom-right). Each has:

- A `HoverHandler` setting the appropriate OS resize cursor
  (`Qt.SizeVerCursor` / `Qt.SizeHorCursor` / `Qt.SizeFDiagCursor` /
  `Qt.SizeBDiagCursor`).
- A `DragHandler` (or `MouseArea` with tracked press/delta, whichever
  proves simpler in practice) that adjusts `root.width`/`root.height` and,
  for the top/left-involving handles, `root.x`/`root.y` so the opposite
  edge/corner stays anchored while dragging.

These handle items are declared as children of `root`, after the existing
move `DragHandler`, so — matching the existing "descendant grabs first"
arbitration already relied on for the panel's interactive controls — a
press directly on a handle resizes, and a press elsewhere on the panel
body still moves it.

### Minimum size / clamping

`contentArea`'s `width`/`height` (already bound to `childrenRect.width` /
`childrenRect.height`) remain untouched and become the standing reference
for "natural content size" — this binding is never broken by a resize.

Add two new properties on `root`:

```qml
readonly property real minimumWidth: contentArea.width + 16
readonly property real minimumHeight: contentArea.height + 16
```

Every resize handler clamps the new width/height against these before
assigning, so a panel can never be shrunk below what its content needs.

### Oversize behaviour

`contentArea` switches from `anchors.centerIn: parent` to top-left
anchoring:

```qml
anchors.left: parent.left
anchors.top: parent.top
anchors.leftMargin: 8
anchors.topMargin: 8
```

Content keeps its natural size; enlarging the panel adds empty space to
the right/below rather than stretching child controls.

### Binding-break mechanics

`root.width`/`root.height` currently have live bindings
(`contentArea.width + 16`, `contentArea.height + 16`). Assigning to a
bound property in QML breaks the binding. No special-case "has the user
resized yet" flag is needed: panels stay auto-sized-to-content until the
first resize drag, at which point the binding silently breaks and the
size becomes a fixed, manually-controlled value from then on.

### Persistence

Add `width`/`height` aliases per panel to the existing
`Settings { category: "layout" }` block in each demo's `main.qml`,
mirroring the existing `x`/`y` aliases, e.g.:

```qml
property alias transformWidth: transformPanel.width
property alias transformHeight: transformPanel.height
```

On first run (no stored value) panels still auto-size to content as
today. After any manual resize, the concrete size is saved and restored
on restart (also breaking the auto-size binding on load, same as a live
resize would).

On load, clamp the restored size up to `minimumWidth`/`minimumHeight` in
`Component.onCompleted`, in case a panel's content has grown (e.g. a new
control added later) since the value was saved:

```qml
Component.onCompleted: {
    width = Math.max(width, minimumWidth)
    height = Math.max(height, minimumHeight)
}
```

### Scope of change within DraggablePanel.qml

All logic lives inside `DraggablePanel.qml` itself, so all four panels in
both demos get resizing automatically — no per-panel opt-in property.

## Testing

This is QML-only interaction logic; no new Python is introduced, so no
new automated tests are practical (matches the existing test suite, which
covers only `panel_registry.py`'s pure-Python logic in
`tests/test_panel_registry.py`).

Verification is manual, run in both `QMLOverlayApp` and `QMLWebGPUOverlay`:

- Drag each of the 8 handles on each of the 4 panels; confirm resize in
  the expected direction(s).
- Confirm a panel cannot be shrunk below its content's natural size.
- Confirm resizing a panel larger leaves empty space rather than
  stretching its controls.
- Confirm resize does not fight the existing move-drag (dragging the body
  still moves; only the edges/corners resize).
- Confirm the Style panel's `dragEnabled: !styleCombo.popup.visible`
  suppression still prevents accidental moves/resizes while its dropdown
  is open.
- Resize a panel, restart the demo, confirm the size persisted.
- Confirm `panel_registry` hit-testing still works correctly against a
  resized panel (clicks inside/outside route correctly).

## Out of scope

- Per-panel opt-in/opt-out of resizing.
- Content that reflows/stretches to fill a larger panel.
- Automated QML interaction tests.
