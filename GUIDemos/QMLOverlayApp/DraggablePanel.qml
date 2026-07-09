// GUIDemos/QMLOverlayApp/DraggablePanel.qml
import QtQuick
import QtQuick.Controls

Frame {
    id: root

    property string panelId: ""
    default property alias content: contentArea.children

    // Panel chrome colours, exposed so a theme can restyle the panel body to
    // match the palette applied to its child controls (see main.qml's
    // stylePresets - e.g. the Dear ImGui themes pair a dark/blue panel body
    // with matching control colours). Defaults reproduce the original look.
    property color panelColor: "#2b2b2b"
    property color panelBorder: "#555555"

    // Callers with a popup-bearing child (e.g. a ComboBox) should bind this
    // to false while that popup is open. grabPermissions below reduces the
    // odds of the DragHandler stealing a click meant for the popup, but a
    // popup's contents are reparented into the window's Overlay layer - a
    // separate branch from this panel's own children - so it isn't reliably
    // covered by the normal "descendant grabs first" arbitration. Disabling
    // the handler outright removes the ambiguity instead of depending on it.
    property bool dragEnabled: true

    // The smallest this panel can be dragged to: content's own natural
    // size (contentArea never itself resizes) plus the 16px chrome margin
    // baked into the width/height bindings below. Resize handles (added in
    // a later change) clamp against these so content never clips.
    readonly property real minimumWidth: contentArea.width + 16
    readonly property real minimumHeight: contentArea.height + 16

    // Frame/Pane applies its own implicit padding around contentItem by
    // default, which insets everything declared inside it - including the
    // DragHandler below - away from the panel's outer edge. Zero it out so
    // the drag area covers the full panel, edge to edge.
    padding: 0

    width: contentArea.width + 16
    height: contentArea.height + 16
    opacity: 0.92

    background: Rectangle {
        color: root.panelColor
        border.color: root.panelBorder
        radius: 6
    }

    function reportRect() {
        panelRegistry.update_rect(root.panelId, root.x, root.y, root.width, root.height)
    }

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

    function raiseToFront() {
        var p = root.parent
        while (p && p.nextZ === undefined) p = p.parent
        if (p) {
            p.nextZ += 1
            root.z = p.nextZ
        }
    }

    // DragHandler (not a plain MouseArea) so it participates in Qt Quick's
    // pointer grab arbitration: it only wins the drag if no descendant
    // control (spin box, combo box, colour swatch...) already grabbed the
    // press first, so the whole panel - background, labels, padding - is
    // draggable without stealing clicks from interactive child widgets.
    DragHandler {
        target: root
        enabled: root.dragEnabled
        // Default threshold comes from the platform's drag-start distance
        // (often 10-15px), which reads as sluggish for a small floating
        // panel - shrink the dead zone so the drag catches immediately.
        dragThreshold: 2
        // Belt-and-braces alongside dragEnabled above: never take over a
        // grab another item/handler already holds (e.g. a spin box mid
        // drag), only ever grab when nothing else wants the press.
        grabPermissions: PointerHandler.TakeOverForbidden
        onActiveChanged: if (active) root.raiseToFront()
    }
    TapHandler {
        onTapped: root.raiseToFront()
    }

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
