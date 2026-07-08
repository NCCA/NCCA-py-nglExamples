// GUIDemos/QMLFloatingWidgets/DraggablePanel.qml
import QtQuick
import QtQuick.Controls

Frame {
    id: root

    property string title: ""
    default property alias content: contentArea.children

    // NOTE: contentArea.implicitWidth/implicitHeight never populate on a
    // plain Item (only layout-aware types compute those from children), so
    // width/height below always collapse to ~16x48 regardless of visible
    // content. This was found and fixed in the sibling GUIDemos/QMLOverlayApp
    // copy of this file (sized from contentArea.childrenRect instead) after
    // it caused a real click-routing bug there. Left as-is here since this
    // demo is kept only as a documented reference, not a working one.
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
