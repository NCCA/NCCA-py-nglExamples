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
        var p = root.parent
        while (p && p.nextZ === undefined) p = p.parent
        if (p) {
            p.nextZ += 1
            root.z = p.nextZ
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
