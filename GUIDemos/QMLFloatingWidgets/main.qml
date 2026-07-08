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
