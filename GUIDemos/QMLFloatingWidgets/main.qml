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
