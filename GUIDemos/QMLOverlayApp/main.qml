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
