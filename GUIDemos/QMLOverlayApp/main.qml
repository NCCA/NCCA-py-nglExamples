// GUIDemos/QMLOverlayApp/main.qml
import QtQuick
import QtQuick.Window
import ncca.ngl.qml 1.0

Item {
    id: overlayRoot
    property int nextZ: 100

    TransformWidget {
        id: transformWidget
        // name left blank: DraggablePanel's own title bar already reads
        // "Transform" - avoid showing the same label twice.
        name: ""
        // NOTE: use the "matrix" Property, not the get_matrix() Slot call.
        // Calling a @Slot(result=Mat4)-decorated method from QML JS and
        // re-passing its return value into another Python @Slot(Mat4)
        // argument fails PySide6's copy-conversion ("Cannot copy-convert
        // ... (Mat4) to C++"), silently delivering None. Property access
        // marshals correctly.
        onValueChanged: pyNGLScene.set_model_matrix(model.matrix)
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
        name: ""
        onColourChanged: pyNGLScene.set_colour(model.r, model.g, model.b)
    }
    DraggablePanel {
        panelId: "colour"
        title: "Colour"
        x: 30
        y: 335
        content: [rgbWidget]
    }

    LookAtWidget {
        id: lookAtWidget
        name: ""
        onValueChanged: pyNGLScene.set_view_matrix(model.matrix)
    }
    DraggablePanel {
        panelId: "camera"
        title: "Camera"
        x: 30
        y: 445
        content: [lookAtWidget]
    }

    Component.onCompleted: {
        pyNGLScene.set_model_matrix(transformWidget.model.matrix)
        pyNGLScene.set_view_matrix(lookAtWidget.model.matrix)
        pyNGLScene.set_colour(rgbWidget.model.r, rgbWidget.model.g, rgbWidget.model.b)
    }
}
