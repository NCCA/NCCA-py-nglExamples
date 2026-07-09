// GUIDemos/QMLOverlayApp/main.qml
import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import ncca.ngl.qml 1.0

Item {
    id: overlayRoot
    property int nextZ: 100

    // ncca.ngl.qml's Label/ToolButton/SpinBox/ComboBox pick up their colours
    // from the inherited Qt Quick Controls palette, not hardcoded values. The
    // Fusion style's default palette is tuned for light backgrounds, so its
    // near-black default text is barely legible on DraggablePanel's dark body.
    // Rather than forcing one fixed colour, offer a few themes via the "Style"
    // panel below. Each preset is a full theme: text/button colours (palette
    // roles) PLUS the panel body colour/border, so a theme can restyle
    // everything at once. The palette is a live QML value, so switching it
    // updates every panel immediately without recreating anything.
    //
    // The last three reproduce Dear ImGui's well-known looks - its dark theme
    // (blue #4296fa accent), the classic blue-purple theme, and the light
    // theme - by mapping ImGui's style colours onto the nearest palette roles:
    //   windowText <- Text, button/buttonText <- Button, highlight <- Header,
    //   base <- FrameBg, panelColor <- WindowBg, panelBorder <- Border.
    readonly property var stylePresets: [
        {
            name: "Light Text (default)",
            windowText: "#f2f2f2", buttonText: "#f2f2f2", highlightedText: "#ffffff",
            panelBg: "#2b2b2b", panelBorder: "#555555",
            button: "#3c3c3c", base: "#232323", highlight: "#2a82da"
        },
        {
            name: "Classic (dark text)",
            windowText: "#202020", buttonText: "#202020", highlightedText: "#ffffff",
            panelBg: "#d9d9d9", panelBorder: "#a0a0a0",
            button: "#c8c8c8", base: "#f0f0f0", highlight: "#2a82da"
        },
        {
            name: "ImGui Dark",
            windowText: "#e6e6e6", buttonText: "#ffffff", highlightedText: "#ffffff",
            panelBg: "#0f0f0f", panelBorder: "#6e6e80",
            button: "#4296fa", base: "#1a2b44", highlight: "#4296fa"
        },
        {
            name: "ImGui Classic (blue)",
            windowText: "#e6e6e6", buttonText: "#e6e6e6", highlightedText: "#ffffff",
            panelBg: "#10101c", panelBorder: "#4b4b78",
            button: "#59669c", base: "#1e1e32", highlight: "#758acc"
        },
        {
            name: "ImGui Light",
            windowText: "#1a1a1a", buttonText: "#ffffff", highlightedText: "#ffffff",
            panelBg: "#f0f0f0", panelBorder: "#c8c8c8",
            button: "#4296fa", base: "#ffffff", highlight: "#4296fa"
        }
    ]
    // Default to the Dear ImGui Classic theme. Found by name rather than a
    // hardcoded index so reordering stylePresets can't silently pick another
    // theme. On second and later runs the Settings block below restores the
    // last-used value instead.
    property int styleIndex: stylePresets.findIndex(function (p) {
        return p.name === "ImGui Classic (blue)"
    })
    readonly property var currentStyle: stylePresets[styleIndex]

    // A DraggablePanel pre-wired to the active theme, so the palette-role and
    // panel-chrome bindings live in one place instead of being repeated on
    // every panel. Instances still set panelId/x/y/content (and the style
    // panel additionally overrides dragEnabled).
    component ThemedPanel: DraggablePanel {
        palette.windowText: overlayRoot.currentStyle.windowText
        palette.buttonText: overlayRoot.currentStyle.buttonText
        palette.text: overlayRoot.currentStyle.windowText
        palette.button: overlayRoot.currentStyle.button
        palette.base: overlayRoot.currentStyle.base
        palette.highlight: overlayRoot.currentStyle.highlight
        palette.highlightedText: overlayRoot.currentStyle.highlightedText
        panelColor: overlayRoot.currentStyle.panelBg
        panelBorder: overlayRoot.currentStyle.panelBorder
    }

    TransformWidget {
        id: transformWidget
        // name shows in the widget's own collapsible header (ToolButton) -
        // matches src/ncca/ngl/qml/main.qml in the PyNGL repo. DraggablePanel
        // no longer draws a separate title, so this is the only label.
        name: "Transform Widget"
        // NOTE: use the "matrix" Property, not the get_matrix() Slot call.
        // Calling a @Slot(result=Mat4)-decorated method from QML JS and
        // re-passing its return value into another Python @Slot(Mat4)
        // argument fails PySide6's copy-conversion ("Cannot copy-convert
        // ... (Mat4) to C++"), silently delivering None. Property access
        // marshals correctly.
        onValueChanged: pyNGLScene.set_model_matrix(model.matrix)
    }
    ThemedPanel {
        id: transformPanel
        panelId: "transform"
        x: 30
        y: 30
        content: [transformWidget]
    }

    RGBColourWidget {
        id: rgbWidget
        name: "RGB Colour Widget"
        onColourChanged: pyNGLScene.set_colour(model.r, model.g, model.b)
    }
    ThemedPanel {
        id: colourPanel
        panelId: "colour"
        x: 30
        y: 335
        content: [rgbWidget]
    }

    LookAtWidget {
        id: lookAtWidget
        name: "Look At"
        onValueChanged: pyNGLScene.set_view_matrix(model.matrix)
    }
    PerspectiveWidget {
        id: perspectiveWidget
        // Aspect is driven by the window/viewport size on resize (see
        // PyNGLScene.resizeGL), not by this widget, so a manual edit here
        // would just be overwritten - fov/near/far are the only params read.
        name: "Perspective"
        onValueChanged: pyNGLScene.set_perspective_params(model.fov, model.near, model.far)
    }
    ThemedPanel {
        id: cameraPanel
        panelId: "camera"
        x: 30
        y: 445
        content: [
            Column {
                spacing: 8
                children: [lookAtWidget, perspectiveWidget]
            }
        ]
    }

    Row {
        id: styleRow
        spacing: 8
        Label {
            text: "Style"
            anchors.verticalCenter: styleCombo.verticalCenter
        }
        ComboBox {
            id: styleCombo
            model: overlayRoot.stylePresets.map(function (preset) { return preset.name })
            currentIndex: overlayRoot.styleIndex
            onActivated: overlayRoot.styleIndex = currentIndex
            // The dropdown list opens outside this panel's registered rect, so
            // the overlay widget's hit_test would forward clicks on it to the
            // GL scene instead of QML. Tell the registry a popup is open so it
            // routes every click to the overlay while the list is showing.
            Connections {
                target: styleCombo.popup
                function onVisibleChanged() {
                    panelRegistry.set_popup_open(styleCombo.popup.visible)
                }
            }
        }
    }
    ThemedPanel {
        id: stylePanel
        panelId: "style"
        x: 30
        y: 555
        // The combo box's dropdown list is reparented into the window's
        // Overlay layer while open, outside this panel's own item tree, so
        // this panel's DragHandler can't reliably tell a click there apart
        // from a click on its own background. Suspend dragging for as long
        // as the popup is open so it can never intercept that click.
        dragEnabled: !styleCombo.popup.visible
        content: [styleRow]
    }

    // Persist the layout (each panel's position and the chosen theme) with the
    // QML-native Settings type from QtCore. It is backed by the same QSettings
    // store as the C++/Python API - the storage path comes from the
    // organization/application name set in main.py - but binds straight to QML
    // properties: property aliases are two-way, so a dragged panel or a theme
    // change is written back on exit, and the stored value is restored into the
    // property at startup. No save-on-close handler needed. First run has no
    // stored values, so the declared defaults (positions above, ImGui Classic
    // theme) are used and then saved.
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

    Component.onCompleted: {
        pyNGLScene.set_model_matrix(transformWidget.model.matrix)
        pyNGLScene.set_view_matrix(lookAtWidget.model.matrix)
        pyNGLScene.set_colour(rgbWidget.model.r, rgbWidget.model.g, rgbWidget.model.b)
        pyNGLScene.set_perspective_params(
            perspectiveWidget.model.fov, perspectiveWidget.model.near, perspectiveWidget.model.far
        )
    }
}
