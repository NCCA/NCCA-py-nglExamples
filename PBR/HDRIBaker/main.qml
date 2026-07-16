// PBR/HDRIBaker/main.qml
// Floating control panels for hdri_demo.py's single teapot. Built on the
// same transparent-overlay pattern as GUIDemos/QMLWebGPUOverlay: draggable
// panels drive the WebGPU scene's @Slots (metallic/roughness/ao/albedo, the
// IBL toggle and the skybox cube selector), and clicks outside a panel fall
// through to the camera.
import QtCore
import QtQuick
import QtQuick.Controls
import ncca.ngl.qml 1.0

Item {
    id: overlayRoot
    property int nextZ: 100

    // A DraggablePanel pre-wired to a legible dark theme. Fusion's default
    // palette is tuned for light backgrounds, so its near-black text is
    // illegible on DraggablePanel's dark body - set the palette roles once
    // here rather than on every panel.
    component ThemedPanel: DraggablePanel {
        palette.windowText: "#f2f2f2"
        palette.buttonText: "#f2f2f2"
        palette.text: "#f2f2f2"
        palette.button: "#3c3c3c"
        palette.base: "#232323"
        palette.highlight: "#2a82da"
        palette.highlightedText: "#ffffff"
        panelColor: "#2b2b2b"
        panelBorder: "#555555"
    }

    // A labelled 0..1 slider that reports its live value on drag. `value` is
    // the initial position; `sliderValue` reads the current one.
    component LabeledSlider: Column {
        id: ls
        property string label: ""
        property real from: 0.0
        property real to: 1.0
        property real value: 0.0
        property alias sliderValue: slider.value
        signal moved(real value)
        spacing: 2
        Label { text: ls.label + ": " + slider.value.toFixed(2) }
        Slider {
            id: slider
            width: 200
            from: ls.from
            to: ls.to
            value: ls.value
            onMoved: ls.moved(value)
        }
    }

    // -------------------------------------------------------------- material
    ThemedPanel {
        id: materialPanel
        panelId: "material"
        x: 20
        y: 20
        Column {
            spacing: 8
            Label { text: "Material"; font.bold: true }
            LabeledSlider {
                id: metallicSlider
                label: "Metallic"
                value: 1.0
                onMoved: scene.set_metallic(value)
            }
            LabeledSlider {
                id: roughnessSlider
                label: "Roughness"
                from: 0.05
                value: 0.25
                onMoved: scene.set_roughness(value)
            }
            LabeledSlider {
                id: aoSlider
                label: "AO"
                value: 1.0
                onMoved: scene.set_ao(value)
            }
        }
    }

    // ---------------------------------------------------------------- albedo
    RGBColourWidget {
        id: albedoWidget
        name: "Albedo"
        onColourChanged: scene.set_albedo(model.r, model.g, model.b)
    }
    ThemedPanel {
        id: albedoPanel
        panelId: "albedo"
        x: 20
        y: 230
        content: [albedoWidget]
    }

    // ----------------------------------------------------------- lighting
    ThemedPanel {
        id: lightingPanel
        panelId: "lighting"
        x: 20
        y: 355
        // The combo's dropdown reparents into the window overlay while open,
        // outside this panel's item tree, so suspend dragging while it shows
        // (matches GUIDemos/QMLWebGPUOverlay).
        dragEnabled: !envCombo.popup.visible
        Column {
            spacing: 8
            Label { text: "Lighting"; font.bold: true }
            CheckBox {
                id: iblCheck
                text: "IBL ambient"
                checked: true
                onToggled: scene.set_use_ibl(checked)
            }
            Row {
                spacing: 8
                Label {
                    text: "Skybox"
                    anchors.verticalCenter: envCombo.verticalCenter
                }
                ComboBox {
                    id: envCombo
                    width: 160
                    // Order matches DEBUG_VIEWS in hdri_demo.py.
                    model: ["Environment", "Irradiance", "Prefilter mip 2"]
                    currentIndex: 0
                    onActivated: scene.set_debug_view(currentIndex)
                    Connections {
                        target: envCombo.popup
                        function onVisibleChanged() {
                            panelRegistry.set_popup_open(envCombo.popup.visible)
                        }
                    }
                }
            }
        }
    }

    // Persist each panel's position between runs (QtCore Settings, backed by
    // the QSettings store keyed on the org/app name set in main()).
    Settings {
        category: "layout"
        property alias materialX: materialPanel.x
        property alias materialY: materialPanel.y
        property alias albedoX: albedoPanel.x
        property alias albedoY: albedoPanel.y
        property alias lightingX: lightingPanel.x
        property alias lightingY: lightingPanel.y
    }

    // Push the panels' initial values into the scene so the teapot's starting
    // material matches what the controls show. This is deferred by a
    // zero-interval Timer rather than run in Component.onCompleted: the slots
    // call the scene's update(), and doing that synchronously while the
    // QQuickWidget is still loading this QML (inside setSource) reenters the
    // WebGPU paint before the overlay's scene graph is ready and deadlocks.
    // Firing on the next event-loop tick lets setSource finish first.
    Timer {
        interval: 0
        running: true
        repeat: false
        onTriggered: {
            scene.set_metallic(metallicSlider.sliderValue)
            scene.set_roughness(roughnessSlider.sliderValue)
            scene.set_ao(aoSlider.sliderValue)
            scene.set_albedo(albedoWidget.model.r, albedoWidget.model.g, albedoWidget.model.b)
            scene.set_use_ibl(iblCheck.checked)
            scene.set_debug_view(envCombo.currentIndex)
        }
    }
}
