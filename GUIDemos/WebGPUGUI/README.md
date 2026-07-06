# WebGPUGUI

Demonstrates embedding a WebGPU viewport inside a Qt Designer GUI. A teapot is
rendered with a PBR shader (`PBRShader.wgsl`) inside a `WebGPUWidget`, and the
surrounding controls from `MainWindow.ui` drive the scene:

- Position / Scale / Rotation spin boxes transform the model
- Colour dialogs set the base and light colours
- Sliders control the PBR metallic, roughness and ambient occlusion parameters
- Spin boxes position the light and scale its intensity

This is the WebGPU equivalent of the OpenGL GUI demos in `GUIDemos`.

## Files

- `main.py` - main window, loads the `.ui` file and connects signals
- `WebGPUScene.py` - the WebGPU rendering widget (camera, pipeline, uniforms)
- `WebGPUWidget.py` - base Qt widget hosting the WebGPU surface
- `PBRShader.wgsl` - PBR vertex / fragment shader
- `MainWindow.ui` - Qt Designer interface

## Controls

- Left-drag in the viewport : rotate camera, Right-drag : pan, Wheel : zoom
- `Esc` : quit
