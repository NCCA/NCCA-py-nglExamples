# ViewToWorldTransform

Shift-click anywhere in the viewport to unproject the screen position into a
world-space point and drop a cube there — the inverse of the usual
model/view/projection pipeline. The unprojection maths
(`view_to_world.unproject_point`) lives in its own pure-numpy module, no
GL/Qt dependency, with pytest coverage in `tests/test_view_to_world.py`.

## Controls
- `Shift`+LMB : place a cube at the unprojected point
- `Space` : clear placed cubes
- Left-drag : orbit, Right-drag : pan, Wheel : zoom, `Esc` : quit

![ViewToWorldTransform](ViewToWorldTransform.png)

## WebGPU version

`main_webgpu.py` shares `view_to_world.py`'s `unproject_point` unchanged
with the OpenGL version — screen-to-world unprojection is pure numpy maths,
independent of the rendering backend.
