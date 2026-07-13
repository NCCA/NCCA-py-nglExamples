# Camera

![](Camera.png)

Demonstrates a hand-rolled UVN camera (`uvn_camera.py`) with 4 selectable views (front,
top, side, perspective) looking at a PBR-lit teapot, spinning cube, and football. An
on-screen overlay (via `ncca.ngl.Text`) shows the controls, active camera/FOV, its view
matrix, and light on/off status, matching the original NGL9Demos/Camera's help text.

## Controls

- `1`-`4` : switch active camera
- Arrow keys : move the active camera's eye
- `r` / `y` / `p` : roll / yaw / pitch the active camera
- `z` / `x` / `c` : toggle the 3 scene lights
- `+` / `-` : adjust field of view
- Left-drag : orbit scene, Right-drag : pan, Wheel : zoom, `space` : reset

## References

- [LearnOpenGL — Camera](https://learnopengl.com/Getting-started/Camera) — view matrices, eye/look/up and camera movement.
- [OpenGL Camera (songho.ca)](https://www.songho.ca/opengl/gl_camera.html) — derivation of the lookAt matrix from the U,V,N basis vectors, as in `uvn_camera.py`.
- [OGLdev — Camera Space](https://ogldev.org/www/tutorial13/tutorial13.html) — the classic UVN camera formulation.
- [OpenGL Transformation (songho.ca)](https://www.songho.ca/opengl/gl_transform.html) — the full model/view/projection pipeline shown in the on-screen matrix overlay.
