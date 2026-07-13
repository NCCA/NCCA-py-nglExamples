# 2D Drawing OpenGL

![](2D.png)

There are two demos in this folder both with the same initial structure.

- [2DDrawing.py](./2DDrawing.py): This demo shows how to draw 2D points using OpenGL and orthographic projection.
- [PanZoom.py](./PanZoom.py): Expands on the previous demo by adding pan and zoom functionality.

## Discussion

Initialization (`__init__` and `initializeGL`) uses a basic PySide6 `QOpenGLWindow` to serve as the canvas for rendering.

`gen_points(1000)` is called to create 1000 particles. Each particle is given a random initial position, velocity (direction and speed), and colour.

This data is stored in `numpy` arrays in a Structure of Arrays (SOA) data layout, and a Vertex Array Object (VAO) is created and the colours are set once, and the positions are updated every frame.

For the Animation Loop (`timerEvent`) a timer is started to call the `timerEvent` method at a regular interval (approximately 60 times per second).

In each step, the particle positions are updated by adding their velocity. The velocity is a combination of the particle's intrinsic direction and the global `wind` vector.

For collision detection, bounds are checked and if any particles have moved beyond the simulation boundaries (`-SIM_WIDTH/2` to `SIM_WIDTH/2` on both X and Y axes).

If a particle hits a boundary, its direction is inverted on the corresponding axis, causing it to "bounce" off the wall. Its position is also clamped to the boundary to prevent it from getting stuck.

Finally, `self.update()` is called to schedule a redraw of the window.

Rendering is done in `paintGL`.

- The projection matrix is updated based on the current `zoom` level, which is controlled by the mouse wheel.
- The particle position data in the VAO is updated with the new positions calculated in `timerEvent`.
- The VAO is drawn, rendering all the particles to the screen as circles (as defined in the shader).
- A text overlay is rendered using `ngl.Text` to display the current wind vector and instructions for the user.

### User Controls

- **Arrow Keys:** Modify the global `wind` vector.
  - **Up/Down:** Increase/decrease the wind's Y component.
  - **Left/Right:** Decrease/increase the wind's X component.
- **Spacebar:** Resets the wind vector to `[0, 0]` and the zoom level to `1.0`.
- **Mouse Wheel:** Zooms the view in and out.
- **Escape Key:** Closes the application.

The Zoom version adds more control by storing the current mouse position and allowing the ortho window to be changed.

## References

- [OpenGL Projection Matrix](https://www.songho.ca/opengl/gl_projectionmatrix.html) — Song Ho Ahn's derivation of the orthographic (and perspective) projection matrix used here.
- [LearnOpenGL — Coordinate Systems](https://learnopengl.com/Getting-started/Coordinate-Systems) — how NDC, projection and the viewport fit together.
- [OpenGL Wiki — Buffer Object Streaming](https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming) — strategies for updating VBO contents every frame, as done for the particle positions.
