# MatrixStack

An OpenGL-style push/pop matrix stack (`matrix_stack.py`), built by hand
instead of using `ncca.ngl.Transform`. Three trolls sit on a stack of
pushed/popped transforms, a ring of small spheres orbits with a wave in Y
whose frequency you can change live, and a reference grid sits underneath —
every draw call is wrapped in its own `push_matrix()`/`pop_matrix()` pair so
you can see exactly which state each object inherits from its parent.

`matrix_stack.py` has no GL/Qt dependency; the WebGPU version of this demo
(`main_webgpu.py`) imports the identical module, since the stack is pure
CPU-side matrix bookkeeping regardless of the rendering backend.

## Controls
- `I` / `O` : increase / decrease the sphere ring's wave frequency
- `W` / `S` : wireframe / solid
- Left-drag : orbit, Right-drag : pan, Wheel : zoom, `Space` : reset, `Esc` : quit

![MatrixStack](MatrixStack.png)
