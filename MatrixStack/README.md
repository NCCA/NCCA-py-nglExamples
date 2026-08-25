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

## WebGPU version

`main_webgpu.py` shares `matrix_stack.py` unchanged with the OpenGL version
— the push/pop logic is pure CPU-side matrix maths, independent of the
rendering backend. It builds all 130 matrix and lighting records first, uploads
them in one uniform buffer, then draws the three trolls and 126 spheres as two
instanced batches. The grid is a flat quad drawn from the final record.

### Why this differs from OpenGL

The OpenGL version sets `MVP`, `normalMatrix` and `Colour` immediately before
each call to `Primitives.draw()`. OpenGL keeps those state changes and draw
commands in order, so each draw uses the uniform values which were current at
that point in the command stream (even though the GPU may do the actual work
later).

WebGPU records a render pass into a command buffer before submitting it.
`queue.write_buffer()` is a queue operation, not a command recorded between
draws in that render pass. If we keep rewriting one bound uniform buffer whilst
encoding the draws, the earlier draws do not get private copies of the earlier
values. By the time the render command buffer is submitted, they all refer to
the same buffer containing the final update. This was why the original WebGPU
version rendered the transforms incorrectly.

We could make a separate uniform buffer for every object, or use dynamic buffer
offsets and change the bind group for each draw. For this demo it is clearer to
run the matrix stack first and put every result into one array. The shader uses
`instance_index` to select the right record, whilst `first_instance` selects the
troll, sphere or grid range. This also lets all of the trolls share one draw and
all of the spheres share another, rather than issuing 129 separate geometry
draws.
