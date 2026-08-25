# Instancing

![](Instancing.png)

A field of cubes drawn two ways with the identical geometry and identical per-instance data: one GPU-instanced draw call, or a Python `for` loop issuing one draw call per cube. Toggle between them and watch the HUD frame-time number, because that number is the entire demo.

- `main.py` — OpenGL, `glDrawArraysInstanced` vs a loop of `glDrawArrays`
- `InstancingWebGPU.py` — WebGPU, `render_pass.draw(36, n)` vs a loop of `render_pass.draw(36, 1, 0, i)`

Both place their cubes with the same numpy-only layout maths, in `instance_layout.py` (unit tested in `tests/`), so the two backends render an identical scene.

## Controls

| Key               | Action                                                          |
| :---------------- | :--------------------------------------------------------------- |
| `I`               | toggle instanced draw / naive draw-call-per-cube loop            |
| `+` / `-`         | double / halve the instance count N (clamped 1 .. 65536)         |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits                 |

Default N is 4096. Try `I` at that count before touching anything else — that is the whole point of the demo.

## Things to know

1. **One draw call, N objects.** `glDrawArraysInstanced`/`draw(vertex_count, instance_count)` tells the GPU to run the same vertex data N times, varying only the attributes that come from a *second* vertex buffer with a per-instance step rate. All the placement, colour and scale data is already sitting in GPU memory; the CPU submits one command regardless of N.
2. **The naive loop pays a per-draw-call tax that has nothing to do with the GPU.** Each Python-level `glDrawArrays`/`draw()` call crosses into PyOpenGL/wgpu-py, then into the driver, then (on the GL side) into a fresh `glUniform4f` upload for that cube's offset/scale/colour. None of that scales — it is pure per-call overhead, and it is exactly what disappears when the whole field becomes one instanced call. Time the *whole frame* with `time.perf_counter`, not just the draw call, or you measure the wrong thing and miss the lesson entirely.
3. **The per-instance record is 8 floats (offset.xyz, scale, colour.rgba), not a 4x4 matrix.** A mat4-per-instance attribute would need four consecutive attribute locations and four `glVertexAttribDivisor` calls (each `vec4` row of the matrix is its own attribute) for a transform this demo only ever uses as translate + uniform scale. Eight floats and two locations do the job; keep the mat4 version for when you actually need per-instance rotation or non-uniform scale.
4. **GL: bind the VAO before touching the divisor.** `glVertexAttribDivisor` state belongs to whichever VAO is currently bound, so setting it right after `glBindVertexArray` (as `_build_cube_vao` does) keeps it from leaking into any VAO built later in the same context.
5. **WebGPU has no `glVertexAttribDivisor` — it is declared on the buffer layout instead.** The second vertex buffer is created with `"step_mode": "instance"`; everything else about reading it is identical to a normal vertex buffer.
6. **The WebGPU naive path uses `first_instance`, not a fresh bind group per cube.** Building n small uniform-buffer bind groups every frame would itself dominate the timing and muddy the comparison. Instead the naive loop calls `draw(36, 1, 0, i)` for i in `range(n)` — same pipeline, same instance buffer, `first_instance=i` selects record i. I checked wgpu-py's native backend (`wgpu.backends.wgpu_native._api.RenderPassEncoder.draw`) and it passes `first_instance` straight through to `wgpuRenderPassEncoderDraw`, which the WebGPU spec defines as also offsetting instance-step-mode attribute fetches — so this is an honest one-draw-call-per-object comparison, not a shortcut. If you're running a backend where that turns out not to hold, the documented fallback is a dynamic uniform-buffer offset per cube instead.

## Tests

```bash
uv run pytest Instancing/tests
```

## References

- [OpenGL Wiki — Vertex Rendering: Instancing](https://www.khronos.org/opengl/wiki/Vertex_Rendering#Instancing) — `glDrawArraysInstanced`, `gl_InstanceID` and `glVertexAttribDivisor`.
- [LearnOpenGL — Instancing](https://learnopengl.com/Advanced-OpenGL/Instancing) — the same technique with a worked asteroid-field example.
- [WebGPU spec — GPURenderPassEncoder.draw](https://www.w3.org/TR/webgpu/#dom-gpurendercommandsmixin-draw) — `firstInstance` semantics and instance-step-mode vertex buffers.
- [WebGPU spec — GPUVertexStepMode](https://www.w3.org/TR/webgpu/#enumdef-gpuvertexstepmode) — the `"instance"` step mode used for the second vertex buffer.
