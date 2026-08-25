# DefferedLighting

A WebGPU deferred shading demo. The first pass renders a teapot and floor into a G-buffer of position, normal and albedo textures (`GBuffer.wgsl`), then a lighting pass (`lighting.wgsl`) reads the G-buffer and shades the scene with two lights.

Note this demo is currently broken (see [TODO.md](../TODO.md)) so don't expect too much from it yet.

```bash
uv run DefferedLighting/SimpleWebGPU.py
```

![](WebGPUNGL.png)

## References

- T. Saito & T. Takahashi, "Comprehensible Rendering of 3-D Shapes", SIGGRAPH 1990 — [SIGGRAPH history](https://history.siggraph.org/learning/comprehensible-rendering-of-3d-shapes-by-saito-and-takahashi/) — the paper that introduced G-buffers.
- S. Hargreaves & M. Harris, "Deferred Shading" (6800 Leagues Under the Sea), NVIDIA/Climax 2004 — [PDF](https://shawnhargreaves.com/DeferredShading.pdf) — the practical real-time formulation.
- [LearnOpenGL — Deferred Shading](https://learnopengl.com/Advanced-Lighting/Deferred-Shading) — G-buffer layout and the lighting pass, as implemented in `GBuffer.wgsl` / `lighting.wgsl`.
- [WebGPU Samples — Deferred Rendering](https://webgpu.github.io/webgpu-samples/?sample=deferredRendering) — an equivalent WebGPU implementation.
