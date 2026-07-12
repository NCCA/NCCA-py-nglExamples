# SimpleWebGPU Example

This example demonstrates how to use PyNGL with WebGPU to create a simple 3D Scene. In particular it demonstrates the use of The PyNGL primitives class and how we need to generate a new pipeline per shader type.

## References

- T. Saito & T. Takahashi, "Comprehensible Rendering of 3-D Shapes", SIGGRAPH 1990 — [SIGGRAPH history](https://history.siggraph.org/learning/comprehensible-rendering-of-3d-shapes-by-saito-and-takahashi/) — the paper that introduced G-buffers.
- S. Hargreaves & M. Harris, "Deferred Shading" (6800 Leagues Under the Sea), NVIDIA/Climax 2004 — [PDF](https://shawnhargreaves.com/DeferredShading.pdf) — the practical real-time formulation.
- [LearnOpenGL — Deferred Shading](https://learnopengl.com/Advanced-Lighting/Deferred-Shading) — G-buffer layout and the lighting pass, as implemented in `GBuffer.wgsl` / `lighting.wgsl`.
- [WebGPU Samples — Deferred Rendering](https://webgpu.github.io/webgpu-samples/?sample=deferredRendering) — an equivalent WebGPU implementation.
