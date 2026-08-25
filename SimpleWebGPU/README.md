# SimpleWebGPU Example

![](WebGPUNGL.png)

This example demonstrates how to use PyNGL with WebGPU to create a simple 3D Scene. In particular it demonstrates the use of The PyNGL primitives class and how we need to generate a new pipeline per shader type.

## References

- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html) — devices, pipelines and render passes.
- [WebGPU Fundamentals — Uniforms](https://webgpufundamentals.org/webgpu/lessons/webgpu-uniforms.html) — the per-frame MVP/normal-matrix uniform buffer pattern.
- [WebGPU Specification](https://www.w3.org/TR/webgpu/) and [WGSL Specification](https://www.w3.org/TR/WGSL/).
- [wgpu-py documentation](https://wgpu-py.readthedocs.io/) — the Python WebGPU binding.
