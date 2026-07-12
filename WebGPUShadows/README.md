# PCF Shadows Using WebGPU

This is a project that demonstrates how to implement PCF (Percentage Closer Filtering) shadows using WebGPU. This has two pipeline one for the Shadow pass which will record a depth texture and another for the main pass which will use the depth texture to render the scene with shadows.



## References

- L. Williams, "Casting Curved Shadows on Curved Surfaces", SIGGRAPH 1978 — [PDF](https://cseweb.ucsd.edu/~ravir/6160-fall04/papers/p270-williams.pdf) — the original shadow-mapping paper.
- W. T. Reeves, D. H. Salesin & R. L. Cook, "Rendering Antialiased Shadows with Depth Maps", SIGGRAPH 1987 — [ACM](https://dl.acm.org/doi/10.1145/37401.37435) — the paper that introduced percentage closer filtering.
- [WebGPU Fundamentals — Shadows](https://webgpufundamentals.org/webgpu/lessons/webgpu-shadows.html) — shadow maps with the WebGPU depth-comparison sampler.
- [LearnOpenGL — Shadow Mapping](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping) — bias, peter-panning and PCF explained.
