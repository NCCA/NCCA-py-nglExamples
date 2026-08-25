# PCF Shadows Using WebGPU

![](WebGPUShadows.png)

A two pass shadow map demo using WebGPU. The first pipeline is a depth-only shadow pass (no fragment stage, no colour attachments) that renders the scene from the light's point of view into a 2048^2 `depth24plus` texture (`ShadowShader.wgsl`). The main pass (`DiffuseShader.wgsl`) re-projects each fragment through the light's view-projection and tests it against that texture using a comparison sampler — `textureSampleCompare` in a 3x3 loop, giving percentage closer filtered (PCF) soft shadow edges.

This is the WebGPU mirror of [`../ShadowMapping`](../ShadowMapping), which renders the same scene in OpenGL — see the table in that README for an API-for-API diff of the two backends.

```bash
uv run WebGPUShadows/PCFShadows.py
```

## Controls

- Left-drag : rotate the camera, arrow keys : move (first person)
- `1` : toggle the light
- `Esc` : quit

## References

- L. Williams, "Casting Curved Shadows on Curved Surfaces", SIGGRAPH 1978 — [PDF](https://cseweb.ucsd.edu/~ravir/6160-fall04/papers/p270-williams.pdf) — the original shadow-mapping paper.
- W. T. Reeves, D. H. Salesin & R. L. Cook, "Rendering Antialiased Shadows with Depth Maps", SIGGRAPH 1987 — [ACM](https://dl.acm.org/doi/10.1145/37401.37435) — the paper that introduced percentage closer filtering.
- [WebGPU Fundamentals — Shadows](https://webgpufundamentals.org/webgpu/lessons/webgpu-shadows.html) — shadow maps with the WebGPU depth-comparison sampler.
- [LearnOpenGL — Shadow Mapping](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping) — bias, peter-panning and PCF explained.
