# WebGPU Render To Texture

The WebGPU version of the [SimpleFBO](../SimpleFBO) demo. It renders a spinning teapot into an offscreen texture in a first pass, then uses that texture to shade a ground plane and a sphere in a second pass. It is the same render-to-texture idea as an OpenGL framebuffer object, just expressed with WebGPU render passes.

![](WebGPURenderToTexture.png)

## How it works

There are two pipelines and two passes per frame.

`TeapotPipeline` owns its own 1024x1024 offscreen colour texture (multisampled, then resolved to a single-sample texture) and a matching depth buffer. It draws the PBR teapot into that texture with a square projection so nothing is stretched. The resolved texture is handed out as `texture_view`.

`ScenePipeline` takes that `texture_view` and draws a plane and a sphere with it, sampling the teapot render as an ordinary 2D texture. This pass goes into the widget's own colour buffer at the full window size, so it fills the window and tracks the display's pixel ratio rather than sitting in a corner.

Left mouse rotates the scene camera, right mouse pans, the wheel zooms, and the teapot in the texture spins on its own.

## Running

```bash
uv run FBODemos/WebGPURenderToTexture/main.py
```

## References

- [LearnOpenGL — Framebuffers](https://learnopengl.com/Advanced-OpenGL/Framebuffers) — the same idea in its OpenGL form.
- [WebGPU Fundamentals — Textures](https://webgpufundamentals.org/webgpu/lessons/webgpu-textures.html) — sampling a texture in a render pass.
- [WebGPU Specification — Render Passes](https://www.w3.org/TR/webgpu/#render-passes) — colour attachments, resolve targets and render-pass descriptors.
