# HDRI Image-Based Lighting — design

## What this is

A new demo folder `PBR/HDRI` showing full image-based lighting from a real HDRI
environment map, in both OpenGL and WebGPU. It loads the equirectangular
`historic_cloister_passage_1k.exr`, bakes the split-sum environment on the GPU
at startup, and lights a 7×7 grid of teapots that sweeps metallic against
roughness, with the HDRI itself drawn behind as a skybox.

## Why it exists (and how it differs from `PBR/IBL`)

There is already a `PBR/IBL` demo, but it deliberately stops short: it lights
from a *procedural* sky, it is OpenGL only, and it cuts the prefiltered
specular mip chain, so reflections stay sharp at every roughness. This demo is
the complete version of that story — a real HDRI, the full split-sum bake
including the prefilter chain, and the same technique ported to WebGPU so the
two backends sit side by side in one folder. Roughness actually blurs the
reflected environment here, which is the whole point of the prefilter stage.

## The GPU bake

Both scripts run the same one-time bake at startup, in the textbook order.
Only stage 0 is CPU work; everything after it renders on the GPU, so it is
written once per backend rather than shared.

| Stage | Output | How |
| --- | --- | --- |
| 0. Load EXR | equirect float32 RGB, uploaded as a 2D texture | OpenEXR/Imath → numpy (`exr_loader.py`) |
| 1. Equirect → cubemap | environment cubemap, 512²/face | render a unit cube six times, sampling the equirect map by direction |
| 2. Irradiance | 32²/face cubemap | cosine-weighted hemisphere convolution of the env cubemap |
| 3. Prefilter | 128²/face cubemap, 5 mips | GGX importance sampling, roughness = mip level |
| 4. BRDF LUT | 512² RG16F 2D texture | one fullscreen-quad pass, the split-sum scale/bias integral |

Stages 1–4 are standard render-to-cubemap / render-to-texture passes. On
OpenGL that is an FBO with cubemap-face colour attachments; on WebGPU it is a
render pass per face (and per mip for the prefilter chain) targeting individual
array layers of a cube texture.

## Runtime render

Once the bake is done, each frame draws:

- **Skybox** — the environment cubemap sampled along the view ray, filling the
  background.
- **Teapot grid** — the Cook-Torrance direct term shared with `PBR/SimplePBR`,
  plus the IBL ambient term:

  ```
  ambient = kD · irradiance(N) · albedo
          + prefiltered(R, roughness) · (F · A + B)
  ```

  Rows sweep metallic 0→1, columns sweep roughness 0→1, the same layout as the
  other PBR demos.

## Files

```
PBR/HDRI/
  images/historic_cloister_passage_1k.exr   # already present
  exr_loader.py        # EXR → numpy; the one headlessly-tested module
  main.py              # OpenGL (QOpenGLWindow), FBO render-to-cubemap bake
  HDRIWebGPU.py        # WebGPU (WebGPUWidget), array-layer render-to-cube bake
  shaders/             # GLSL: equirect2cube, irradiance, prefilter, brdf, skybox, pbr
  *.wgsl               # the WGSL equivalents
  tests/test_exr_loader.py
  README.md
  HDRI.png             # OpenGL screenshot
  HDRIWebGPU.png       # WebGPU screenshot
```

`pyproject.toml` gains the `OpenEXR` (and its `Imath`) dependency for stage 0.

## Controls (both backends)

| Key | Effect |
| --- | --- |
| `I` | toggle IBL ambient on/off |
| `E` | cycle debug view: off / irradiance as skybox / prefilter mip as skybox / BRDF LUT overlay |
| LMB | rotate |
| RMB | pan |
| Wheel | zoom |
| Space | reset camera |
| Esc | quit |

## Testing and known limits

`exr_loader.py` is pure Python and numpy with no GL or Qt, so it is the one
part that gets a headless unit test — channel order, dtype, shape, and that the
HDR range survives the load (values above 1.0 are preserved). Everything
downstream is on the GPU, and this sandbox has no working offscreen GL/WebGPU
context, so the render path is verified by running the demos on a real machine,
the same limitation already noted across the repo. Both scripts take a
`--smoketest` flag that bakes, draws one frame, prints `SMOKETEST OK` and quits,
for use where a real context exists.

The fiddliest piece is the WebGPU prefilter bake — rendering to each face and
each mip level of a cube texture — and it is written from scratch rather than
shared with the OpenGL FBO path. The existing `WebGPURenderToTexture` demo and
the `pyngl-webgpu` skill are the references for the wgpu-py specifics.

## References

- LearnOpenGL — [Diffuse Irradiance](https://learnopengl.com/PBR/IBL/Diffuse-irradiance)
  and [Specular IBL](https://learnopengl.com/PBR/IBL/Specular-IBL), the
  derivations the bake stages are ported from.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 —
  [course notes](https://blog.selfshadow.com/publications/s2013-shading-course/),
  the split-sum approximation and the importance-sampled BRDF LUT.
- The existing [`PBR/IBL`](../../../PBR/IBL) and [`PBR/SimplePBR`](../../../PBR/SimplePBR)
  demos for the shared Cook-Torrance term and grid layout.
