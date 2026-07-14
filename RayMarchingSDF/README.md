# Ray Marching Signed Distance Fields

![](RayMarchingSDF.png)

A ground plane, a sphere, a box and a torus melted together with a smooth minimum, plus one sphere orbiting overhead and melting through the rest of the scene as it goes. There is no geometry at all — the whole thing is one fragment shader per backend, sphere-traced pixel by pixel. This is the demo I use to show students that rendering doesn't have to mean triangles.

- `main.py` — OpenGL, drawing the classic no-VBO fullscreen triangle from [ScreenTri](../ScreenTri)
- `RayMarchingWebGPU.py` — WebGPU, the same trick with a `@builtin(vertex_index)` triangle (see [OITransparency](../OITransparency)'s composite pass)

Both shaders — `shaders/RayMarchFragment.glsl` and `RayMarch.wgsl` — are line-for-line transcriptions of each other, and both are transcriptions of `sdf_maths.py`, a numpy-only reference implementation unit tested in `tests/`. Same function names in all three: `sd_sphere`, `sd_box`, `sd_torus`, `sd_plane`, `smooth_min`, `scene`. If you change the scene in one, change it in all three or the two renderers will visibly disagree.

## How sphere tracing works

Rather than rasterising triangles, each pixel fires a ray from the camera and walks it forward. The scene isn't a mesh, it's a *distance field*: `scene(p)` returns how far `p` is from the nearest surface, negative if you're inside something. That distance is always a safe step size — nothing in the scene is closer than that, in any direction — so you march by exactly that far, evaluate again, and repeat:

```glsl
float travelled = 0.0;
for (int i = 0; i < MAX_STEPS; ++i) {
    vec3 p = camPos + rayDir * travelled;
    float d = scene(p, time, smoothK);
    if (d < EPSILON) break;      // close enough: call it a hit
    travelled += d;
    if (travelled > FAR) break;  // gave up
}
```

100 steps, epsilon `1e-3`, far plane 40 units — past that the ray is considered to have escaped into the sky. The surface normal falls out of the same field for free: nudge `p` a tiny amount along each axis and see how the distance changes (a central-difference gradient), no vertex normals required.

Compound shapes come from combining distance fields with `min`. A hard `min` gives a sharp seam where two shapes meet; `smooth_min` (Inigo Quilez's polynomial smin) rounds it into a fillet, which is what makes the orbiting sphere look like it's *melting* through the rest of the scene rather than just intersecting it. `+`/`-` widen or narrow that blend radius live.

Shadows and ambient occlusion are the same trick again, aimed differently: a soft shadow is a second march, this time from the surface towards the light, tracking how close it grazes other geometry along the way; AO is five short taps along the normal checking whether the field is "more full" nearby than empty space would predict.

## Controls

| Key | Action |
| :-- | :-- |
| `S` | toggle soft shadows |
| `O` | toggle ambient occlusion |
| `N` | visualise surface normals |
| `I` | visualise the iteration count as a heat map — blue is cheap, red is near `MAX_STEPS`, and it's the best single picture for explaining why ray marching cost depends on the view and the scene |
| `+` / `-` | widen / narrow the smooth-min blend radius |
| `Space` | pause / resume the orbiting sphere |
| LMB / RMB / wheel | rotate / pan / zoom, `Esc` quits |

The camera uses the usual PyNGL mouse orbit (`spin_x_face`/`spin_y_face` for rotate, `model_position` for pan and zoom), but there's no model matrix to apply it to — instead `main.py` and `RayMarchingWebGPU.py` turn that state into a camera position and an orthonormal forward/right/up basis on the CPU, and the shader only has to combine that basis with each pixel's screen-space offset and field of view to get a ray direction.

## GLSL vs WGSL

The two shaders are close enough to diff directly. The differences that remain are backend syntax, not the ray marcher itself:

- WGSL has no `#version`/`#define` — constants use `const` instead of GLSL's `#version 410 core` preamble and `const` qualifiers (same keyword, just consistently used for everything in WGSL).
- Uniforms arrive differently: GLSL gets a flat list of `uniform` scalars/vectors set individually from Python; WGSL gets one `Params` uniform buffer whose numpy dtype mirrors the struct layout by hand (each `vec3` padded out to 16 bytes by the `f32` that follows it).
- Loop syntax (`for (int i = 0; ...)` vs `for (var i = 0; ...)`) and `bool` uniforms (GLSL `int` toggle vs WGSL `u32`) are the only other differences — the SDF functions and the `scene()` composition are identical statement for statement.

## Tests

```bash
uv run pytest RayMarchingSDF/tests
```

Covers exact distances (a sphere's surface is zero, its centre is negative), the two `smooth_min` properties that make it a smooth *minimum* (never above the hard `min`, and converging to it once the two inputs are far enough apart), and that `estimate_normal` recovers a sphere's radial normal from its distance field alone.

## References

- Inigo Quilez, [Distance Functions](https://iquilezles.org/articles/distfunctions/) — the primitive SDFs and `smooth_min` used here.
- Inigo Quilez, [Ray Marching and Signed Distance Fields](https://iquilezles.org/articles/raymarchingdf/) — the march loop, normals-from-gradient and soft shadow technique.
- [The Book of Shaders — Signed Distance Functions](https://thebookofshaders.com/07/) — a gentler introduction if the above two are a lot to take in at once.
- [ScreenTri](../ScreenTri), [OITransparency](../OITransparency) — the fullscreen-triangle tricks this demo's "geometry" is built from.
