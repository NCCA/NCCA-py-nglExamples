# SkyBox / EnvMap

![](SkyBoxEnvMap.png)

A procedural sky (see `cubemap_gen.py` — numpy only, no image assets needed) drawn as a skybox, plus a teapot shaded with reflection, refraction and a Schlick-Fresnel mix sampled from the same cubemap. There are two versions of the same demo:

- `main.py` — OpenGL, `GL_TEXTURE_CUBE_MAP` uploaded with `glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ...)` per face
- `SkyBoxEnvMapWebGPU.py` — WebGPU, one texture with 6 array layers viewed with `dimension="cube"` and sampled as `texture_cube<f32>`

Both generate and upload the identical six-face cubemap from `cubemap_gen.py`, so the two demos render the same sky.

## Controls

| Key | Action |
| :-- | :-- |
| `M` | cycle teapot shading mode: reflect / refract / Fresnel mix / plain diffuse |
| `+` / `-` | increase / decrease the index of refraction (refract & Fresnel modes) |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits |

## The teaching points

1. **A skybox is a unit cube sampled with a direction, not a UV.** The cube's own local position *is* the cubemap lookup direction, which is why no texture coordinates are needed for it.
2. **The view matrix loses its translation for the skybox.** Stripping translation (`P * mat4(mat3(V))` in GLSL, an equivalent CPU-built rotation-only matrix in WGSL) keeps the sky centred on the camera regardless of where it moves — it should feel infinitely far away.
3. **The skybox is pushed to the far plane and drawn *last*.** `gl_Position = pos.xyww` (GLSL) / setting `clip.z = clip.w` (WGSL) forces every skybox fragment to the far depth plane; combined with `GL_LEQUAL`/`less-equal` depth compare and depth *write* disabled, it only survives where nothing opaque has already been drawn there. Despite what most tutorials do, drawing opaque geometry first and the skybox last is the *faster* order on modern hardware — early-z rejects the (full-screen, expensive-ish) skybox fragments behind already-shaded pixels instead of the skybox blindly filling the screen before anything else exists to reject against.
4. **`reflect()`/`refract()` need world space, not view space.** Because the cubemap is sampled in world space, the teapot shader carries the model matrix and a world-space normal matrix (`Mat3.from_mat4(M).inverse().transposed()`, not the usual `MV`-based one) plus the camera's world position, rather than the view-space lighting setup used elsewhere in these demos.
5. **Fresnel mix is what makes glass/water look right.** A pure reflect or refract teapot looks flat; blending them with a Schlick-Fresnel term (`F0 + (1-F0)(1-cosθ)^5`) makes grazing angles reflect more and head-on angles refract more, which is what a real dielectric surface does.

## Cubemap face order and the horizon-continuity test

The classic cubemap bug is getting face order or per-face orientation wrong, which shows up as a visible seam at a cube edge. `cubemap_gen.py` computes each face's colour from a proper per-texel 3D direction vector (not a naive "paint a gradient across the image" approach), and the tests check that the shared edge between the `+x` and `+z` faces produces bit-identical pixels from both faces — the same check is worth doing visually with a manual orbit in either demo.

## Tests

```bash
uv run pytest SkyBoxEnvMap/tests
```

## References

- [LearnOpenGL — Cubemaps](https://learnopengl.com/Advanced-OpenGL/Cubemaps) — skybox rendering, the `mat3(view)` translation-stripping trick, and reflection/refraction cubemap sampling.
- [OpenGL Wiki — Cubemap Texture](https://www.khronos.org/opengl/wiki/Cubemap_Texture) — the canonical face order and per-face `s,t` mapping table.
- B. Smits, "Efficiency Issues for Ray Tracing" (Schlick's approximation origin context) and [Wikipedia — Schlick's approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation) — the Fresnel term used in the mix mode.
- [WebGPU spec — texture_cube sampling](https://gpuweb.github.io/gpuweb/wgsl/#texturesample) — `textureSample` with `texture_cube<f32>`.
