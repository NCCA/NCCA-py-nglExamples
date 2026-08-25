# ShowMipmap

This demo uploads each mip level of a texture by hand, filling every level with a flat colour (red at level 0, then green, blue, yellow, white and magenta as the levels get smaller). A grid of textured cubes recedes into the distance so you can see exactly which level the sampler picks — the colour changes at each mip transition.

The uploaded levels are clamped with `GL_TEXTURE_BASE_LEVEL`, `GL_TEXTURE_MAX_LEVEL` and `GL_TEXTURE_MAX_LOD` so only the hand-made levels are ever sampled.

```bash
uv run ShowMipmap/ShowMipmap.py
```

![](MipMap.png)

## References

- L. Williams, "Pyramidal Parametrics", SIGGRAPH 1983 — [ACM](https://dl.acm.org/doi/10.1145/964967.801126) — the paper that introduced mip-mapping.
- [glTexParameter — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexParameter.xhtml) — `GL_TEXTURE_MIN_LOD` / `MAX_LOD` / `MAX_LEVEL` as used here.
- [LearnOpenGL — Textures](https://learnopengl.com/Getting-started/Textures) — mipmap generation and the min-filter modes.
