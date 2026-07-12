# ShowMipMap

This demo manually creates mip-maps as blocks of colours Red - Green - Blue - Yellow - White - Purple, when running you can see what level mip map is used by the colours displayed. Level 0 is Red (closest) Purple is lowest level. 

GL_TEXTURE_MIN_LOD
Sets the minimum level-of-detail parameter. This floating-point value limits the selection of highest resolution mipmap (lowest mipmap level). The initial value is -1000.

GL_TEXTURE_MAX_LOD
Sets the maximum level-of-detail parameter. This floating-point value limits the selection of the lowest resolution mipmap (highest mipmap level). The initial value is 1000.

GL_TEXTURE_MAX_LEVEL
Sets the index of the highest defined mipmap level. This is an integer value. The initial value is 1000.

To set the levels.


## References

- L. Williams, "Pyramidal Parametrics", SIGGRAPH 1983 — [ACM](https://dl.acm.org/doi/10.1145/964967.801126) — the paper that introduced mip-mapping.
- [glTexParameter — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexParameter.xhtml) — `GL_TEXTURE_MIN_LOD` / `MAX_LOD` / `MAX_LEVEL` as used here.
- [LearnOpenGL — Textures](https://learnopengl.com/Getting-started/Textures) — mipmap generation and the min-filter modes.
