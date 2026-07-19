# Simple FBO

![Simple FBO Demo](SimpleFBO.png)

This demo shows how to use a framebuffer object (FBO) to render to a texture. This works with two passes, the first pass renders a teapot to a texture. The 2nd pass then renders this teapot as the texture to the quad and sphere.

## References

- [LearnOpenGL — Framebuffers](https://learnopengl.com/Advanced-OpenGL/Framebuffers) — render-to-texture with FBOs, exactly the two-pass structure used here.
- [OpenGL Frame Buffer Object (songho.ca)](https://www.songho.ca/opengl/gl_fbo.html) — FBO setup, completeness rules and attachment types.
- [OpenGL Wiki — Framebuffer Object](https://www.khronos.org/opengl/wiki/Framebuffer_Object) — the reference documentation.
