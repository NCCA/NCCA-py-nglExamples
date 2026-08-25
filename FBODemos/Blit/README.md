# Blit

![Blit Demo](Blit.png)

This demo shows how to use the glBlitFramebuffer function to copy a framebuffer to another framebuffer. In this demo a shader is run to generate different check pattern in the framebuffer.

Each colour attachment is a different check pattern and the user can then choose which one to display.

It also shows how to create a screen quad for rendering the framebuffer to the screen.

## References

- [glBlitFramebuffer — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glBlitFramebuffer.xhtml) — the framebuffer-to-framebuffer copy used here.
- [OpenGL Wiki — Framebuffer Object](https://www.khronos.org/opengl/wiki/Framebuffer_Object) — colour attachments and read/draw framebuffer bindings.
- [LearnOpenGL — Framebuffers](https://learnopengl.com/Advanced-OpenGL/Framebuffers) — rendering offscreen and displaying the result on a screen quad.
