# AnimatedTextures

500 GPU billboards (camera-facing quads built in a geometry shader from a
single point each) sample one of 3 fire sprite-sheet textures, scrolling
through 10 animation frames over time. Pure-black texels are discarded for
chroma-key transparency.

## Controls
`space` : toggle animation
Left-drag : orbit, Right-drag : pan, Wheel : zoom

## References

- [LearnOpenGL — Geometry Shader](https://learnopengl.com/Advanced-OpenGL/Geometry-Shader) — emitting new primitives (here, a quad per point) from a geometry shader.
- [opengl-tutorial — Billboards](http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/) — the camera-facing quad construction used for each sprite.
- [LearnOpenGL — Blending](https://learnopengl.com/Advanced-OpenGL/Blending) — fragment `discard` and transparency, the basis of the chroma-key effect.
- W. T. Reeves, "Particle Systems — A Technique for Modeling a Class of Fuzzy Objects", ACM TOG 1983 — [ACM](https://dl.acm.org/doi/10.1145/357318.357320) — the origin of sprite-based fire/smoke effects.
