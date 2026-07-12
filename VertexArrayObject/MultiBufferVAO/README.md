# Boid

This demo creates a simple Boid shaped VertexArrayObject using just vertices but using a multi-buffer VAO which is a single buffer for the vertices and a separate buffer for the Normals.

## References

- [OpenGL Wiki — Vertex Specification](https://www.khronos.org/opengl/wiki/Vertex_Specification) — binding multiple VBOs to one VAO, one per attribute.
- [OpenGL Wiki — Vertex Specification Best Practices](https://www.khronos.org/opengl/wiki/Vertex_Specification_Best_Practices) — separate vs interleaved buffer trade-offs.
