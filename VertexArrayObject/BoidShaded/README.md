# BoidShaded

This demo shows how to create a simple Boid shaped VertexArrayObject using just vertices. It uses the SimpleVAO factory to create the VertexArrayObject from a list of vertices stored in the Vec3Array class.

Normals are calculated using the Utils.calculate_normal function.

This demos uses one buffer packed Vertices[] -> Normnals[] and shows how to set the buffer offsets for correct rendering.

## References

- [OpenGL Wiki — Vertex Specification](https://www.khronos.org/opengl/wiki/Vertex_Specification) — attribute offsets into a single packed buffer, as used here.
- [OpenGL Wiki — Vertex Specification Best Practices](https://www.khronos.org/opengl/wiki/Vertex_Specification_Best_Practices) — packed vs interleaved vs separate buffer layouts.
- [LearnOpenGL — Basic Lighting](https://learnopengl.com/Lighting/Basic-Lighting) — why per-face normals (cross products) are needed for shading.
