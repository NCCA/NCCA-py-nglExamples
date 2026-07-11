#version 330 core

/// Full-screen triangle generated from gl_VertexID -- no VBO needed (an
/// empty VAO must still be bound in core profile). Same technique as the
/// ScreenTri demo.

out vec2 uv;

void main()
{
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
