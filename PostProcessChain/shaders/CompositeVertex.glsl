#version 330 core

/// Full-screen triangle generated from gl_VertexID -- no VBO needed (an
/// empty VAO must still be bound in core profile). Identical technique to
/// OITransparency/ScreenTri; shared verbatim by every post-process pass in
/// this demo (bright-pass, blur H/V, tonemap composite).

out vec2 uv;

void main()
{
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
