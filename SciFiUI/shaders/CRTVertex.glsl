#version 410 core
// Full-screen triangle generated from gl_VertexID -- no vertex buffer needed.
// https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/
out vec2 texCoord;

void main()
{
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    texCoord = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
