#version 410 core

// The whole "mesh" for this demo is three vertices with no vertex buffer at
// all -- the classic fullscreen-triangle trick (see ScreenTri/ for the
// simplest possible version of this). gl_VertexID picks out (-1,-1),
// (3,-1), (-1,3): a triangle that covers the viewport and then some, so
// there's no seam down the middle of the screen the way a quad-of-two-
// triangles would have.

out vec2 ndc;

void main()
{
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    ndc = vec2(x, y);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
