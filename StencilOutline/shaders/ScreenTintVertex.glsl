#version 330 core

/// Full-screen triangle generated from gl_VertexID -- no VBO needed (an
/// empty VAO must still be bound in core profile). Same technique as
/// PostProcessChain/shaders/CompositeVertex.glsl. Used by the `V`
/// stencil-visualise mode: the fragment stage is stencil-tested EQUAL 1
/// per pixel, so only pixels already marked by the selected object's
/// stencil write get tinted.
void main()
{
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
