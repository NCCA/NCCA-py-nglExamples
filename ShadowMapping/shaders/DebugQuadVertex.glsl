#version 330 core

// Tiny standalone shader for the screen-corner debug inset: positions are
// already in clip space (a unit quad), the viewport is shrunk to the
// corner rectangle before drawing so this maps exactly onto it.
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;

out vec2 uv;

void main()
{
    uv = inUV;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
