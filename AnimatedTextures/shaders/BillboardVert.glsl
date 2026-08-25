#version 330 core
layout(location = 0) in vec4 inVert;
layout(location = 1) in float inOffset;

flat out float whichTexture;
flat out float frameOffset;

void main()
{
    gl_Position = inVert;
    whichTexture = inVert.w;
    frameOffset = inOffset;
}
