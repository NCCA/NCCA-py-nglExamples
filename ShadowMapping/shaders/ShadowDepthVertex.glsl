#version 330 core

// Depth-only pass: render the scene from the light's point of view into the
// shadow map's depth texture. Only position is needed.
layout(location = 0) in vec3 inVert;

uniform mat4 M;
uniform mat4 lightSpaceMatrix;

void main()
{
    gl_Position = lightSpaceMatrix * M * vec4(inVert, 1.0);
}
