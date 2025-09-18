#version 410 core

/// @brief projection matrix passed in from camera class in main app
uniform mat4 MVP;

layout (location=0)in vec3 inVert;
layout (location=1)in vec3 inNormal;
layout (location=2)in vec2 inUV;

out vec2 vertUV;

void main()
{
 // calculate the vertex position
 gl_Position = MVP*vec4(inVert, 1.0);
 vertUV=inUV;
}
