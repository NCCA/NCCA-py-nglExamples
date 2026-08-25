#version 330 core

layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat3 normalMatrix;

out vec3 viewNormal;
out vec3 viewPos;

void main()
{
    viewNormal = normalize(normalMatrix * inNormal);
    viewPos = (MV * vec4(inVert, 1.0)).xyz;
    gl_Position = MVP * vec4(inVert, 1.0);
}
