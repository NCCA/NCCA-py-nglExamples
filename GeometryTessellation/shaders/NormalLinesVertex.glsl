#version 330 core

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;

uniform mat4 MV;
uniform mat4 project;
uniform mat3 normalMatrix;

// view-space position / normal, consumed by the geometry shader to build
// the extended line endpoint; gl_Position is the projected line *start*.
out vec3 vPosView;
out vec3 vNormalView;

void main()
{
    vPosView = (MV * vec4(inVert, 1.0)).xyz;
    vNormalView = normalize(normalMatrix * inNormal);
    gl_Position = project * vec4(vPosView, 1.0);
}
