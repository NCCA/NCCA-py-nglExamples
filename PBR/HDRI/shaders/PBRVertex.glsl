#version 410 core
// Identical vertex stage to PBR/SimplePBR's PBRVertex.glsl -- copied rather
// than shared, since demo folders stay standalone (no cross-folder imports).
layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

out vec3 WorldPos;
out vec3 Normal;

uniform mat4 MVP;
uniform mat3 normalMatrix;
uniform mat4 M;

void main()
{
    WorldPos = vec3(M * vec4(inVert, 1.0));
    Normal = normalMatrix * inNormal;
    gl_Position = MVP * vec4(inVert, 1.0);
}
