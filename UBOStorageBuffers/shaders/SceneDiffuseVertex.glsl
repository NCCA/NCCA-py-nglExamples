#version 330 core

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// SceneBlock is bound once by the CPU (binding point 0) and shared, byte
// for byte, by this program AND the SceneGrid program below -- glUniformBlockBinding
// wires each program's *own* "SceneBlock" interface to the same binding point.
layout(std140) uniform SceneBlock
{
    mat4 VP;
    vec4 lightPos;
    vec4 lightColour;
};

// per-object matrices are ordinary uniforms -- they change every draw call,
// so putting them in a UBO would buy nothing (see README).
uniform mat4 M;
uniform mat3 normalMatrix;

out vec3 fragPosWorld;
out vec3 fragNormal;

void main()
{
    vec4 worldPos = M * vec4(inVert, 1.0);
    fragPosWorld = worldPos.xyz;
    fragNormal = normalize(normalMatrix * inNormal);
    gl_Position = VP * worldPos;
}
