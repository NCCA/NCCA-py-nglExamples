#version 330 core

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

uniform mat4 MVP;
uniform mat4 M;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

out vec3 FragPosWorld;
out vec3 NormalWorld;
out vec4 FragPosLightSpace;

void main()
{
    vec4 worldPos = M * vec4(inVert, 1.0);
    FragPosWorld = worldPos.xyz;
    NormalWorld = normalize(normalMatrix * inNormal);
    // world position re-projected through the light's view-projection --
    // compared against the depth map in the fragment shader.
    FragPosLightSpace = lightSpaceMatrix * worldPos;
    gl_Position = MVP * vec4(inVert, 1.0);
}
