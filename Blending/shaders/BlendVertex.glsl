#version 330 core

/// @brief the vertex passed in
layout (location = 0) in vec3 inVert;
/// @brief the normal passed in
layout (location = 1) in vec3 inNormal;

uniform mat4 MVP;
/// transforms normals into view space (inverse transpose of MV)
uniform mat3 normalMatrix;

out vec3 fragNormal;

void main()
{
    fragNormal = normalize(normalMatrix * inNormal);
    gl_Position = MVP * vec4(inVert, 1.0);
}
