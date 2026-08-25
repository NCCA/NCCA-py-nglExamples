#version 330 core

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

uniform mat4 MVP;
uniform mat4 M;
// world-space normal matrix (Mat3.from_mat4(M).inverse().transposed()) --
// reflect/refract are computed in world space here, not view space, so we
// need M alone rather than the usual view*model normal matrix.
uniform mat3 normalMatrix;

out vec3 worldPos;
out vec3 worldNormal;

void main()
{
    vec4 wp = M * vec4(inVert, 1.0);
    worldPos = wp.xyz;
    worldNormal = normalize(normalMatrix * inNormal);
    gl_Position = MVP * vec4(inVert, 1.0);
}
