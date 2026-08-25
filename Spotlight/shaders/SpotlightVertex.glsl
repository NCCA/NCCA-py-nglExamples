#version 410 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

out vec3 fragmentNormal;
out vec3 worldPosition;

uniform mat4 M;
uniform mat4 MVP;
uniform mat3 normal_matrix;

void main()
{
    fragmentNormal = normalize(normal_matrix * inNormal);
    worldPosition = vec3(M * vec4(inVert, 1.0));
    gl_Position = MVP * vec4(inVert, 1.0);
}
