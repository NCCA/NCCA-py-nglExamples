#version 330 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;

out vec3 fragPos;
out vec3 fragNormal;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat3 normalMatrix;

void main()
{
    fragPos = vec3(MV * vec4(inVert, 1.0));
    fragNormal = normalMatrix * inNormal;
    gl_Position = MVP * vec4(inVert, 1.0);
}
