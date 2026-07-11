#version 330 core

/// Shared vertex shader for the opaque, naive-blend and OIT accumulation
/// passes. As well as the usual clip-space position and view-space normal
/// it passes the view-space z down to the fragment shader, which the OIT
/// weight function needs.

layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;

uniform mat4 MVP;
uniform mat4 MV;
/// inverse transpose of MV
uniform mat3 normalMatrix;

out vec3 fragNormal;
out float viewZ;

void main()
{
    fragNormal = normalize(normalMatrix * inNormal);
    vec4 eye = MV * vec4(inVert, 1.0);
    viewZ = eye.z; // negative in front of the camera
    gl_Position = MVP * vec4(inVert, 1.0);
}
