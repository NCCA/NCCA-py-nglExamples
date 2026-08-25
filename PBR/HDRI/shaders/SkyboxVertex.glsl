#version 410 core

layout(location = 0) in vec3 inVert;

// MVP here is already P * mat4(mat3(V)) -- translation stripped on the CPU
// so the skybox never moves relative to the camera.
uniform mat4 MVP;

out vec3 direction;

void main()
{
    // the cube's local position *is* the sample direction: the skybox is a
    // unit cube, we never need to normalise on the CPU side, the GPU
    // interpolation + normalize() in the fragment shader is enough.
    direction = inVert;
    vec4 pos = MVP * vec4(inVert, 1.0);
    // force depth to the far plane (z == w after the divide) so the skybox
    // never wins a depth test against real geometry, whatever draw order.
    gl_Position = pos.xyww;
}
