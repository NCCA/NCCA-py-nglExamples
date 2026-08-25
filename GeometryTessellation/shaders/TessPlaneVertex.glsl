#version 410 core

// Passes control points straight through in *world* space -- the TCS/TES
// need world-space positions to compute a camera-distance LOD and to
// displace along world y, so all the transform work (view/projection)
// happens after tessellation, in the TES.
layout(location = 0) in vec3 inVert;

uniform mat4 M;

out vec3 vWorldPos;

void main()
{
    vWorldPos = (M * vec4(inVert, 1.0)).xyz;
}
