#version 330 core

/// @brief cube vertex position (per-vertex, VBO 0)
layout (location = 0) in vec3 inVert;
/// @brief cube vertex normal (per-vertex, VBO 0)
layout (location = 1) in vec3 inNormal;
/// @brief per-instance offset (xyz) + uniform scale (w) -- VBO 1, divisor 1
layout (location = 3) in vec4 instOffsetScale;
/// @brief per-instance colour -- VBO 1, divisor 1
layout (location = 4) in vec4 instColour;

uniform mat4 MVP;
uniform mat3 normalMatrix;

/// true: read the per-instance attributes above (one glDrawArraysInstanced
/// call draws every cube). false: read uOffsetScale/uColour instead, so the
/// exact same shader can also be driven by a naive Python loop of n
/// glDrawArrays calls -- one code path, two draw strategies, so the frame
/// time difference you see is *only* about the draw call count.
uniform bool instanced;
uniform vec4 uOffsetScale;
uniform vec4 uColour;

out vec3 fragNormal;
out vec4 fragColour;

void main()
{
    vec4 offsetScale = instanced ? instOffsetScale : uOffsetScale;
    fragColour = instanced ? instColour : uColour;
    vec3 pos = inVert * offsetScale.w + offsetScale.xyz;
    fragNormal = normalize(normalMatrix * inNormal);
    gl_Position = MVP * vec4(pos, 1.0);
}
