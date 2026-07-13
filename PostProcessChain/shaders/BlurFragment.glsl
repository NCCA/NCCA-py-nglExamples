#version 330 core

/// Separable 5-tap Gaussian blur, one direction per draw (`horizontal`
/// selects H vs V), ping-ponged between two half-res FBOs by main.py for
/// `n_passes` iterations. Offsets are scaled by the texel size of
/// whichever half-res target is currently bound as the source, so the
/// same shader works for both ping-pong buffers regardless of which one
/// is being sampled from this pass.

uniform sampler2D image;
uniform bool horizontal;
uniform vec2 texelSize; // 1.0 / half-res framebuffer size

in vec2 uv;
layout (location = 0) out vec4 fragColour;

// Fixed 5-tap Gaussian weights (sigma ~= 1.6), centre tap first.
const float WEIGHTS[3] = float[](0.227027, 0.316216, 0.070270);

void main()
{
    vec2 step = horizontal ? vec2(texelSize.x, 0.0) : vec2(0.0, texelSize.y);
    vec3 result = texture(image, uv).rgb * WEIGHTS[0];
    for (int i = 1; i < 3; ++i)
    {
        vec2 offset = step * float(i);
        result += texture(image, uv + offset).rgb * WEIGHTS[i];
        result += texture(image, uv - offset).rgb * WEIGHTS[i];
    }
    fragColour = vec4(result, 1.0);
}
