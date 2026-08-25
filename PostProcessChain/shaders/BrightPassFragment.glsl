#version 330 core

/// Bright pass: sample the full-res HDR scene with LINEAR filtering into a
/// half-resolution RGBA16F target, keeping only the amount of each channel
/// above `threshold`. This both downsamples (free softness/speed for the
/// blur that follows) and isolates the pixels that will bloom.

uniform sampler2D sceneTex;
uniform float threshold;

in vec2 uv;
layout (location = 0) out vec4 fragColour;

void main()
{
    vec3 colour = texture(sceneTex, uv).rgb;
    fragColour = vec4(max(colour - vec3(threshold), 0.0), 1.0);
}
