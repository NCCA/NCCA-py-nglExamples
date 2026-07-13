#version 330 core

/// Final composite to the screen: adds the (upsampled, LINEAR-filtered)
/// bloom blur back onto the full-res HDR scene, applies exposure, then one
/// of three tonemap operators, then gamma 2.2. The three-way operator
/// choice and the exposure/gamma steps are exactly what makes an HDR
/// pipeline "HDR" -- without them the >1.0 emissive values would simply
/// clip to white the moment they landed in an 8-bit framebuffer, which is
/// the naive-looking failure mode `H` (split screen, left half) shows
/// deliberately.
///
/// Reinhard and ACES constants mirror tonemap_maths.py verbatim -- see
/// that module's docstring/tests for the numpy reference these numbers
/// must agree with.

uniform sampler2D sceneTex;
uniform sampler2D bloomTex;
uniform bool bloomEnabled;
uniform float bloomStrength;
uniform float exposure;
uniform int operatorMode; // 1 = clamp (none), 2 = Reinhard, 3 = ACES fitted
uniform bool splitScreen;
uniform float screenWidth;

in vec2 uv;
layout (location = 0) out vec4 fragColour;

vec3 reinhard(vec3 c)
{
    return c / (1.0 + c); // matches tonemap_maths.reinhard
}

vec3 acesFitted(vec3 c)
{
    // Narkowicz 2015 fitted ACES curve -- matches tonemap_maths.ACES_*.
    const float a = 2.51;
    const float b = 0.03;
    const float cc = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0, 1.0);
}

vec3 tonemap(vec3 colour, int mode)
{
    if (mode == 2) return reinhard(colour);
    if (mode == 3) return acesFitted(colour);
    return clamp(colour, 0.0, 1.0); // 1: none, just clamp
}

void main()
{
    vec3 scene = texture(sceneTex, uv).rgb;
    vec3 bloom = bloomEnabled ? texture(bloomTex, uv).rgb : vec3(0.0);
    vec3 hdr = scene + bloomStrength * bloom;

    bool rawSide = splitScreen && gl_FragCoord.x < screenWidth * 0.5;
    vec3 ldr;
    if (rawSide)
    {
        // Left half of the split-screen demo: no exposure, no bloom, no
        // tonemap operator -- just clamp straight to [0,1]. This is what
        // an ordinary 8-bit-framebuffer renderer would show.
        ldr = clamp(scene, 0.0, 1.0);
    }
    else
    {
        ldr = tonemap(hdr * exposure, operatorMode);
    }

    fragColour = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
