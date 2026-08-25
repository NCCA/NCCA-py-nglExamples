#version 330 core

/// Final composite to the screen.
///
/// In modes 1 and 2 the transparent geometry was blended straight into the
/// opaque buffer, so this is just a copy. In OIT mode it resolves the two
/// accumulation targets:
///
///     transparent = accum.rgb / accum.a          (the weighted average)
///     final = opaque * reveal + transparent * (1 - reveal)

uniform sampler2D opaqueTex;
uniform sampler2D accumTex;
uniform sampler2D revealTex;
uniform bool useOIT;

in vec2 uv;
layout (location = 0) out vec4 fragColour;

void main()
{
    vec3 opaque = texture(opaqueTex, uv).rgb;
    if (!useOIT)
    {
        fragColour = vec4(opaque, 1.0);
        return;
    }
    vec4 accum = texture(accumTex, uv);
    float reveal = texture(revealTex, uv).r;
    vec3 transparent = accum.rgb / max(accum.a, 1e-5);
    fragColour = vec4(opaque * reveal + transparent * (1.0 - reveal), 1.0);
}
