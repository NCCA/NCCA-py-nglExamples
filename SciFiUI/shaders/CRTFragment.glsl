#version 410 core
// CRT post-process pass. The scene is rendered in monochrome into an FBO
// texture; this shader applies everything that makes it feel like a dying
// 1979 shipboard monitor:
//   - barrel distortion (curved glass)
//   - phosphor tint (green or amber, from a uniform)
//   - scanlines and a subtle slot mask
//   - a slow rolling brightness bar
//   - per-pixel noise, flicker and a vignette
uniform sampler2D screenTex;
uniform float iTime;
uniform vec2 iResolution;
uniform vec3 phosphor;
uniform int effectsOn;

in vec2 texCoord;
out vec4 fragColour;

// cheap hash based noise
float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    vec2 uv = texCoord;

    if (effectsOn == 1)
    {
        // barrel distortion - kept small so mouse picking stays accurate
        vec2 c = uv * 2.0 - 1.0;
        float r2 = dot(c, c);
        c *= 1.0 + 0.035 * r2 + 0.015 * r2 * r2;
        uv = c * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        {
            fragColour = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    // base sample plus a cheap 4 tap glow so bright lines bloom slightly
    vec2 px = 1.0 / iResolution;
    vec3 col = texture(screenTex, uv).rgb;
    vec3 glow = texture(screenTex, uv + vec2(px.x * 2.0, 0.0)).rgb
              + texture(screenTex, uv - vec2(px.x * 2.0, 0.0)).rgb
              + texture(screenTex, uv + vec2(0.0, px.y * 2.0)).rgb
              + texture(screenTex, uv - vec2(0.0, px.y * 2.0)).rgb;
    col += glow * 0.22;

    // scene is drawn in monochrome, convert to a single intensity then tint
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    vec3 tinted = phosphor * lum;

    if (effectsOn == 1)
    {
        // scanlines
        tinted *= 0.80 + 0.20 * sin(uv.y * iResolution.y * 3.14159);
        // very subtle vertical slot mask
        tinted *= 0.95 + 0.05 * sin(uv.x * iResolution.x * 3.14159);
        // slow rolling bar drifting down the screen
        float bar = exp(-45.0 * pow(fract(uv.y - iTime * 0.04) - 0.5, 2.0));
        tinted *= 1.0 + 0.06 * bar;
        // static / noise
        tinted += phosphor * 0.035 * hash(uv * iResolution + vec2(iTime * 120.0));
        // mains hum flicker
        tinted *= 0.97 + 0.03 * sin(iTime * 47.0);
    }

    // vignette (always on, part of the tube)
    vec2 v = uv * (1.0 - uv);
    tinted *= pow(v.x * v.y * 18.0, 0.28);

    fragColour = vec4(tinted, 1.0);
}
