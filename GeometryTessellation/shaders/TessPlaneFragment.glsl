#version 410 core

in vec3 teWorldPos;
in vec3 teNormalWorld;
in float teHeight;

uniform vec3 lightDirWorld;
uniform float heightScale;

out vec4 fragColour;

void main()
{
    vec3 n = normalize(teNormalWorld);
    float nDotL = max(dot(n, normalize(lightDirWorld)), 0.0);

    // Shade by height + N.L: a low/high colour ramp modulated by the
    // Lambert term, so both the noise displacement and the tessellation
    // density (visible via the wireframe toggle) are readable at once.
    float t = clamp(teHeight / max(heightScale, 0.0001) * 0.5 + 0.5, 0.0, 1.0);
    vec3 low = vec3(0.15, 0.25, 0.35);
    vec3 high = vec3(0.85, 0.8, 0.65);
    vec3 base = mix(low, high, t);

    vec3 colour = base * (0.25 + 0.75 * nDotL);
    fragColour = vec4(colour, 1.0);
}
