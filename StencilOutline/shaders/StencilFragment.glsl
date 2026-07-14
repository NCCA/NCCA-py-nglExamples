#version 330 core

in vec3 fragNormal;

/// flat per-object colour, shaded with a single directional light
uniform vec4 Colour;
/// view-space light direction
uniform vec3 lightDir;

layout (location = 0) out vec4 fragColour;

void main()
{
    float ndotl = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);
    vec3 shaded = Colour.rgb * (0.3 + 0.7 * ndotl);
    fragColour = vec4(shaded, Colour.a);
}
