#version 330 core

/// Plain shaded output used for the opaque pass and for modes 1 and 2
/// (naive and sorted alpha blending). Transparency happens in the blend
/// unit, not here.

in vec3 fragNormal;
in float viewZ;

uniform vec4 Colour;
/// view-space light direction
uniform vec3 lightDir;

layout (location = 0) out vec4 fragColour;

void main()
{
    // double-sided N.L so the backs of the panels shade too
    float ndotl = abs(dot(normalize(fragNormal), normalize(lightDir)));
    vec3 shaded = Colour.rgb * (0.25 + 0.75 * ndotl);
    fragColour = vec4(shaded, Colour.a);
}
