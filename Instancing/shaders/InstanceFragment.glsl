#version 330 core

in vec3 fragNormal;
in vec4 fragColour;

uniform vec3 lightDir;

layout (location = 0) out vec4 outColour;

void main()
{
    float ndotl = max(dot(normalize(fragNormal), normalize(lightDir)), 0.0);
    vec3 shaded = fragColour.rgb * (0.35 + 0.65 * ndotl);
    outColour = vec4(shaded, fragColour.a);
}
