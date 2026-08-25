#version 330 core

in vec3 fragNormal;

/// rgb is the surface colour, a is the opacity used by the blend equation.
/// Note the shader does nothing special for transparency -- it just writes
/// an alpha value. What happens to it is decided entirely by the blend
/// state set on the CPU side (glEnable(GL_BLEND) / glBlendFunc).
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
