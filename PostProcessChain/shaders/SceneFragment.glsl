#version 330 core

/// Scene pass: renders straight into an RGBA16F HDR framebuffer (no
/// clamping, no gamma -- both happen only in the final tonemap pass).
///
/// Two modes selected per-draw by the `emissive` uniform:
///   emissive == true:  outputs `Colour` verbatim, deliberately with
///                       components > 1.0 (e.g. (8, 4, 0.5)) -- these are
///                       the "too bright to display" values the bloom
///                       pass will pick out.
///   emissive == false: ordinary N.L Lambert shading in view space, so
///                       the grid/teapot read as normally lit geometry
///                       that never itself exceeds 1.0.

uniform vec3 Colour;
uniform bool emissive;
uniform vec3 lightDir; // view-space, normalised

in vec3 viewNormal;
in vec3 viewPos;

layout (location = 0) out vec4 fragColour;

void main()
{
    if (emissive)
    {
        fragColour = vec4(Colour, 1.0);
        return;
    }
    float nl = max(dot(normalize(viewNormal), normalize(lightDir)), 0.0);
    vec3 ambient = Colour * 0.15;
    fragColour = vec4(ambient + Colour * nl, 1.0);
}
