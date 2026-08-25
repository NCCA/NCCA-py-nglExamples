#version 330 core

/// Unlit flat colour output, shared by the outline pass (orange silhouette)
/// and the stencil-visualise pass (translucent tint over the stencilled
/// pixels) -- both just want "paint this Colour, no shading".
uniform vec4 Colour;

layout (location = 0) out vec4 fragColour;

void main()
{
    fragColour = Colour;
}
