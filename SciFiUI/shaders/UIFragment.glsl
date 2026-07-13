#version 410 core
// Flat colour fragment shader. The whole scene is drawn in monochrome
// intensities; the CRT post-process pass applies the phosphor tint.
uniform vec4 Colour;
out vec4 fragColour;

void main()
{
    fragColour = Colour;
}
