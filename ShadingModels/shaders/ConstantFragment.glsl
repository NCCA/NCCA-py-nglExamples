#version 330 core
uniform vec3 Colour;
/// @brief our output fragment colour
layout (location = 0) out vec4 fragColour;
void main ()
{
  fragColour.rgb = Colour;
}
