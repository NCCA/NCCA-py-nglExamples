#version 410 core

in vec3 vertColour;
/// @brief our output fragment colour
layout (location =0)out vec4 fragColour;

void main ()
{
    fragColour.rgb=vertColour;
}
