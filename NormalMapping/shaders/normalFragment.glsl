#version 330 core
#pragma optionNV(unroll all)
/// @brief our output fragment colour
out vec4 fragColour;
in vec4 perNormalColour;

void main ()
{
	fragColour = perNormalColour;
}

