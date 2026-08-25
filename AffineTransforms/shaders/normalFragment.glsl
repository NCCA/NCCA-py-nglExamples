#version 330 core
layout (location=0) out vec4 fragColour;
in vec4 perNormalColour;

void main()
{
  fragColour = perNormalColour;
}
