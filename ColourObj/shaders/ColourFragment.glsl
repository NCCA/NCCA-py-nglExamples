#version 410 core
in vec3 vertColour;
layout(location = 0) out vec4 fragColour;

void main()
{
    fragColour = vec4(vertColour, 1.0);
}
