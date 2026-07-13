#version 330 core

uniform vec3 lineColour;
out vec4 fragColour;

void main()
{
    fragColour = vec4(lineColour, 1.0);
}
