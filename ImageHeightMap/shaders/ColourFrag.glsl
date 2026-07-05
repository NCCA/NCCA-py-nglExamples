#version 330 core
in vec3 vertColour;
out vec4 fragColour;

void main()
{
    fragColour = vec4(vertColour, 1.0);
}
