#version 330 core

in vec3 direction;
uniform samplerCube skybox;
out vec4 fragColour;

void main()
{
    fragColour = texture(skybox, direction);
}
