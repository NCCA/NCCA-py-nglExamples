#version 330 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inColour;

out vec3 vertColour;

uniform mat4 MVP;

void main()
{
    vertColour = inColour;
    gl_Position = MVP * vec4(inVert, 1.0);
}
