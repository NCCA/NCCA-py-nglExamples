#version 410 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 colour;

out vec3 vertex_colour;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * vec4(position, 0.0, 1.0);
    vertex_colour = colour;
}
