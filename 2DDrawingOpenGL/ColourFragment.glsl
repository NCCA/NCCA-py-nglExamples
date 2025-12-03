#version 410 core


in vec3 vertex_colour;
out vec4 fragment_colour;
void main()
{

    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0)
    {
        discard;
    }

    fragment_colour = vec4(vertex_colour, 1.0);
}
