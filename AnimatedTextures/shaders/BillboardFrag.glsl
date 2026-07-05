#version 330 core
flat in float texID;
in vec2 texCoord;
out vec4 fragColour;

uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;

void main()
{
    vec4 colour;
    if (texID < 0.5)
        colour = texture(tex1, texCoord);
    else if (texID < 1.5)
        colour = texture(tex2, texCoord);
    else
        colour = texture(tex3, texCoord);

    if (colour.rgb == vec3(0.0))
        discard;

    fragColour = colour;
}
