#version 330 core

in vec2 vertUV;
out vec4 fragColour;

uniform sampler2D spriteTex;

void main()
{
    vec4 texel = texture(spriteTex, vertUV);
    // Alpha-tested cutout so the sprite still reads as a soft round dot
    // with GL_BLEND off (the "B" default state) -- only real blending
    // needs the depth-write-off / back-to-front dance done in main.py.
    if (texel.a < 0.05)
        discard;
    fragColour = texel;
}
