#version 410 core
layout (location = 0) out vec4 fragColour;
uniform sampler2D tex;
in vec2 vertUV;
// Note: no .bgr swizzle here (unlike the C++ source) -- this demo's DXT1
// encoder (dxt_texture.py) compresses straight RGB, not squish's BGR
// output, so a plain .rgb read is correct.
void main()
{
    fragColour = texture(tex, vertUV);
}
