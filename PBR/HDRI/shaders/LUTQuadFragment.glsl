#version 410 core
in vec2 uv;
uniform sampler2D lut;
out vec4 fragColour;

void main() {
    // A (scale) in red, B (bias) in green -- matches the classic
    // LearnOpenGL LUT preview image.
    vec2 ab = texture(lut, uv).rg;
    fragColour = vec4(ab, 0.0, 1.0);
}
