#version 410 core
in vec3 direction;
uniform samplerCube skybox;
uniform float lod;          // debug: prefilter mip to show; 0 for base env
out vec4 fragColour;
void main() {
    vec3 colour = textureLod(skybox, direction, lod).rgb;
    colour = colour / (colour + vec3(1.0));       // Reinhard tonemap
    colour = pow(colour, vec3(1.0 / 2.2));        // gamma
    fragColour = vec4(colour, 1.0);
}
