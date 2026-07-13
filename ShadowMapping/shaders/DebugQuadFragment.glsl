#version 330 core

in vec2 uv;
uniform sampler2D shadowMap;
out vec4 fragColour;

void main()
{
    // The light uses an *orthographic* projection, so NDC depth is already
    // linear in view-space distance -- unlike a perspective shadow map,
    // there is no 1/z term to undo here, so displaying the raw texture
    // value is already a correctly "linearised" visualisation.
    float d = texture(shadowMap, uv).r;
    fragColour = vec4(vec3(d), 1.0);
}
