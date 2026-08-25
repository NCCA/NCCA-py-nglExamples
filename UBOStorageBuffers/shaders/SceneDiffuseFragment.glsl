#version 330 core

in vec3 fragPosWorld;
in vec3 fragNormal;

out vec4 fragColour;

layout(std140) uniform SceneBlock
{
    mat4 VP;
    vec4 lightPos;
    vec4 lightColour;
};

// The bug this demo exists to show: a naive CPU-side struct packs
// specularColour right after albedo's 3 floats (offset 12) and shininess
// at 24. std140 disagrees: a vec3 only consumes 12 bytes, but the NEXT
// 16-byte-aligned member is pushed up -- so specularColour really lives at
// offset 16 and shininess (a float, 4-byte alignment) packs after it at 28.
// THIS shader (compiled once, layout fixed forever) reads those std140
// offsets. Feed it the naive buffer and specularColour reads back as
// (specular.g, specular.b, shininess) -- scrambled -- while shininess
// reads the zero padding at 28. The demo's HUD shows the driver's own
// GL_UNIFORM_OFFSET answers as ground truth.
layout(std140) uniform MaterialBlock
{
    vec3 albedo;
    vec3 specularColour;
    float shininess;
};

uniform vec3 viewPos;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(lightPos.xyz - fragPosWorld);
    vec3 V = normalize(viewPos - fragPosWorld);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    // In the corrupted/naive case shininess reads 0, so max() clamps the
    // exponent to 1 -- the highlight loses its tight falloff and smears
    // across the whole lit side, tinted by the scrambled specularColour
    // (whose blue channel now holds the CPU-side shininess value, 64).
    float spec = pow(max(dot(N, H), 0.0), max(shininess, 1.0));

    vec3 ambient = 0.15 * albedo;
    vec3 diffuse = diff * albedo * lightColour.rgb;
    vec3 specular = spec * specularColour * lightColour.rgb;

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
