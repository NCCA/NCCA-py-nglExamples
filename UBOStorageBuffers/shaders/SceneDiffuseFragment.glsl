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
// shininess right after albedo's 3 floats (offset 12). Because vec3 has a
// 16-byte base alignment in std140, THIS shader (compiled once, layout
// fixed forever) actually reads shininess from offset 16. Feed it the
// naive buffer and shininess reads back as 0 -- the specular highlight
// flattens out even though albedo still looks correct.
layout(std140) uniform MaterialBlock
{
    vec3 albedo;
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
    // shininess == 0 (the corrupted case) still works arithmetically --
    // pow(x, 0) == 1 -- so the highlight just stops falling off with angle
    // and paints every lit fragment full-strength specular white instead.
    float spec = pow(max(dot(N, H), 0.0), max(shininess, 1.0));

    vec3 ambient = 0.15 * albedo;
    vec3 diffuse = diff * albedo * lightColour.rgb;
    vec3 specular = spec * lightColour.rgb * step(1.0, shininess);

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
