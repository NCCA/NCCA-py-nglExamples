#version 410 core
in vec3 fragmentNormal;
in vec3 worldPosition;
layout(location = 0) out vec4 fragColour;

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};

struct Light
{
    vec3 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform Material material;
uniform vec3 viewerPos;
#define NUM_LIGHTS 3
uniform Light light[NUM_LIGHTS];

vec4 pointLight(int i, vec3 n, vec3 eyeDir)
{
    vec3 L = normalize(light[i].position - worldPosition);
    float lambert = max(dot(n, L), 0.0);
    vec4 diffuse = material.diffuse * light[i].diffuse * lambert;
    vec4 ambient = material.ambient * light[i].ambient;
    vec4 specular = vec4(0.0);
    if (lambert > 0.0)
    {
        vec3 halfV = normalize(eyeDir + L);
        float nDotHV = max(dot(n, halfV), 0.0);
        specular = material.specular * light[i].specular * pow(nDotHV, material.shininess);
    }
    return ambient + diffuse + specular;
}

void main()
{
    vec3 n = normalize(fragmentNormal);
    vec3 eyeDir = normalize(viewerPos - worldPosition);
    vec4 colour = vec4(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        colour += pointLight(i, n, eyeDir);
    }
    colour.a = 1.0;
    fragColour = colour;
}
