#version 410 core

layout(location = 0) out vec4 fragColour;

in vec2 texCoord;
in vec3 fragmentNormal;
in vec3 eyeDirection;
in vec3 lightDir;
in vec3 halfVector;
in vec3 vPosition;

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};
struct Light
{
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};
uniform Material material;
uniform Light light;
uniform sampler2D diffuseTexture;

vec4 pointLight(vec4 texColour)
{
    vec3 N = normalize(fragmentNormal);
    vec3 L = normalize(lightDir);
    float lambertTerm = dot(N, L);

    vec4 ambient = material.ambient * light.ambient;
    vec4 diffuse = vec4(0.0);
    vec4 specular = vec4(0.0);

    if (lambertTerm > 0.0)
    {
        diffuse = texColour * light.diffuse * lambertTerm;
        vec3 halfV = normalize(halfVector);
        float ndotHV = max(dot(N, halfV), 0.0);
        specular = material.specular * light.specular * pow(ndotHV, material.shininess);
    }
    return ambient + diffuse + specular;
}

void main()
{
    fragColour = pointLight(texture(diffuseTexture, texCoord));
}
