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

struct SpotLight
{
    vec3 position;
    vec3 direction;
    vec4 diffuse;
    vec4 specular;
    float spotCosCutoff;
    float spotCosInnerCutoff;
    float spotExponent;
    float constantAttenuation;
    float linearAttenuation;
    float quadraticAttenuation;
};

uniform Material material;
#define NUM_LIGHTS 4
uniform SpotLight light[NUM_LIGHTS];
uniform vec3 viewerPos;

vec4 spotLight(int i, vec3 n, vec3 eyeDir)
{
    vec3 toLight = light[i].position - worldPosition;
    float d = length(toLight);
    vec3 L = toLight / d;

    float attenuation = 1.0 / (light[i].constantAttenuation +
                                light[i].linearAttenuation * d +
                                light[i].quadraticAttenuation * d * d);

    float spotDot = dot(-L, normalize(light[i].direction));
    float spotAttenuation;
    if (spotDot < light[i].spotCosCutoff)
    {
        spotAttenuation = 0.0;
    }
    else
    {
        float spotValue = smoothstep(light[i].spotCosCutoff, light[i].spotCosInnerCutoff, spotDot);
        spotAttenuation = pow(spotValue, light[i].spotExponent);
    }
    attenuation *= spotAttenuation;

    vec3 reflection = normalize(reflect(-L, n));
    float nDotL = max(0.0, dot(n, L));
    float nDotR = max(0.0, dot(eyeDir, reflection));

    vec4 diffuse = material.diffuse * light[i].diffuse * nDotL * attenuation;
    vec4 specular = material.specular * light[i].specular * pow(nDotR, material.shininess) * attenuation;
    return diffuse + specular;
}

void main()
{
    vec3 n = normalize(fragmentNormal);
    vec3 eyeDir = normalize(viewerPos - worldPosition);

    fragColour = material.ambient * 0.2;
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        fragColour += spotLight(i, n, eyeDir);
    }
    fragColour.a = 1.0;
}
