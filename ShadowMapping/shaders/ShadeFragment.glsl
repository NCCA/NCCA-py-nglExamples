#version 330 core

in vec3 FragPosWorld;
in vec3 NormalWorld;
in vec4 FragPosLightSpace;

uniform vec3 lightDir;   // world-space direction *towards* the light
uniform vec4 Colour;
uniform sampler2D shadowMap;
uniform float bias;
uniform bool pcfEnabled;

out vec4 fragColour;

// Manual shadow-map read: NOT sampler2DShadow / textureProj on purpose --
// keeping the perspective divide, the [-1,1] -> [0,1] remap and the depth
// compare explicit is the point of this demo. GL_TEXTURE_COMPARE_MODE is
// left at NONE for the same reason (see main.py::_create_shadow_map).
float shadow_calc(vec4 fragPosLightSpace)
{
    vec3 proj = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj = proj * 0.5 + 0.5;

    // Outside the light's far plane (or, thanks to CLAMP_TO_BORDER with a
    // border colour of 1.0, outside the frustum in x/y): fully lit, not
    // shadowed. Without CLAMP_TO_BORDER here, GL_REPEAT would tile the
    // shadow map and paint bogus shadows outside the light frustum.
    if (proj.z > 1.0)
    {
        return 0.0;
    }

    float currentDepth = proj.z;
    float shadow = 0.0;

    if (pcfEnabled)
    {
        vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
        for (int x = -1; x <= 1; ++x)
        {
            for (int y = -1; y <= 1; ++y)
            {
                float pcfDepth = texture(shadowMap, proj.xy + vec2(x, y) * texel).r;
                shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;
            }
        }
        shadow /= 9.0;
    }
    else
    {
        float closestDepth = texture(shadowMap, proj.xy).r;
        shadow = (currentDepth - bias > closestDepth) ? 1.0 : 0.0;
    }
    return shadow;
}

void main()
{
    vec3 N = normalize(NormalWorld);
    vec3 L = normalize(lightDir);
    float diff = max(dot(N, L), 0.0);

    float shadow = shadow_calc(FragPosLightSpace);

    vec3 ambient = 0.18 * Colour.rgb;
    vec3 diffuse = (1.0 - shadow) * diff * Colour.rgb;

    fragColour = vec4(ambient + diffuse, Colour.a);
}
