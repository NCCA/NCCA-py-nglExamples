#version 330 core
// Cook-Torrance direct lighting is copied from PBR/SimplePBR/shaders/PBRFragment.glsl
// (https://learnopengl.com/PBR/Lighting), extended here with the split-sum
// image-based ambient term from https://learnopengl.com/PBR/IBL/Specular-IBL,
// which is itself Karis's derivation (B. Karis, "Real Shading in Unreal
// Engine 4", SIGGRAPH 2013, https://blog.selfshadow.com/publications/s2013-shading-course/).
//
// CUT STAGE: the brief's stage 2 (a roughness-mipped GGX-prefiltered
// specular chain) was not built -- see ibl_precompute.py's module
// docstring. `prefilteredColor` below is a single unblurred sample of the
// base environment cubemap along the reflection vector R, so specular
// reflections stay sharp at all roughness values instead of blurring out;
// everything else (diffuse irradiance, the BRDF LUT scale/bias) is real.
layout (location = 0) out vec4 fragColour;

in vec3 WorldPos;
in vec3 Normal;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

uniform samplerCube irradianceMap;
uniform samplerCube envMap;      // base (unfiltered) cubemap -- see CUT STAGE above
uniform sampler2D brdfLUT;
uniform bool useIBL;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
// Roughness-aware Fresnel used for the ambient term only (LearnOpenGL IBL
// chapter) -- clamps F0 so grazing angles on rough dielectrics don't blow
// out to pure white.
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
void main()
{
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - WorldPos);
    vec3 R = reflect(-V, N);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation (identical to SimplePBR)
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i)
    {
        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 nominator    = NDF * G * F;
        float denominator = 4.0 * max(dot(V, N), 0.0) * max(dot(L, N), 0.0) + 0.001;
        vec3 brdf = nominator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + brdf) * radiance * NdotL;
    }

    vec3 ambient;
    if (useIBL)
    {
        // diffuse IBL term: irradiance map is already the pre-integrated
        // cosine-weighted incoming radiance, so it plugs straight in as
        // the Lambertian term (no extra 1/PI -- that normalisation is
        // baked into the convolution, see ibl_precompute.py).
        vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);
        vec3 irradiance = texture(irradianceMap, N).rgb;
        vec3 diffuse = irradiance * albedo;

        // specular IBL term: split-sum approximation, prefiltered
        // environment sample x (F * A + B) from the BRDF LUT -- see the
        // CUT STAGE note above re: prefilteredColor.
        vec3 prefilteredColor = texture(envMap, R).rgb;
        vec2 envBRDF = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
        vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

        ambient = (kD * diffuse + specular) * ao;
    }
    else
    {
        // direct-only fallback, identical flat ambient hack to SimplePBR
        ambient = vec3(0.03) * albedo * ao;
    }

    vec3 color = ambient + Lo;

    // HDR tonemapping + gamma correction (unchanged from SimplePBR)
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    fragColour = vec4(color, 1.0);
}
