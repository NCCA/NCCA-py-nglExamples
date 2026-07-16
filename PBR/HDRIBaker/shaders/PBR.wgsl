// Split-sum image-based-lighting PBR shader for the WebGPU teapot grid.
// A direct port of shaders/PBRVertex.glsl + shaders/PBRFragment.glsl (the
// OpenGL Task 4 demo) to WGSL, following the vertex layout and bind-group
// conventions of PBR/PBRTexture/PBRTexture.wgsl. The direct Cook-Torrance
// term for the 4 analytic lights is identical to PBRTexture.wgsl; the
// ambient term adds the split-sum IBL lookup (irradiance_cube, prefilter_cube,
// brdf_lut) gated by `useIBL`, falling back to the same flat ambient hack
// used everywhere else in the PBR demos when it's off.

const PI : f32 = 3.14159265359;

// @group(0) per-draw transforms, bound with a dynamic offset (one slot per teapot)
struct Transforms {
    MVP : mat4x4<f32>,
    M : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
    // (metallic, roughness, ao, unused)
    material : vec4<f32>,
    // base colour in .rgb (.a unused)
    albedo : vec4<f32>,
};
@group(0) @binding(0) var<uniform> transforms : Transforms;

// @group(1) scene lights, camera and the IBL toggle, shared by every draw this frame
struct Scene {
    lightPositions : array<vec4<f32>, 4>,
    lightColors : array<vec4<f32>, 4>,
    camPos : vec4<f32>,
    useIBL : u32,
    // prefilter_mips - 1: the roughest mip the chain actually has. Follows the
    // loaded map set, which is free to be baked with a shorter chain.
    maxReflectionLod : f32,
};
@group(1) @binding(0) var<uniform> scene : Scene;

// @group(2) the baked split-sum IBL textures from Task 5's bake
@group(2) @binding(0) var irradianceMap : texture_cube<f32>;
@group(2) @binding(1) var prefilterMap : texture_cube<f32>;
@group(2) @binding(2) var brdfLUT : texture_2d<f32>;
@group(2) @binding(3) var iblSampler : sampler;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) WorldPos : vec3<f32>,
    @location(1) Normal : vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) inVert : vec3<f32>,
    @location(1) inNormal : vec3<f32>,
    @location(2) inUV : vec2<f32>,
) -> VSOut {
    var out : VSOut;
    out.WorldPos = (transforms.M * vec4<f32>(inVert, 1.0)).xyz;
    out.Normal = (transforms.normalMatrix * vec4<f32>(inNormal, 0.0)).xyz;
    out.position = transforms.MVP * vec4<f32>(inVert, 1.0);
    return out;
}

fn DistributionGGX(N : vec3<f32>, H : vec3<f32>, roughness : f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let nom = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

fn GeometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let nom = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn GeometrySmith(N : vec3<f32>, V : vec3<f32>, L : vec3<f32>, roughness : f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta : f32, F0 : vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Roughness-aware Fresnel used for the ambient term only (LearnOpenGL IBL
// chapter) -- clamps F0 so grazing angles on rough dielectrics don't blow
// out to pure white.
fn fresnelSchlickRoughness(cosTheta : f32, F0 : vec3<f32>, roughness : f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let albedo = transforms.albedo.rgb;
    let metallic = transforms.material.x;
    let roughness = transforms.material.y;
    let ao = transforms.material.z;

    let N = normalize(in.Normal);
    let V = normalize(scene.camPos.xyz - in.WorldPos);
    let R = reflect(-V, N);

    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    var Lo = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let lightPos = scene.lightPositions[i].xyz;
        let L = normalize(lightPos - in.WorldPos);
        let H = normalize(V + L);
        let distance = length(lightPos - in.WorldPos);
        let attenuation = 1.0 / (distance * distance);
        let radiance = scene.lightColors[i].xyz * attenuation;

        let NDF = DistributionGGX(N, H, roughness);
        let G = GeometrySmith(N, V, L, roughness);
        let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        let nominator = NDF * G * F;
        let denominator = 4.0 * max(dot(V, N), 0.0) * max(dot(L, N), 0.0) + 0.001;
        let brdf = nominator / denominator;

        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD = kD * (1.0 - metallic);

        let NdotL = max(dot(N, L), 0.0);
        Lo = Lo + (kD * albedo / PI + brdf) * radiance * NdotL;
    }

    var ambient : vec3<f32>;
    if (scene.useIBL != 0u) {
        // diffuse IBL term: the irradiance map is already the pre-integrated
        // cosine-weighted incoming radiance, so it plugs straight in as
        // the Lambertian term (no extra 1/PI -- that normalisation is
        // baked into the convolution).
        let F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
        let kS = F;
        let kD = (1.0 - kS) * (1.0 - metallic);
        let irradiance = textureSample(irradianceMap, iblSampler, N).rgb;
        let diffuse = irradiance * albedo;

        // specular IBL term: split-sum approximation, prefiltered
        // environment sample (roughness picks the mip) x (F * A + B) from
        // the BRDF LUT.
        let prefilteredColor = textureSampleLevel(
            prefilterMap, iblSampler, R, roughness * scene.maxReflectionLod
        ).rgb;
        let envBRDF = textureSample(
            brdfLUT, iblSampler, vec2<f32>(max(dot(N, V), 0.0), roughness)
        ).rg;
        let specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

        ambient = (kD * diffuse + specular) * ao;
    } else {
        // direct-only fallback, identical flat ambient hack to the rest of
        // the PBR demos.
        ambient = vec3<f32>(0.03) * albedo * ao;
    }

    var color = ambient + Lo;

    // HDR tonemapping + gamma correction
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
