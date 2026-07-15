// Physically Based Rendering with a full metallic/roughness texture set.
// A direct port of shaders/PBRVertex.glsl + shaders/PBRFragment.glsl to WGSL.
// Based on https://learnopengl.com/PBR/Lighting

const PI : f32 = 3.14159265359;

// @group(0) per-draw transforms, bound with a dynamic offset (one slot per object)
struct Transforms {
    MVP : mat4x4<f32>,
    M : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
    // texture rotation packed as a column-major mat2: (c, s, -s, c)
    texRot : vec4<f32>,
};
@group(0) @binding(0) var<uniform> transforms : Transforms;

// @group(1) scene lights and camera, shared by every draw this frame
struct Lights {
    lightPositions : array<vec4<f32>, 4>,
    lightColors : array<vec4<f32>, 4>,
    camPos : vec4<f32>,
};
@group(1) @binding(0) var<uniform> scene : Lights;

// @group(2) the active material's texture pack (see texture_pack_webgpu.py)
@group(2) @binding(0) var albedoMap : texture_2d<f32>;
@group(2) @binding(1) var normalMap : texture_2d<f32>;
@group(2) @binding(2) var metallicMap : texture_2d<f32>;
@group(2) @binding(3) var roughnessMap : texture_2d<f32>;
@group(2) @binding(4) var aoMap : texture_2d<f32>;
@group(2) @binding(5) var texSampler : sampler;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) TexCoords : vec2<f32>,
    @location(1) WorldPos : vec3<f32>,
    @location(2) Normal : vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) inVert : vec3<f32>,
    @location(1) inNormal : vec3<f32>,
    @location(2) inUV : vec2<f32>,
) -> VSOut {
    var out : VSOut;
    // rotate the uvs for a little visual variety, matching the OpenGL demo
    let texRotMat = mat2x2<f32>(transforms.texRot.xy, transforms.texRot.zw);
    out.TexCoords = texRotMat * inUV;
    out.WorldPos = (transforms.M * vec4<f32>(inVert, 1.0)).xyz;
    out.Normal = (transforms.normalMatrix * vec4<f32>(inNormal, 0.0)).xyz;
    out.position = transforms.MVP * vec4<f32>(inVert, 1.0);
    return out;
}

// Build a world-space normal from the tangent-space normal map. The tangent
// frame is derived from screen-space derivatives so the mesh needs no tangents.
fn getNormalFromMap(TexCoords : vec2<f32>, WorldPos : vec3<f32>, Normal : vec3<f32>) -> vec3<f32> {
    let tangentNormal = textureSample(normalMap, texSampler, TexCoords).xyz * 2.0 - 1.0;

    let Q1 = dpdx(WorldPos);
    let Q2 = dpdy(WorldPos);
    let st1 = dpdx(TexCoords);
    let st2 = dpdy(TexCoords);

    let N = normalize(Normal);
    let T = normalize(Q1 * st2.y - Q2 * st1.y);
    let B = -normalize(cross(N, T));
    let TBN = mat3x3<f32>(T, B, N);

    return normalize(TBN * tangentNormal);
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
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let albedo = pow(textureSample(albedoMap, texSampler, in.TexCoords).rgb, vec3<f32>(2.2));
    let metallic = textureSample(metallicMap, texSampler, in.TexCoords).r;
    let roughness = textureSample(roughnessMap, texSampler, in.TexCoords).r;
    let ao = textureSample(aoMap, texSampler, in.TexCoords).r;

    let N = getNormalFromMap(in.TexCoords, in.WorldPos, in.Normal);
    let V = normalize(scene.camPos.xyz - in.WorldPos);

    // F0 is 0.04 for dielectrics, the albedo for metals (metallic workflow)
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    var Lo = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let lightPos = scene.lightPositions[i].xyz;
        let L = normalize(lightPos - in.WorldPos);
        let H = normalize(V + L);
        let distance = length(lightPos - in.WorldPos);
        let attenuation = 1.0 / (distance * distance);
        let radiance = scene.lightColors[i].xyz * attenuation;

        // Cook-Torrance BRDF
        let NDF = DistributionGGX(N, H, roughness);
        let G = GeometrySmith(N, V, L, roughness);
        let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        let specular = numerator / denominator;

        let kS = F;
        // energy conservation, and only non-metals get diffuse lighting
        var kD = vec3<f32>(1.0) - kS;
        kD = kD * (1.0 - metallic);

        let NdotL = max(dot(N, L), 0.0);
        Lo = Lo + (kD * albedo / PI + specular) * radiance * NdotL;
    }

    let ambient = vec3<f32>(0.03) * albedo * ao;
    var color = ambient + Lo;

    // HDR tonemap then gamma correct
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
