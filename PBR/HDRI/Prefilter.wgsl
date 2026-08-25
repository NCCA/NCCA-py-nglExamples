// GGX-weighted specular prefilter, ported from shaders/PrefilterFragment.glsl
// (itself a port of LearnOpenGL's Specular-IBL prefilter.fs).

struct CaptureUniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    roughness : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
};
@group(0) @binding(0) var<uniform> capture : CaptureUniforms;
@group(1) @binding(0) var environmentMap : texture_cube<f32>;
@group(1) @binding(1) var environmentSampler : sampler;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) localPos : vec3<f32>,
};

@vertex
fn vertex_main(@location(0) inVert : vec3<f32>) -> VSOut {
    var out : VSOut;
    out.localPos = inVert;
    out.position = capture.projection * capture.view * vec4<f32>(inVert, 1.0);
    return out;
}

const PI : f32 = 3.14159265359;

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

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
fn RadicalInverse_VdC(bits_in : u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn Hammersley(i : u32, N : u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(N), RadicalInverse_VdC(i));
}

fn ImportanceSampleGGX(Xi : vec2<f32>, N : vec3<f32>, roughness : f32) -> vec3<f32> {
    let a = roughness * roughness;

    let phi = 2.0 * PI * Xi.x;
    let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    var H : vec3<f32>;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);

    let sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let N = normalize(in.localPos);
    let R = N;
    let V = R;
    let roughness = capture.roughness;

    const SAMPLE_COUNT : u32 = 1024u;
    var prefilteredColor = vec3<f32>(0.0);
    var totalWeight = 0.0;

    for (var i : u32 = 0u; i < SAMPLE_COUNT; i++) {
        let Xi = Hammersley(i, SAMPLE_COUNT);
        let H = ImportanceSampleGGX(Xi, N, roughness);
        let L = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = max(dot(N, L), 0.0);
        if NdotL > 0.0 {
            let D = DistributionGGX(N, H, roughness);
            let NdotH = max(dot(N, H), 0.0);
            let HdotV = max(dot(H, V), 0.0);
            let pdf = D * NdotH / (4.0 * HdotV) + 0.0001;

            let resolution = 512.0; // source cubemap per-face size (keep in sync with ENV_SIZE)
            let saTexel = 4.0 * PI / (6.0 * resolution * resolution);
            let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);

            var mipLevel = 0.5 * log2(saSample / saTexel);
            if roughness == 0.0 {
                mipLevel = 0.0;
            }

            prefilteredColor += textureSampleLevel(environmentMap, environmentSampler, L, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }

    prefilteredColor = prefilteredColor / totalWeight;
    return vec4<f32>(prefilteredColor, 1.0);
}
