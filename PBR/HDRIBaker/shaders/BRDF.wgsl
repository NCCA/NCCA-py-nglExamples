// Split-sum BRDF integration, ported from shaders/BRDFFragment.glsl (itself
// a port of LearnOpenGL's Specular-IBL brdf.fs). Full-screen triangle VS
// generated from @builtin(vertex_index), no vertex buffer needed.

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) texCoords : vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index : u32) -> VSOut {
    var out : VSOut;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.texCoords = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

const PI : f32 = 3.14159265359;

fn RadicalInverse_VdC(bits_in : u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
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

fn GeometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
    // note that we use a different k for IBL
    let a = roughness;
    let k = (a * a) / 2.0;

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

fn IntegrateBRDF(NdotV : f32, roughness : f32) -> vec2<f32> {
    var V : vec3<f32>;
    V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

    var A = 0.0;
    var B = 0.0;

    let N = vec3<f32>(0.0, 0.0, 1.0);

    const SAMPLE_COUNT : u32 = 1024u;
    for (var i : u32 = 0u; i < SAMPLE_COUNT; i++) {
        let Xi = Hammersley(i, SAMPLE_COUNT);
        let H = ImportanceSampleGGX(Xi, N, roughness);
        let L = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = max(L.z, 0.0);
        let NdotH = max(H.z, 0.0);
        let VdotH = max(dot(V, H), 0.0);

        if NdotL > 0.0 {
            let G = GeometrySmith(N, V, L, roughness);
            let G_Vis = (G * VdotH) / (NdotH * NdotV);
            let Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);
    return vec2<f32>(A, B);
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec2<f32> {
    return IntegrateBRDF(in.texCoords.x, in.texCoords.y);
}
