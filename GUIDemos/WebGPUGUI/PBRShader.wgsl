// Physically Based Rendering (PBR) shader, translated from GLSL.
// Original GLSL code based on https://learnopengl.com/#!PBR/Lighting

// Vertex Shader

struct TransformUniforms {
    MVP : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
    M : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> transforms : TransformUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.worldPos = (transforms.M * vec4<f32>(input.position, 1.0)).xyz;
    output.normal = normalize((transforms.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.position = transforms.MVP * vec4<f32>(input.position, 1.0);
    return output;
}

// Fragment Shader

struct MaterialParams {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
};
@group(1) @binding(0) var<uniform> material: MaterialParams;

struct LightParams {
    lightPosition: vec3<f32>,
    lightColor: vec3<f32>,
};
@group(1) @binding(1) var<uniform> light: LightParams;

struct ViewParams {
    camPos: vec3<f32>,
    exposure: f32,
};
@group(1) @binding(2) var<uniform> view: ViewParams;


const PI = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let nom = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let nom = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometrySchlickGGX(NdotV, roughness);
    let ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let V = normalize(view.camPos - in.worldPos);

    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, material.albedo, material.metallic);

    var Lo = vec3<f32>(0.0);
    // This is a single light, in a real system we would loop over lights
    let L = normalize(light.lightPosition - in.worldPos);
    let H = normalize(V + L);
    let distance = length(light.lightPosition - in.worldPos);
    let attenuation = 1.0 / (distance * distance);
    let radiance = light.lightColor * attenuation;

    // Cook-Torrance BRDF
    let NDF = distributionGGX(N, H, material.roughness);
    let G   = geometrySmith(N, V, L, material.roughness);
    let F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    let nominator    = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    let brdf = nominator / denominator;

    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - material.metallic;

    let NdotL = max(dot(N, L), 0.0);

    Lo += (kD * material.albedo / PI + brdf) * radiance * NdotL;

    let ambient = vec3<f32>(0.03) * material.albedo * material.ao;

    var color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3<f32>(1.0));
    // gamma correct
    color = pow(color, vec3<f32>(1.0/view.exposure));

    return vec4<f32>(color, 1.0);
}
