struct GBuffer {
    position: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
};

@group(0) @binding(0) var g_position: texture_2d<f32>;
@group(0) @binding(1) var g_normal: texture_2d<f32>;
@group(0) @binding(2) var g_albedo: texture_2d<f32>;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};

struct LightUniforms {
    lights: array<Light, 20>,
    num_lights: u32,
};

@group(0) @binding(3) var<uniform> light_uniforms: LightUniforms;

struct ViewParams {
    camPos: vec3<f32>,
    exposure: f32,
};
@group(0) @binding(4) var<uniform> view: ViewParams;


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


@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}


@fragment
fn fragment_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let g_buffer = GBuffer(
        textureLoad(g_position, vec2<i32>(frag_coord.xy), 0),
        textureLoad(g_normal, vec2<i32>(frag_coord.xy), 0),
        textureLoad(g_albedo, vec2<i32>(frag_coord.xy), 0)
    );

    // Nothing was rasterised here, so the G-buffer still holds the clear value.
    // Bail out before normalize(vec3(0.0)) turns the whole pixel into NaN.
    if (dot(g_buffer.normal.xyz, g_buffer.normal.xyz) < 0.5) {
        return vec4<f32>(0.4, 0.4, 0.4, 1.0);
    }

    let N = normalize(g_buffer.normal.xyz);
    let V = normalize(view.camPos - g_buffer.position.xyz);
    let albedo = g_buffer.albedo.rgb;
    let metallic = g_buffer.albedo.a;
    let roughness = g_buffer.normal.a;
    let ao = g_buffer.position.a;

    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    var Lo = vec3<f32>(0.0);

    for (var i: u32 = 0u; i < light_uniforms.num_lights; i = i + 1u) {
        let L = normalize(light_uniforms.lights[i].position - g_buffer.position.xyz);
        let H = normalize(V + L);
        let distance = length(light_uniforms.lights[i].position - g_buffer.position.xyz);
        let attenuation = 1.0 / (distance * distance);
        let radiance = light_uniforms.lights[i].color * attenuation;

        // Cook-Torrance BRDF
        let NDF = distributionGGX(N, H, roughness);
        let G   = geometrySmith(N, V, L, roughness);
        let F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        let nominator    = NDF * G * F;
        let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        let brdf = nominator / denominator;

        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD *= 1.0 - metallic;

        let NdotL = max(dot(N, L), 0.0);

        Lo += (kD * albedo / PI + brdf) * radiance * NdotL;
    }

    let ambient = vec3<f32>(0.03) * albedo * ao;

    var color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3<f32>(1.0));
    // gamma correct
    color = pow(color, vec3<f32>(1.0/view.exposure));

    return vec4<f32>(color, 1.0);
}
