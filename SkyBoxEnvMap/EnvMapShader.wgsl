// Reflective / refractive teapot: reflect()/refract() are evaluated in
// *world* space (not view space) because the cubemap itself is sampled in
// world space, so this shader carries M and a world-space normal matrix
// rather than the usual MV.
struct Transform {
    mvp : mat4x4<f32>,
    model : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
};

// mode_ior: x = shading mode (0 reflect, 1 refract, 2 fresnel mix, 3 diffuse), y = IOR
struct Params {
    cam_pos : vec4<f32>,
    light_dir : vec4<f32>,
    mode_ior : vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform : Transform;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var envTexture : texture_cube<f32>;
@group(0) @binding(3) var envSampler : sampler;

fn extract_mat3x3(m : mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) world_pos : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
};

@vertex
fn vertex_main(input : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    let world = transform.model * vec4<f32>(input.position, 1.0);
    out.world_pos = world.xyz;
    out.world_normal = normalize(extract_mat3x3(transform.normal_matrix) * input.normal);
    out.position = transform.mvp * vec4<f32>(input.position, 1.0);
    return out;
}

struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
};

@fragment
fn fragment_main(input : FragmentInput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    let i = normalize(input.world_pos - params.cam_pos.xyz);
    let mode = i32(params.mode_ior.x);
    let ior = params.mode_ior.y;
    var colour : vec3<f32>;

    if (mode == 0) {
        let r = reflect(i, n);
        colour = textureSample(envTexture, envSampler, r).rgb;
    } else if (mode == 1) {
        let r = refract(i, n, 1.0 / ior);
        colour = textureSample(envTexture, envSampler, r).rgb;
    } else if (mode == 2) {
        let reflect_dir = reflect(i, n);
        let refract_dir = refract(i, n, 1.0 / ior);
        // Schlick's approximation
        let f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
        let cos_theta = max(dot(-i, n), 0.0);
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
        let refract_colour = textureSample(envTexture, envSampler, refract_dir).rgb;
        let reflect_colour = textureSample(envTexture, envSampler, reflect_dir).rgb;
        colour = mix(refract_colour, reflect_colour, fresnel);
    } else {
        let diffuse = max(dot(n, normalize(params.light_dir.xyz)), 0.0);
        colour = vec3<f32>(0.8, 0.75, 0.7) * (0.2 + 0.8 * diffuse);
    }

    return vec4<f32>(colour, 1.0);
}
