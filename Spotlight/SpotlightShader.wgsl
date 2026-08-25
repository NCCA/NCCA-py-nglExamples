struct Uniforms {
    m: mat4x4<f32>,
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    viewer_pos: vec4<f32>,
};

struct SpotLight {
    position: vec4<f32>,
    direction: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    params: vec4<f32>,      // spotCosCutoff, spotCosInnerCutoff, spotExponent, unused
    attenuation: vec4<f32>, // constant, linear, quadratic, unused
};

struct Lights {
    lights: array<SpotLight, 4>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> lights: Lights;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) in_vert: vec3<f32>, @location(1) in_normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_position = (u.m * vec4<f32>(in_vert, 1.0)).xyz;
    out.normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

const material_ambient = vec4<f32>(0.274725, 0.1995, 0.0745, 1.0);
const material_diffuse = vec4<f32>(0.75164, 0.60648, 0.22648, 1.0);
const material_specular = vec4<f32>(0.628281, 0.555802, 0.3666065, 1.0);
const material_shininess = 51.2;

fn spot_light(i: u32, n: vec3<f32>, world_pos: vec3<f32>, eye_dir: vec3<f32>) -> vec4<f32> {
    let light = lights.lights[i];
    let to_light = light.position.xyz - world_pos;
    let d = length(to_light);
    let l = to_light / d;

    let attenuation = 1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d);

    let spot_dot = dot(-l, normalize(light.direction.xyz));
    var spot_attenuation = 0.0;
    if (spot_dot >= light.params.x) {
        let spot_value = smoothstep(light.params.x, light.params.y, spot_dot);
        spot_attenuation = pow(spot_value, light.params.z);
    }
    let total_attenuation = attenuation * spot_attenuation;

    let reflection = normalize(reflect(-l, n));
    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_r = max(0.0, dot(eye_dir, reflection));

    let diffuse = material_diffuse * light.diffuse * n_dot_l * total_attenuation;
    let specular = material_specular * light.specular * pow(n_dot_r, material_shininess) * total_attenuation;
    return diffuse + specular;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let eye_dir = normalize(u.viewer_pos.xyz - in.world_position);

    var colour = material_ambient * 0.2;
    for (var i = 0u; i < 4u; i = i + 1u) {
        colour = colour + spot_light(i, n, in.world_position, eye_dir);
    }
    colour.a = 1.0;
    return colour;
}
