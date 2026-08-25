struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = (u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let lambert = max(dot(n, light_dir), 0.0);
    let colour = u.colour.rgb * (0.3 + 0.7 * lambert);
    return vec4<f32>(colour, 1.0);
}
